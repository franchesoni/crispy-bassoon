import os
from config import datasets_path
os.environ["DETECTRON2_DATASETS"] = str(datasets_path)
import ast
import torch
import numpy as np
import tqdm
import cv2
import matplotlib.pyplot as plt
from pathlib import Path


from mess.datasets.TorchvisionDataset import TorchvisionDataset, get_detectron2_datasets


DINO_RESIZE = (644, 644)
DINO_PATCH_SIZE = 14


def norm(x):
    if x.max() == x.min():
        if x.max() != 0:
            RuntimeWarning(f"max and min are equal but x in nonzero, with constant at: {x.max()}")
        if x.max() == 255:
            x = x / 255
        else:
            RuntimeWarning(f"max and min are equal but x in nonzero, with constant at: {x.max()}")
        return x
    if x.dtype == bool:
        x = x * 1
    return (x - x.min()) / (x.max() - x.min())


def visualize(img, class_mask, pred, clicks):
    # resize pred to class_mask.size
    rpred = cv2.resize(
        np.array(pred*255).astype(np.uint8), class_mask.shape[::-1], interpolation=cv2.INTER_NEAREST
    )
    # draw clicks
    click_map = np.zeros_like(pred)
    for click in clicks:
        row, col = click[1:3]
        click_map[row, col] = 1 if click[3] else -1
    rclick_map = cv2.resize(
        np.array(click_map).astype(np.uint8), class_mask.shape[::-1], interpolation=cv2.INTER_NEAREST
    )
    # create big image
    H, W = img.shape[:2]
    bigimg = np.zeros((2*H, 2*W, 3))
    bigimg[:H, :W] = norm(img)
    bigimg[:H, W:] = norm(class_mask)[..., None] * np.array([1.0, 1.0, 1.0])[None, None]
    bigimg[H:, :W] = norm(rclick_map)[..., None] * np.array([1.0, 0.0, 1.0])[None, None]
    bigimg[H:, W:] = np.clip(norm(rpred)[..., None] * np.array([1.0, 0.0, 0.0])[None, None] + norm(class_mask)[..., None] * np.array([0., 1., 1])[None, None] + norm(rclick_map)[..., None] * np.array([0.5, 0.5, .5])[None, None], 0, 1)

    # visualize
    plt.imsave("tmp/current.png", bigimg)
    return bigimg


def to_numpy(x):
    return np.array(x)


def first_click(sample_ind, gt_mask, feat_map):
    click = (sample_ind, 46 // 2, 46 // 2, gt_mask[46 // 2, 46 // 2] > 0.5)
    seed_vector = get_click_vector(feat_map, click)
    return click, seed_vector


def get_click_vector(feat_map, click):
    """Interpolates the feature map to the image size and takes the vector of the clicked pixel"""
    sample_ind, row, col, _ = click
    return feat_map[row, col]


def get_corrective_click(sample_ind, pred, ds_gt_mask):
    # get the click that corrects the worst prediction. Because the gound truth is downsampled it should have values in [0, 1] corresponding to the overlap of each patch with the hr gt_mask. We then just compute the maximum absolute error between pred (binary) and ds_gt_mask.
    errors = np.abs(1*pred - norm(ds_gt_mask))
    if errors.max() == 0:
        return None
    max_errors = np.array(errors == errors.max())
    while True:
        eroded_max_errors = cv2.erode(max_errors.astype(np.uint8), np.ones((3, 3)))
        if eroded_max_errors.sum() == 0:
            break
        else:
            max_errors = eroded_max_errors
    # take the index of the worst prediction (max error)
    worst_pred_ind = np.unravel_index(np.argmax(max_errors), errors.shape)
    # build the click
    return (sample_ind, worst_pred_ind[0], worst_pred_ind[1], ds_gt_mask[worst_pred_ind]>127)


def classify_patches(
    clicks_so_far,
    seed_vectors,
    feat_map,
):  
    H, W, F = feat_map.shape
    if len(seed_vectors) == 0:
        return np.zeros((H,W)).astype(bool)
    feats = torch.from_numpy(feat_map.reshape(H * W, F))  # (HW, F) # flatten
    seed_vectors = torch.from_numpy(np.array(seed_vectors))  # (P, F)
    distances = torch.cdist(feats[None], seed_vectors[None])[0]  # (HW, P)
    ann_is_pos = torch.tensor([0 < click[3] for click in clicks_so_far])  # (P,)

    # compute probabilities using radial basis function regression
    P, d = distances.shape[1], feats.shape[1]  # and N=HW
    r = (P ** (1 / (2 * d))) * distances.max() / 10
    alphas = torch.exp(-((distances / r) ** 2) / 2)  # (N, P)
    denom = alphas.sum(dim=1) + 1e-15  # (N,)
    logits = (alphas * (2 * ann_is_pos[None] - 1)).sum(
        axis=1
    ) / denom.squeeze()  # (N,), convert labels to -1, 1

    # reshape to 46, 46
    logits = logits.reshape(H, W)
    return logits > 0


def get_metric(pred, ds_gt_mask):
    bin_ds_gt_mask = ds_gt_mask > 0.5
    # compute tp, fp, tn, fn
    tp = (pred & bin_ds_gt_mask).sum()
    fp = (pred & ~bin_ds_gt_mask).sum()
    tn = (~pred & ~bin_ds_gt_mask).sum()
    fn = (~pred & bin_ds_gt_mask).sum()
    # compute acc, jacc
    acc = (tp + tn) / (tp + tn + fp + fn)
    jacc = tp / (tp + fp + fn)
    # return everything in a dict
    return {"acc": acc, "jacc": jacc, "tp": tp, "fp": fp, "tn": tn, "fn": fn}
    



def main(precomputed_dir, dstdir, ds_name, seed):
    dev = False 
    plot = False
    dstdir = Path(dstdir) / ds_name
    dstdir.mkdir(exist_ok=True, parents=True)
    # prepare directories
    precomputed_dir = Path(precomputed_dir)
    assert precomputed_dir.exists()
    assert ds_name in get_detectron2_datasets()
    print("running", ds_name)

    # get data
    ds = TorchvisionDataset(ds_name, transform=to_numpy, mask_transform=to_numpy)
    class_indices, class_names = np.arange(len(ds.class_names)), ds.class_names
    n_digits = len(str(len(ds)))
    values_to_ignore = [255] + [
        ind
        for ind, cls_name in zip(class_indices, class_names)
        if cls_name
        in [
            "background",
            "others",
            "unlabeled",
            "background (waterbody)",
            "background or trash",
        ]
    ]
    ds_indices = np.arange(len(ds))
    np.random.seed(seed)
    np.random.shuffle(ds_indices)
    if dev:
        ds_indices = np.arange(len(ds))

    # start looping over classes
    if not (dstdir / f'results_seed_{seed}.json').exists():
        resuming = False
        metrics_after_per_class = {}
        metrics_before_per_class = {}
        sample_inds_per_class = {}
    else:
        with open(dstdir / f'results_seed_{seed}.json', 'r') as f:
            res = f.read()
        res = ast.literal_eval(res.replace('tensor', ''))
        metrics_after_per_class = res['metrics_after']
        metrics_before_per_class = res['metrics_before']
        sample_inds_per_class = res['sample_inds']
        resuming = True



    print('getting empty mask indices')
    empty_mask_indices_per_class = {ci: [] for ci in class_indices}
    for ind, sample in tqdm.tqdm(enumerate(ds), total=len(ds)):
        for class_ind in class_indices:
            if (sample[1]==class_ind).sum() == 0:
                empty_mask_indices_per_class[class_ind].append(ind)

    for class_ind, class_name in zip(class_indices, class_names):
        if class_ind in values_to_ignore:
            continue
        if resuming and class_ind in res['metrics_after']:
            print('skipping class', class_name)
            continue
        print("running class", class_name)
 
        # for each class, we'll save the clicks and metrics
        clicks_so_far, seed_vectors, metrics_before, metrics_after, sample_inds = [], [], [], [], []
        # just to help end the loop
        perfect_in_a_row = (
            0  # number of images in a row with perfectly classified patches
        )
        ds_indices_ind = 0

        empty_mask_indices = empty_mask_indices_per_class[class_ind]
        max_clicks = (len(ds)-len(empty_mask_indices))*5
        print(f'starting loop for {max_clicks} clicks')
        while True:
            if len(clicks_so_far) == max_clicks:
                print('reached max clicks, quitting loop')
                break
            if perfect_in_a_row >= 10:
                print('reached 10 perfect predictions in a row, quitting loop')
                break
            print(f"img number: {ds_indices_ind % len(ds_indices)}, clicks so far: {len(clicks_so_far)}, max clicks: {max_clicks}", end="\r")
            # skip if empty
            sample_ind = ds_indices[ds_indices_ind % len(ds_indices)]
            if sample_ind in empty_mask_indices:
                ds_indices_ind += 1
                continue
            # get sample
            img, gt_mask = ds[sample_ind]
            class_mask = gt_mask == class_ind
            ds_class_mask = cv2.resize(
                (class_mask * 255).astype(np.uint8),
                (46, 46),
                interpolation=cv2.INTER_LINEAR,
            )
            # load features
            feat_map = np.load(
                precomputed_dir
                / ds_name
                / "dino"
                / f"img_features_{str(sample_ind).zfill(n_digits)}.npy",
                allow_pickle=True,
            )
            # make prediction
            pred = classify_patches(clicks_so_far, seed_vectors, feat_map)  # in order of certainty, param can be number of synthetic points (if int) or a score threshold (if float)

            if plot:
                visualize(img, class_mask, pred, [c for c in clicks_so_far if c[0] == sample_ind])
            metric = get_metric(pred, ds_class_mask)
            metrics_before.append(metric)

            # get new click
            click = get_corrective_click(sample_ind, pred, ds_class_mask)
            if click is not None:  # proposed points are wrong, correct one
                seed_vector = get_click_vector(feat_map, click)
                perfect_in_a_row = 0
                clicks_so_far.append(click)
                seed_vectors.append(seed_vector)
                ## compute metric
                pred = classify_patches(clicks_so_far, seed_vectors, feat_map)
                if plot:
                    visualize(
                    img,
                    class_mask,
                    pred,
                    [c for c in clicks_so_far if c[0] == sample_ind],
                )
                metric = get_metric(pred, ds_class_mask)
                metrics_after.append(metric)
            else:
                perfect_in_a_row += 1
                metrics_after.append(metric)
            sample_inds.append(sample_ind)

            # advance next image
            ds_indices_ind += 1


            metrics_after_per_class[class_name] = metrics_after
            metrics_before_per_class[class_name] = metrics_before
            sample_inds_per_class[class_name] = sample_inds
            res = {
                "metrics_after": metrics_after_per_class,
                "metrics_before": metrics_before_per_class,
                "sample_inds": sample_inds_per_class,
            }
        with open(dstdir / f'results_seed_{seed}.json', 'w') as f:
                f.write(str(res).replace("'", '"'))
        np.save(dstdir / f"clicks_so_far_{class_name}_seed_{seed}.npy", clicks_so_far,)
        np.save(dstdir / f"seed_vectors_{class_name}_seed_{seed}.npy", seed_vectors, )


if __name__ == "__main__":
    from fire import Fire
    Fire(main)
