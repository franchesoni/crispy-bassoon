import os
from config import datasets_path

os.environ["DETECTRON2_DATASETS"] = str(datasets_path)
import torch
import numpy as np
from PIL import Image
import tqdm
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

from mess.datasets.TorchvisionDataset import TorchvisionDataset, get_detectron2_datasets



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


def visualize(img, class_mask, pred, clicks, dstfile='tmp/current.png'):
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
    plt.imsave(dstfile, bigimg)
    return bigimg




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
    



def main(precomputed_dir, clicks_vectors_dir, dstdir, ds_name, seed):
    plot = False
    plotdir = 'tmp'
    dstdir = Path(dstdir) / ds_name
    clicks_vectors_dir = Path(clicks_vectors_dir)
    dstdir.mkdir(exist_ok=True, parents=True)
    # prepare directories
    precomputed_dir = Path(precomputed_dir)
    assert precomputed_dir.exists()
    assert ds_name in get_detectron2_datasets()
    print("running", ds_name)

    # get data
    ds = TorchvisionDataset(ds_name, transform=np.array, mask_transform=np.array)
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


    # start looping over classes
    metrics_per_class = {}
    for class_ind, class_name in zip(class_indices, class_names):
        if class_ind in values_to_ignore:
            continue
        if plot:
            (Path(plotdir) / class_name.replace(' ', '_')).mkdir(exist_ok=True, parents=True) 
        (dstdir / class_name).mkdir(exist_ok=True, parents=True)
        print("running class", class_name)
 
        clicks_so_far = np.load(clicks_vectors_dir / f"clicks_so_far_{class_name}_seed_{seed}.npy", allow_pickle=True)
        seed_vectors = np.load(clicks_vectors_dir / f"seed_vectors_{class_name}_seed_{seed}.npy", allow_pickle=True)
        # for each class, we'll save the clicks and metrics
        metrics = {}

        print('getting empty mask indices')
        empty_mask_indices = []
        for ind, sample in tqdm.tqdm(enumerate(ds), total=len(ds)):
            if (sample[1]==class_ind).sum() == 0:
                empty_mask_indices.append(ind)
        
        for n_clicks in 2**np.arange(int(np.log2(len(clicks_so_far)))+1):
            print(f'starting loop with {n_clicks} clicks')
            metrics[n_clicks] = []

            for sample_ind in tqdm.tqdm(range(len(ds))):
                # skip if empty
                if sample_ind in empty_mask_indices:
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
                pred = classify_patches(clicks_so_far[:n_clicks], seed_vectors[:n_clicks], feat_map)  # in order of certainty, param can be number of synthetic points (if int) or a score threshold (if float)
                Image.fromarray(pred).save(dstdir / class_name / f"sample_{sample_ind}_n_clicks_{n_clicks}_seed_{seed}.png")

                if plot:
                    visualize(img, class_mask, pred, [c for c in clicks_so_far if c[0] == sample_ind], dstfile=Path(plotdir) / class_name.replace(' ', '_') / f"sample_{sample_ind}_n_clicks_{n_clicks}.png")
                metric = get_metric(pred, ds_class_mask)
                metrics[n_clicks].append(metric)
                metrics_per_class[class_name] = metrics
            with open(dstdir / f'results_seed_{seed}.json', 'w') as f:
                f.write(str(metrics_per_class).replace("'", '"'))


if __name__ == "__main__":
    from fire import Fire
    Fire(main)
