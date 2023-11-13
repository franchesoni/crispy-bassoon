import os
from config import datasets_path
os.environ["DETECTRON2_DATASETS"] = str(datasets_path)
import ast
import torch
import numpy as np
import time
import tqdm
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from numba import jit


from mess.datasets.TorchvisionDataset import TorchvisionDataset, get_detectron2_datasets


DINO_RESIZE = (644, 644)
DINO_PATCH_SIZE = 14

def compute_soft_mask(class_mask, masks, logits):
    soft_mask = np.zeros(class_mask.shape, dtype=float)
    for mask, logit in zip(masks, logits):
        soft_mask += mask['segmentation'] * logit
    return soft_mask

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


def visualize(img, class_mask, pred):
    # create big image
    H, W = img.shape[:2]
    bigimg = np.zeros((2*H, 2*W, 3))
    bigimg[:H, :W] = norm(img)
    bigimg[:H, W:] = norm(class_mask)[..., None] * np.array([1.0, 1.0, 1.0])[None, None]
    bigimg[H:, :W] = norm(pred)[..., None] * np.array([1.0, 1.0, 1.0])[None, None]
    bigimg[H:, W:] = np.clip(norm(pred)[..., None] * np.array([1.0, 0.0, 0.0])[None, None] + norm(class_mask)[..., None] * np.array([0., 1., 1])[None, None], 0,1 )

    # visualize
    plt.imsave("tmp/current.png", bigimg)
    breakpoint()
    return bigimg


def to_numpy(x):
    return np.array(x)



def get_inter_union(y_true, y_pred):
    return np.logical_and(y_true, y_pred).sum(), np.logical_or(y_true, y_pred).sum()


def get_clicked_segment(pred_mask, sam_masks, gt_mask):  
    """Based on IoU, return when no more improvement is possible. Cycles could happen."""
    max_iou_improvement = -np.inf  # Initialize to negative infinity
    chosen_mask_ind, is_pos = None, None  # Initialize variables
    pred_mask = pred_mask > 0
    
    current_inter, current_union = get_inter_union(gt_mask, pred_mask)
    current_iou = current_inter / current_union
    
    for mask_ind, mask in enumerate(sam_masks):
        # check if this mask can give us a relevant improvement
        max_possible_new_IoU_pos = min((current_inter + mask['area']), current_union) / current_union
        max_possible_improvement_pos = max_possible_new_IoU_pos - current_iou
        max_possible_new_IoU_neg = current_inter / max(current_inter, (current_union - mask['area'])) if current_inter > 0 else 0
        max_possible_improvement_neg = max_possible_new_IoU_neg - current_iou
        max_possible_improvement = max(max_possible_improvement_pos, max_possible_improvement_neg)
        if max_possible_improvement <= max_iou_improvement:
            continue

        # if not, compute the improvement it gives when clicked (pos and neg)
        seg = mask['segmentation']
        # Positive mask combination
        new_pos_pred = np.logical_or(pred_mask, seg)
        new_pos_inter, new_pos_union = get_inter_union(gt_mask, new_pos_pred)
        new_pos_iou = new_pos_inter / new_pos_union
        iou_improvement = new_pos_iou - current_iou
        
        if iou_improvement > max_iou_improvement:
            max_iou_improvement = iou_improvement
            chosen_mask_ind = mask_ind
            is_pos = True
            
        # Negative mask combination
        new_neg_pred = np.logical_and(pred_mask, np.logical_not(seg))
        new_neg_inter, new_neg_union = get_inter_union(gt_mask, new_neg_pred)
        new_neg_iou = new_neg_inter / new_neg_union
        iou_improvement = new_neg_iou - current_iou
        
        if iou_improvement > max_iou_improvement:
            max_iou_improvement = iou_improvement
            chosen_mask_ind = mask_ind
            is_pos = False
            
    if max_iou_improvement <= 0:
        return None
    
    return (chosen_mask_ind, is_pos)




def classify_masks(
    clicks_so_far,
    seed_vectors,
    mask_feats,
):  
    M, F = mask_feats.shape
    if len(seed_vectors) == 0:
        return np.zeros(M).astype(bool)
    feats = torch.from_numpy(mask_feats)  # (M, F) # flatten
    seed_vectors = torch.from_numpy(np.array(seed_vectors))  # (P, F)
    distances = torch.cdist(feats[None], seed_vectors[None])[0]  # (M, P)
    ann_is_pos = torch.tensor([0 < click[2] for click in clicks_so_far])  # (P,)

    # compute probabilities using radial basis function regression
    P, d = distances.shape[1], feats.shape[1]  
    r = (P ** (1 / (2 * d))) * distances.max() / 10
    alphas = torch.exp(-((distances / r) ** 2) / 2)  # (M, P)
    denom = alphas.sum(dim=1) + 1e-15  # (M,)
    logits = (alphas * (2 * ann_is_pos[None] - 1)).sum(
        axis=1
    ) / denom.squeeze()  # (N,), convert labels to -1, 1
    return logits 


def get_metric(soft_pred, class_mask):
    mae = np.abs(soft_pred - class_mask).mean()
    bin_gt_mask = class_mask > 0.5
    hard_pred = soft_pred > 0
    # compute tp, fp, tn, fn
    tp = (hard_pred & bin_gt_mask).sum()
    fp = (hard_pred & ~bin_gt_mask).sum()
    tn = (~hard_pred & ~bin_gt_mask).sum()
    fn = (~hard_pred & bin_gt_mask).sum()
    # compute acc, jacc
    acc = (tp + tn) / (tp + tn + fp + fn)
    jacc = tp / (tp + fp + fn)
    # return everything in a dict
    return {"acc": acc, "jacc": jacc, "tp": tp, "fp": fp, "tn": tn, "fn": fn, 'mae': mae}
    



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
        if res == '':
            resuming = False
            metrics_after_per_class = {}
            metrics_before_per_class = {}
            sample_inds_per_class = {}
        else:
            res = eval(res.replace('tensor', '').replace('nan', "float('nan')"))
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
        friendly_class_name = class_name.replace(' ', '_').replace('/', '-')
        if resuming and (class_name in res['metrics_after'] or friendly_class_name in res['metrics_after']):
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
            t0 = time.time()
            # skip if empty
            sample_ind = ds_indices[ds_indices_ind % len(ds_indices)]
            if sample_ind in empty_mask_indices:
                ds_indices_ind += 1
                continue
            # get sample
            img, gt_mask = ds[sample_ind]
            class_mask = gt_mask == class_ind

            t1 = time.time()
            # load masks
            masks = np.load(precomputed_dir / ds_name / "sam" / f"sam_masks_{str(sample_ind).zfill(n_digits)}.npy", allow_pickle=True)
            # load features
            mask_feats = np.load(
                precomputed_dir
                / ds_name
                / "dinosam"
                / f"dinosam_feats_{str(sample_ind).zfill(n_digits)}.npy",
                allow_pickle=True,
            )
            t2 = time.time()
            # make prediction
            logits = np.array(classify_masks(clicks_so_far, seed_vectors, mask_feats))
            t3 = time.time()
            # compute soft mask
            soft_mask = compute_soft_mask(class_mask, masks, logits)
            # soft_mask = (np.array([m['segmentation'] for m in masks]) * np.array(logits.reshape(-1, 1, 1))).sum(axis=0)

            t4 = time.time()
            if plot:
                visualize(img, class_mask, soft_mask)
            metric = get_metric(soft_mask, class_mask)
            # print('metric:', metric)
            metrics_before.append(metric)

            t5 = time.time()
            # get new click
            click = get_clicked_segment(soft_mask, masks, class_mask)
            click = (sample_ind, *click) if click is not None else None
            t6 = time.time()
            if click is not None:  # proposed points are wrong, correct one
                seed_vector = mask_feats[click[1]]
                perfect_in_a_row = 0
                clicks_so_far.append(click)
                seed_vectors.append(seed_vector)
                ## compute metric
                logits = classify_masks(clicks_so_far, seed_vectors, mask_feats)
                soft_mask = (np.array([m['segmentation'] for m in masks]) * logits.reshape(-1, 1, 1).numpy()).sum(axis=0)
                if plot:
                    visualize(
                    img,
                    class_mask,
                    soft_mask
                )
                metric = get_metric(soft_mask, class_mask)
                # print('metric:', metric)
                metrics_after.append(metric)
            else:
                perfect_in_a_row += 1
                metrics_after.append(metric)
            t7 = time.time()
            sample_inds.append(sample_ind)

            # advance next image
            ds_indices_ind += 1


            metrics_after_per_class[friendly_class_name] = metrics_after
            metrics_before_per_class[friendly_class_name] = metrics_before
            sample_inds_per_class[friendly_class_name] = sample_inds
            res = {
                "metrics_after": metrics_after_per_class,
                "metrics_before": metrics_before_per_class,
                "sample_inds": sample_inds_per_class,
            }
            t8 = time.time()
            # print elapsed times
            #print(f"t1: {t1-t0:.2f}, t2: {t2-t1:.2f}, t3: {t3-t2:.2f}, t4: {t4-t3:.2f}, t5: {t5-t4:.2f}, t6: {t6-t5:.2f}, t7: {t7-t6:.2f}, t8: {t8-t7:.2f}")
            

        with open(dstdir / f'results_seed_{seed}.json', 'w') as f:
                f.write(str(res).replace("'", '"'))
        np.save(dstdir / f"clicks_so_far_{friendly_class_name}_seed_{seed}.npy", clicks_so_far,)
        np.save(dstdir / f"seed_vectors_{friendly_class_name}_seed_{seed}.npy", seed_vectors, )


if __name__ == "__main__":
    from fire import Fire
    Fire(main)
