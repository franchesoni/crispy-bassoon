import shutil
import os
from pathlib import Path

import numpy as np
import torch

from IISS.create_segmentation import create_segmentation
from metrics import aggregate_metrics, compute_global_metrics, compute_tps_fps_tns_fns

def classify(seed_vectors, ann_is_pos, masks_feat_per_frame):
    """
    Uses the Nadaraya-Watson estimator to classify each one of the masks given the clicks. The clicks are tuples (frame, mask_index, label), e.g. they're already associated to a mask.
    `seed_vectors` is a list of vectors associated to all clicks so far
    """
    feats = torch.stack([mask_feat for masks_feat in masks_feat_per_frame for mask_feat in masks_feat])  # (N, F)
    seed_vectors = torch.stack(seed_vectors)  # (P, F)
    # compue distances
    distances = torch.cdist(feats[None], seed_vectors[None])[0]  # (N, P)

    # compute probabilities using radial basis function regression
    P, d = distances.shape[1], len(masks_feat_per_frame[0][0].flatten())
    r = (P**(1/(2*d))) * distances.max() / 10
    alphas = torch.exp(-(distances / r)**2/2)  # (N, P)
    denom = alphas.sum(dim=1) + 1e-15  # (N,)
    logits = (alphas * (2 * ann_is_pos[None] - 1) ).sum(axis=1) / denom.squeeze()  # (N,), convert labels to -1, 1
    labels = 2 * (logits > 0) - 1
    return labels.cpu().numpy()

def clean_clicks(clicks):
    """
    Removes all clicks that are overwritten later in the sequence.
    """
    coords = [(click[0], click[1]) for click in clicks]
    clicks = [clicks[i] for i in range(len(clicks)) if coords[i] not in coords[i+1:]]
    return clicks
    
def oracle_SAM(load_ind_img_mask_fn, precomputed_dir, n_images, runname='tmp', reset=False):
    precomputed_dir = Path(precomputed_dir)
    assert precomputed_dir.exists()
    ndigits_pre = len(os.listdir(precomputed_dir)[0].split('_')[2].split('.')[0])

    dstdir = Path(runname)
    try:
        dstdir.mkdir(parents=True)
    except FileExistsError:
        if reset:
            shutil.rmtree(dstdir)
            dstdir.mkdir()
            print('removed last run')
        else:
            print('run already exists and not resetting...')
            return
    
    metrics = []
    for i in range(n_images):
        global_ds_ind, img, gt = load_ind_img_mask_fn(i)
        sam_masks = np.load(precomputed_dir / f'sam_masks_{str(global_ds_ind).zfill(ndigits_pre)}.npy', allow_pickle=True)

        # for each mask, set the label to positive if more than half of it is contained in the ground truth
        labels = []
        for mask_dict in sam_masks:
            mask = mask_dict['segmentation']
            if np.sum(mask * gt) / np.sum(mask) > 0.5:
                labels.append(1)
            else:
                labels.append(-1)

        pred = create_segmentation([sam_masks], labels, [])
        metrics.append(compute_global_metrics(*compute_tps_fps_tns_fns(pred, [gt])))
    metrics = aggregate_metrics(metrics, None)
    # save to file
    with open(dstdir / 'metrics.json', 'w') as f:
        f.write(str(metrics).replace("'", '"'))
        



