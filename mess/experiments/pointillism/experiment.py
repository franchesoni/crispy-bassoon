
import os
from config import datasets_path
os.environ['DETECTRON2_DATASETS'] = str(datasets_path)
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.feature import peak_local_max


from mess.datasets.TorchvisionDataset import TorchvisionDataset, get_detectron2_datasets


DINO_RESIZE = (644, 644)
DINO_PATCH_SIZE = 14

def norm(x):
    return (x - x.min()) / (x.max() - x.min())

def visualize(img, gt_mask, synthetic_points, dstdir='tmp'):
    """
    Save the img with the mask overlapping and the synthetic points on top.
    The synthetic points are tuples with (sample_ind, row, col, logit).
    """
    sample_ind = 0
    if not os.path.exists(dstdir):
        os.makedirs(dstdir)
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(np.clip(norm(img) + gt_mask[..., None] * np.array([1., 0., 1.])[None, None] * 0.3, 0, 1))
    for point in synthetic_points:
        sample_ind, row, col, logit = point
        # get the color according to the logit. Logit = -1 is red   Logit = 1 is blue
        if str(logit) in ['True', 'False']:
            logit = 1 if logit else -1
        color = (max(0, -logit), 0, max(0, logit))
        ax.scatter(col, row, color=color, s=100, marker='o')
    plt.savefig(os.path.join(dstdir, f'current.png'))
    plt.close()
    print('showed image', sample_ind)
    breakpoint()
    pass


def to_numpy(x):
    return np.array(x)

def click_center(sample_ind, img, gt_mask, feat_map):
  click = (sample_ind, gt_mask.shape[0] // 2, gt_mask.shape[1] // 2, gt_mask[gt_mask.shape[0] // 2, gt_mask.shape[1] // 2])
  seed_vector = get_click_vector(feat_map, click, img.shape)
  return click, seed_vector


def get_click_vector(feat_map, click, img_shape):
    """Interpolates the feature map to the image size and takes the vector of the clicked pixel"""
    sample_ind, row, col, _ = click
    row_pct, col_pct = row / img_shape[0], col / img_shape[1]

    pixel_height = (1 / feat_map.shape[0])
    pixel_width = (1 / feat_map.shape[1])
    feat_row_pct = row_pct / pixel_height - pixel_height / 2
    feat_col_pct = col_pct / pixel_width - pixel_width / 2

    topleft = max(np.floor(feat_row_pct), 0), max(np.floor(feat_col_pct), 0)
    bottomright = min(np.ceil(feat_row_pct), feat_map.shape[0]-1), min(np.ceil(feat_col_pct), feat_map.shape[1]-1)
    topright = topleft[0], bottomright[1]
    bottomleft = bottomright[0], topleft[1]

    topleft_val = feat_map[int(topleft[0]), int(topleft[1])]
    topright_val = feat_map[int(topright[0]), int(topright[1])]
    bottomleft_val = feat_map[int(bottomleft[0]), int(bottomleft[1])]
    bottomright_val = feat_map[int(bottomright[0]), int(bottomright[1])]

    # interpolate
    tl_weight = np.abs(feat_row_pct - topleft[0]) + np.abs(feat_col_pct - topleft[1])
    tr_weight = np.abs(feat_row_pct - topright[0]) + np.abs(feat_col_pct - topright[1])
    bl_weight = np.abs(feat_row_pct - bottomleft[0]) + np.abs(feat_col_pct - bottomleft[1])
    br_weight = np.abs(feat_row_pct - bottomright[0]) + np.abs(feat_col_pct - bottomright[1])

    feat_vector = (tl_weight * topleft_val + tr_weight * topright_val + bl_weight * bottomleft_val + br_weight * bottomright_val) / 4
    return feat_vector


def get_corrective_click(sample_ind, synthetic_points, gt_mask):
    for point in synthetic_points:
        if (point[3]>0) != gt_mask[point[1], point[2]]:
            return (*point[:3], gt_mask[point[1], point[2]])
    return None





def synthesize_points(sample_ind, clicks_so_far, seed_vectors, feat_map, MIN_DISTANCE=3, resize_factors=(1,1)):  # in order of certainty, param can be number of synthetic points (if int) or a score threshold (if float)
    H, W, F = feat_map.shape
    feats = torch.from_numpy(feat_map.reshape(H*W, F)) # (HW, F) # flatten
    seed_vectors = torch.from_numpy(np.array(seed_vectors))  # (P, F)
    distances = torch.cdist(feats[None], seed_vectors[None])[0]  # (HW, P)
    ann_is_pos = torch.tensor([0<click[3] for click in clicks_so_far])  # (P,)

    # compute probabilities using radial basis function regression
    P, d = distances.shape[1], feats.shape[1]  # and N=HW
    r = (P**(1/(2*d))) * distances.max() / 10
    alphas = torch.exp(-(distances / r)**2/2)  # (N, P)
    denom = alphas.sum(dim=1) + 1e-15  # (N,)
    logits = (alphas * (2 * ann_is_pos[None] - 1) ).sum(axis=1) / denom.squeeze()  # (N,), convert labels to -1, 1
    certainty = np.array(torch.abs(logits))  # because logits are weighted averages of -1 and +1, taking abs is between 0 and 1

    # add some noise to break ties, this noise doesn't modify local extrema
    certainty = certainty + np.random.randn(*certainty.shape) * np.diff(np.sort(np.unique(certainty))).min() / 10
    cer_img = certainty.reshape(H, W)
    local_extrema = peak_local_max(np.array(cer_img*255).astype(np.uint8), min_distance=MIN_DISTANCE)
    
    # sort according to certainties
    local_extrema_certainties = np.take(certainty, np.ravel_multi_index(local_extrema.T, cer_img.shape)) # the output array of shape (N,)
    local_extrema_certainties_indices = np.argsort(local_extrema_certainties)[::-1]  # decreasing
    local_extrema = local_extrema[local_extrema_certainties_indices]  # (N, 3) <- each row has (frame_ind, row, col)
    local_extrema = np.concatenate((sample_ind*np.ones((local_extrema.shape[0], 1), dtype=int), local_extrema), axis=1)

    # filter out inputs (we're using only one click)
    sclicks_coords = np.array([c[:3] for c in clicks_so_far])  # only one click
    is_input = np.all(sclicks_coords[None] == local_extrema[:, None], axis=2).any(axis=1)
    local_extrema = local_extrema[~is_input]

    # add labels for each point
    local_extrema_logits = np.take(logits.ravel(), np.ravel_multi_index(local_extrema[:, 1:].T, cer_img.shape)) 
    local_extrema = np.concatenate([local_extrema, local_extrema_logits[:, None]], axis=1)
    rowf, colf = resize_factors
    synthetic_points = [(int(c[0]),
        int((c[1]*DINO_PATCH_SIZE+(DINO_PATCH_SIZE/2)) / rowf),
        int((c[2]*DINO_PATCH_SIZE+(DINO_PATCH_SIZE/2)) / colf),
        c[3]
        ) for c in local_extrema]

    # keep from five points up to the threshold
    if np.array(synthetic_points)[np.abs(np.array(synthetic_points)[:, 3])>0.5].shape[0] > 5:
        synthetic_points = [sp for ind, sp in enumerate(synthetic_points) if (np.abs(np.array(synthetic_points)[:, 3])>0.5)[ind]]
    else:
        synthetic_points = synthetic_points[:5]
    return synthetic_points








def get_metric(sample_ind, clicks_so_far, gt_mask):
    correct, total = 0, 0
    for click in clicks_so_far:
        if sample_ind == click[0]:
            total += 1
            if gt_mask[click[1], click[2]] == click[3]:
                correct += 1
    acc = correct / total if total else 0
    print('accuracy:', acc)
    return acc


def main(precomputed_dir, dstdir, ds_name, seed):
    precomputed_dir = Path(precomputed_dir)
    assert precomputed_dir.exists()
    assert ds_name in get_detectron2_datasets()
    print('running', ds_name)

    ds = TorchvisionDataset(ds_name, transform=to_numpy, mask_transform=to_numpy)
    class_indices, class_names = np.arange(len(ds.class_names)), ds.class_names
    n_digits = len(str(len(ds)))
    values_to_ignore = [255] + [ind for ind, cls_name in zip(class_indices, class_names) if cls_name in ['background', 'others', 'unlabeled', 'background (waterbody)', 'background or trash']]
    max_clicks = len(ds)  # roughly one click per image

    ds_indices = np.arange(len(ds))
    np.random.seed(seed)
    np.random.shuffle(ds_indices)

    metrics_per_class = {}
    for class_ind, class_name in zip(class_indices, class_names):
        if class_ind in values_to_ignore:
            continue

        clicks_so_far, seed_vectors, metrics = [], [], []
        perfect_in_a_row = 0  # number of images in a row with perfectly synthetised clicks

        # FIRST CLICK
        ## load first sample
        ds_indices_ind = 0
        sample_ind = ds_indices[ds_indices_ind % len(ds_indices)]
        img, gt_mask = ds[sample_ind]
        class_mask = gt_mask == class_ind
        feat_map = np.load(precomputed_dir / ds_name / 'dino' / f'img_features_{str(sample_ind).zfill(n_digits)}.npy', allow_pickle=True)

        ## make first click (in the center)
        click, seed_vector = click_center(sample_ind, img, class_mask, feat_map)

        ## save first click
        clicks_so_far.append(click)
        visualize(img, class_mask, clicks_so_far, dstdir='tmp')
        seed_vectors.append(seed_vector)
        ## compute metric
        metric = get_metric(sample_ind, clicks_so_far, class_mask)
        metrics.append(metric)
        ## advance to next image
        ds_indices_ind += 1

        while not len(clicks_so_far) == max_clicks and perfect_in_a_row < len(ds):
            print(f'img number: {ds_indices_ind}', end = '\r')
            # get sample
            sample_ind = ds_indices[ds_indices_ind % len(ds_indices)]
            img, gt_mask = ds[sample_ind]
            class_mask = gt_mask == class_ind
            if class_mask.sum() == 0:
                ds_indices_ind += 1
                continue
            feat_map = np.load(precomputed_dir / ds_name / 'dino' / f'img_features_{str(sample_ind).zfill(n_digits)}.npy', allow_pickle=True)
            # get points
            rowf, colf = DINO_RESIZE[0] / img.shape[0], DINO_RESIZE[1] / img.shape[1]
            synthetic_points = synthesize_points(sample_ind, clicks_so_far, seed_vectors, feat_map, MIN_DISTANCE=3, resize_factors=(rowf, colf))  # in order of certainty, param can be number of synthetic points (if int) or a score threshold (if float)


            assert len(synthetic_points), 'no synthesized points! Fix your synthesizer or put center clicks'
            # get the click that corrects the most certain synthetic point

            visualize(img, class_mask, synthetic_points, dstdir='tmp')

            click = get_corrective_click(sample_ind, synthetic_points, class_mask)
            if click is not None:  # proposed points are wrong, correct one
                seed_vector = get_click_vector(feat_map, click, img.shape)
                perfect_in_a_row = 0
                clicks_so_far.append(click)
                seed_vectors.append(seed_vector)
                ## compute metric
                metric = get_metric(sample_ind, synthetic_points, class_mask)
                metrics.append(metric)
            else:
                perfect_in_a_row += 1
                metrics.append(1)

            # advance next image
            ds_indices_ind += 1
            visualize(img, class_mask, [c for c in clicks_so_far if c[0] == sample_ind], dstdir='tmp')

            synthetic_points = synthesize_points(sample_ind, clicks_so_far, seed_vectors, feat_map, MIN_DISTANCE=3, resize_factors=(rowf, colf))  # in order of certainty, param can be number of synthetic points (if int) or a score threshold (if float)
            visualize(img, class_mask, synthetic_points, dstdir='tmp')

        metrics_per_class[class_name] = metrics
        

        



if __name__ == '__main__':
    from fire import Fire
    Fire(main)
    