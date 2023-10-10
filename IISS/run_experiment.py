import shutil
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from IISS.extract_masks import extract_masks_list
from IISS.compute_features import compute_features_list
from IISS.project_masks import project_masks
from IISS.classify import classify
from IISS.create_segmentation import create_segmentation

from metrics import compute_tps_fps_tns_fns, compute_global_metrics


def get_clicked_segment(pred_masks, dinosam_masks, gt_masks):  # optimize for accuracy
    max_err_reduction = -np.inf
    for frame_ind, fmasks in enumerate(dinosam_masks):
        pred_mask = pred_masks[frame_ind]
        gt_mask = gt_masks[frame_ind]
        current_err = np.logical_xor(pred_mask, gt_mask).sum()
        for mask_ind, mask in enumerate(fmasks):
            if mask['area'] <= max_err_reduction:  # can't reduce error more
                continue
            seg = mask['segmentation']
            new_pos_pred = np.logical_or(pred_mask, seg)
            new_pos_error = np.logical_xor(new_pos_pred, gt_mask).sum()
            err_reduction = current_err - new_pos_error
            if err_reduction > max_err_reduction:
                max_err_reduction = err_reduction
                chosen_frame_ind = frame_ind
                chosen_mask_ind = mask_ind
                is_pos = True
            # although I thought positive error reduction implies negative error increase, this is false, so we have to check
            new_neg_pred = np.logical_and(pred_mask, np.logical_not(seg))
            new_neg_error = np.logical_xor(new_neg_pred, gt_mask).sum()
            err_reduction = current_err - new_neg_error
            if err_reduction > max_err_reduction:
                max_err_reduction = err_reduction
                chosen_frame_ind = frame_ind
                chosen_mask_ind = mask_ind
                is_pos = False
    return (chosen_frame_ind, chosen_mask_ind, is_pos)




def run_experiment(load_sample_fn, n_images, max_total_clicks, runname):
    MAX_CLICKS = max_total_clicks // n_images
    N = n_images
    noplt = False
    dstdir = f'runs/{runname}'
    dstdir = Path(dstdir)
    try:
        dstdir.mkdir(parents=True)
    except FileExistsError:
        shutil.rmtree(dstdir)
        dstdir.mkdir()
        print('removed last run')

    # load images and gt_masks
    print('loading images')
    images, gt_masks = [], []
    for i in range(N):
        img, gt = load_sample_fn(i)
        images.append(img)
        gt_masks.append(gt)

    
    if Path('state.npy').exists():
        state = np.load('state.npy', allow_pickle=True).item()
        sam_masks_per_frame = state['sam_masks_per_frame']
        masks_feat_per_frame = state['masks_feat_per_frame']     
        features = state['features']
    else:
        # compute SAM masks
        print('computing SAM masks')
        sam_masks_per_frame = extract_masks_list(images)
        # compute features
        print('computing features')
        features = compute_features_list(images)
        # project masks
        print('projecting masks')
        masks_feat_per_frame = []
        for frame_ind, feat in enumerate(features):
            masks_feat = project_masks(sam_masks_per_frame[frame_ind], feat)
            masks_feat_per_frame.append(masks_feat)
        to_save = {'sam_masks_per_frame': sam_masks_per_frame,
            'masks_feat_per_frame': masks_feat_per_frame,
            'features': features}
        np.save('state.npy', to_save)

    if not noplt:
        (dstdir / 'sam_masks').mkdir()
        print('saving imgs and masks...')
        for frame_ind, frame_masks in enumerate(sam_masks_per_frame):
            if frame_ind == 8:
                break
            plt.imsave(dstdir / 'sam_masks' / f'img_{str(frame_ind).zfill(2)}.png', images[frame_ind])
            plt.imsave(dstdir / 'sam_masks' / f'gt_{str(frame_ind).zfill(2)}.png', gt_masks[frame_ind])
            for mask_ind, mask in enumerate(frame_masks):
                plt.imsave(dstdir / 'sam_masks' / f'mask_{str(frame_ind).zfill(2)}_{str(mask_ind).zfill(3)}.png', mask['segmentation'])


    # start loop
    pred_masks = [np.zeros_like(mask) for mask in gt_masks]  # init at 0
    clicks, metrics = [], [compute_global_metrics(*compute_tps_fps_tns_fns(pred_masks, gt_masks))]
    (dstdir / 'preds').mkdir()
    
    for ind in range(MAX_CLICKS * N):
        print('click', ind+1, 'of', MAX_CLICKS * N)
        clicked_segment = get_clicked_segment(pred_masks, sam_masks_per_frame, gt_masks)  # click the mask that reduces the error the most, (frame, mask_index, label)
        print(clicked_segment)
        clicks.append(clicked_segment)

        # classify
        labels = classify(masks_feat_per_frame, clicks)
        # create segmentation
        pred_masks = create_segmentation(sam_masks_per_frame, labels, clicks)

        if not noplt:
            for frame_ind, frame_masks in enumerate(sam_masks_per_frame):
                if frame_ind == 8:
                    break
                plt.imsave(dstdir / 'preds' / f'pred_{str(ind+1).zfill(3)}_{str(frame_ind).zfill(2)}.png', pred_masks[frame_ind]*1, vmin=0, vmax=1)


        metdict = compute_global_metrics(*compute_tps_fps_tns_fns(pred_masks, gt_masks))
        metrics.append(metdict)
        with open(f'{dstdir}/metrics.json', 'w') as f:
            f.write(str(metrics).replace("'", '"'))

