import shutil
import os
from pathlib import Path

import numpy as np
from IISS.create_segmentation import create_segmentation
from IISS.run_experiment import get_clicked_segment
from metrics import compute_global_metrics, compute_tps_fps_tns_fns

def samseg_iis(load_img_with_masks_as_batch_fn, n_images, precomputed_dir, max_clicks_per_image, runname, reset):
    precomputed_dir = Path(precomputed_dir)
    assert precomputed_dir.exists()
    ndigits_pre = len(os.listdir(precomputed_dir)[0].split('_')[2].split('.')[0])
    n_digits = len(str(n_images))

    dstdir = Path(runname)
    try:
        dstdir.mkdir(parents=True)
    except FileExistsError:
        if reset:
            # input('resetting, you sure? else press ctrl+c')
            shutil.rmtree(dstdir)
            dstdir.mkdir()
            print('removed last run')
        else:
            print('run already exists and not resetting...')
            return
    
    metrics_per_image = []
    for imgind in range(n_images):
        print(f"processing {imgind+1}/{n_images}", end="\r")
        _, img, gt_masks = load_img_with_masks_as_batch_fn(imgind)
        subdstdir = dstdir / f"img{str(imgind).zfill(n_digits)}"
 
        for maskind, gt_mask in enumerate(gt_masks):
            if gt_mask is None:
                continue
            subdstdirmask = subdstdir / f"class{str(maskind).zfill(2)}"
            subdstdirmask.mkdir(parents=True)

            images, gt_masks = [img], [gt_mask]
            sam_masks = np.load(precomputed_dir / f'sam_masks_{str(imgind).zfill(ndigits_pre)}.npy', allow_pickle=True)

            # do the clicking sequence
            pred_masks = [np.zeros_like(mask) for mask in gt_masks]  # init at 0
            clicks, metrics = [], [compute_global_metrics(*compute_tps_fps_tns_fns(pred_masks, gt_masks))]

            for click_ind in range(max_clicks_per_image):
                clicked_segment = get_clicked_segment(pred_masks, [sam_masks], gt_masks)
                clicks.append(clicked_segment)
                pred_masks = create_segmentation([sam_masks], labels=[0]*len(sam_masks), clicks=clicks)
                metdict = compute_global_metrics(*compute_tps_fps_tns_fns(pred_masks, gt_masks))
                metrics.append(metdict)
            with open(subdstdirmask / f'metrics.json', 'w') as f:
                f.write(str(metrics).replace("'", '"'))