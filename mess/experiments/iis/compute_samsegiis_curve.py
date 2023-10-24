import os
from config import datasets_path
os.environ['DETECTRON2_DATASETS'] = str(datasets_path)
import ast
from PIL import Image
import shutil
from pathlib import Path
import tqdm
import numpy as np

from IISS.create_segmentation import create_segmentation
from IISS.large_experiment import get_clicked_segment
from metrics import compute_global_metrics, compute_tps_fps_tns_fns

from mess.datasets.TorchvisionDataset import TorchvisionDataset, get_detectron2_datasets

def to_numpy(x):
    return np.array(x)

def handle_dstdir(dstdir, reset, resume):
    dstdir = Path(dstdir)
    assert not (resume and reset), "you can't both resume and reset"
    if not (resume or reset):
        if dstdir.exists():
            raise FileExistsError('run already exists, you should resume or reset')
    elif reset:
        if dstdir.exists():
            shutil.rmtree(dstdir)
            print('removed last run')
        else:
            print('creating brand new run')
        dstdir.mkdir(parents=True)
    elif resume:
        if dstdir.exists():
            print('resuming last run')
        else:
            print('creating brand new run')
            dstdir.mkdir(parents=True)
    return dstdir
        

def main(precomputed_dir, dstdir, max_clicks_per_image=10, reset=False, resume=False, ds=None):
    """`precomputed_dir` is a folder where the precomputed variables are stored. The variables are stored at `precomputed_dir / ds_name / sam / sam_masks_000.npy` where `000` is the image index on the dataset."""
    precomputed_dir = Path(precomputed_dir)
    assert precomputed_dir.exists()
    breakpoint()

    ds_names = get_detectron2_datasets()
    TEST_DATASETS=['atlantis_sem_seg_test', 'chase_db1_sem_seg_test', 'corrosion_cs_sem_seg_test', 'cryonuseg_sem_seg_test', 'cub_200_sem_seg_test', 'cwfid_sem_seg_test', 'dark_zurich_sem_seg_val', 'deepcrack_sem_seg_test', 'dram_sem_seg_test', 'foodseg103_sem_seg_test', 'isaid_sem_seg_val', 'kvasir_instrument_sem_seg_test', 'mhp_v1_sem_seg_test', 'paxray_sem_seg_test_bones', 'paxray_sem_seg_test_diaphragm', 'paxray_sem_seg_test_lungs', 'paxray_sem_seg_test_mediastinum', 'pst900_sem_seg_test', 'suim_sem_seg_test', 'worldfloods_sem_seg_test_irrg', 'zerowaste_sem_seg_test', 'ndd20_sem_seg_test', 'mypascalvoc_sem_seg_test', 'mysbd_sem_seg_test', 'mygrabcut_sem_seg_test']
    assert (ds is None) or ds in TEST_DATASETS 
    TEST_DATASETS = TEST_DATASETS if ds is None else [ds]
    ds_names = sorted([ds_name for ds_name in ds_names if ds_name in TEST_DATASETS])

    for ds_name in ds_names:

        try:
            dstdir = handle_dstdir(Path(dstdir) / ds_name, reset, resume)
        except FileExistsError:
            continue

        if resume:
            with open(dstdir / f'{ds_name}.json', 'r') as f:
                metrics_for_dataset = ast.literal_eval(f.read())
        else:
            (dstdir / 'vis' / ds_name).mkdir(exist_ok=True, parents=True)
            metrics_for_dataset = {}

        print(f'running {ds_name}')
        ds = TorchvisionDataset(ds_name, transform=to_numpy, mask_transform=to_numpy)
        print('dataset length:', len(ds), 'already annotated:', len(metrics_for_dataset))
        class_indices, class_names = np.arange(len(ds.class_names)), ds.class_names
        n_digits = len(str(len(ds)))
        values_to_ignore = [255] + [ind for ind, cls_name in zip(class_indices, class_names) if cls_name in ['background', 'others', 'unlabeled', 'background (waterbody)', 'background or trash']]
        for sample_ind, sample in tqdm.tqdm(enumerate(ds)):
            if resume and sample_ind in metrics_for_dataset:
                continue  # skip those that we already computed
            agg_mask_saved, img_saved = False, False
            metrics_for_dataset[sample_ind] = {}
            sam_masks = np.load(precomputed_dir / ds_name / 'sam' / f'sam_masks_{str(sample_ind).zfill(n_digits)}.npy', allow_pickle=True)
            mask = sample[1]
            gt_masks = [mask == value for value in np.unique(mask) if value not in values_to_ignore]
            for mvalue_ind, value in enumerate(np.unique(mask)):
                if value in values_to_ignore:
                    continue
                gt_masks = [(mask == value)]
                class_name = ds.class_names[value].replace(' ','-').replace('(','').replace(')','').replace('/','-').replace('_', '-')
                if gt_masks[0].sum() == 0:
                    metrics_for_dataset[sample_ind][class_name] = None
                    continue

                pred_masks = [np.zeros_like(gt_mask) for gt_mask in gt_masks]
                clicks, metrics = [], [compute_global_metrics(*compute_tps_fps_tns_fns(pred_masks, gt_masks))]

                for click_ind in range(max_clicks_per_image):
                    clicked_segment = get_clicked_segment(pred_masks, [sam_masks], gt_masks)
                    if clicked_segment is None:
                        break
                    clicks.append(clicked_segment)
                    pred_masks = create_segmentation([sam_masks], labels=[0]*len(sam_masks), clicks=clicks)
                    metdict = compute_global_metrics(*compute_tps_fps_tns_fns(pred_masks, gt_masks))
                    metrics.append(metdict)
                    if sample_ind < 10:  # plot
                        if not agg_mask_saved:
                            # compute the sum of all sam masks
                            agg_sam_mask = np.zeros_like(pred_masks[0]).astype(np.uint8)
                            for sam_mask in sam_masks:
                                agg_sam_mask += (sam_mask['segmentation']).astype(np.uint8)
                            Image.fromarray(agg_sam_mask).save(dstdir / 'vis' / ds_name / f'sample_{sample_ind}_agg_sam_mask.png')
                            agg_mask_saved = True
                        if not img_saved:
                            # plot image
                            img = sample[0]
                            Image.fromarray(img).save(dstdir  / 'vis' / ds_name / f'sample_{sample_ind}_img.png')
                            img_saved = True
                        # plot clicked mask
                        clicked_sam_mask = sam_masks[clicked_segment[1]]['segmentation']
                        Image.fromarray(clicked_sam_mask).save(dstdir / 'vis' / ds_name / f'class_{class_name}_sample_{sample_ind}_click_{click_ind}_selected.png')
                        # plot error map
                        error_map = np.zeros((clicked_sam_mask.shape + (3,)))
                        error_map[np.logical_and(gt_masks[0],pred_masks[0])] = np.array([255, 255, 255])
                        error_map[np.logical_and(~gt_masks[0], pred_masks[0])] = np.array([0, 0, 255])
                        error_map[np.logical_and(gt_masks[0], ~pred_masks[0])] = np.array([255, 0, 0])
                        Image.fromarray(error_map.astype(np.uint8)).save(dstdir / 'vis' / ds_name / f'class_{class_name}_sample_{sample_ind}_click_{click_ind}_error_map.png')
                    

                    


                metrics_for_dataset[sample_ind][class_name] = metrics
            print(f'processed {sample_ind+1}/{len(ds)}', end='\r')
            with open(dstdir / f'{ds_name}.json', 'w') as f:
                f.write(str(metrics_for_dataset).replace("'", '"'))
                

    print('great!')

if __name__ == '__main__':
    from fire import Fire
    Fire(main)

# example command
# python -m mess.experiments.iis.compute_samsegiis_curve /home/franchesoni/adisk/precomputed/ example_clicking --ds=chase_db1_sem_seg_test --max_clicks_per_image=5 --reset
