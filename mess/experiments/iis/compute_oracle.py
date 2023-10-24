import os
from config import datasets_path
os.environ['DETECTRON2_DATASETS'] = str(datasets_path)
import ast
import shutil
from pathlib import Path
import tqdm
import numpy as np

from IISS.create_segmentation import create_segmentation
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
        

def main(precomputed_dir, dstdir, reset=False, resume=False, ds=None):
    """`precomputed_dir` is a folder where the precomputed variables are stored. The variables are stored at `precomputed_dir / ds_name / sam / sam_masks_000.npy` where `000` is the image index on the dataset."""
    precomputed_dir = Path(precomputed_dir)
    assert precomputed_dir.exists()

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

        if resume and (dstdir / f'{ds_name}.json').exists():
            with open(dstdir / f'{ds_name}.json', 'r') as f:
                metrics_for_dataset = ast.literal_eval(f.read())
        else:
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

                # for each mask, set the label to positive if more than half of it is contained in the ground truth
                soft_labels, hard_labels = [], []
                for mask_dict in sam_masks:
                    mask = mask_dict['segmentation']
                    percentage_correct = np.sum(mask * gt_masks[0]) / np.sum(mask)
                    soft_labels.append(2 * percentage_correct - 1)
                    hard_labels.append(2 * (percentage_correct>0.5) - 1)


                soft_pred = create_segmentation([sam_masks], soft_labels, [])
                hard_pred = create_segmentation([sam_masks], hard_labels, [])
                metrics_for_dataset[sample_ind][class_name] = {'soft_oracle': compute_global_metrics(*compute_tps_fps_tns_fns(soft_pred, gt_masks)),
                                                                'hard_oracle': compute_global_metrics(*compute_tps_fps_tns_fns(hard_pred, gt_masks))}
            print(f'processed {sample_ind+1}/{len(ds)}', end='\r')
            with open(dstdir / f'{ds_name}.json', 'w') as f:
                f.write(str(metrics_for_dataset).replace("'", '"'))
                

    print('great!')





if __name__ == '__main__':
    from fire import Fire
    Fire(main)

