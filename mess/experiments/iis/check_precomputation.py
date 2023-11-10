from pathlib import Path
import os
from config import datasets_path
os.environ['DETECTRON2_DATASETS'] = str(datasets_path)
import numpy as np
import torch
from PIL import Image

from mess.datasets.TorchvisionDataset import TorchvisionDataset, get_detectron2_datasets

def to_numpy(x):
    return np.array(x)

def describe(var, level=0):
    indent = '  ' * level
    if isinstance(var, (np.ndarray, torch.Tensor)):
        print(f'{indent}Type: {type(var)}')
        print(f'{indent}Shape: {var.shape}')
        print(f'{indent}Min: {var.min()}')
        print(f'{indent}Max: {var.max()}')
        print(f'{indent}Dtype: {var.dtype}')
        # also print the number of unique values
        if isinstance(var, np.ndarray):
            print(f'{indent}Unique: {np.unique(var).shape[0]}')
        elif isinstance(var, torch.Tensor):
            print(f'{indent}Unique: {torch.unique(var).shape[0]}')
    elif isinstance(var, Image.Image):
        print(f'{indent}Type: {type(var)}')
        print(f'{indent}Size: {var.size}')
        print(f'{indent}Mode: {var.mode}')
        # if mode isn't rgb raise an alert
        if var.mode != 'RGB':
            print(f'{indent}'+'-'*80)
            print(f'{indent}WARNING: mode is {var.mode}')
        print(f'{indent}Format: {var.format}')
    elif isinstance(var, (list, tuple)):
        print(f'{indent}Type: {type(var)}')
        print(f'{indent}Length: {len(var)}')
        for i, item in enumerate(var):
            print(f'{indent}Item {i}:')
            describe(item, level + 1)
    elif isinstance(var, dict):
        print(f'{indent}Type: dict')
        print(f'{indent}Keys: {list(var.keys())}')
        for key, value in var.items():
            print(f'{indent}Key: {key}')
            describe(value, level + 1)
    else:
        print(f'{indent}Type: {type(var)}')
        print(f'{indent}Value: {var}')

def describe_show_found(ds_names):
    found, not_found = [], []
    for ds_name in ds_names:
        try:
            ds = TorchvisionDataset(ds_name, transform=to_numpy, mask_transform=to_numpy)
            print("Dataset", ds_name)
            print("classes:", ds.class_names)
            print('Sample of dataset', ds_name, 'of size', len(ds))
            describe(ds[0], level=1)
            # save some images
            dstdir = Path(f'examples/{ds_name}')
            dstdir.mkdir(exist_ok=True, parents=True)
            classes_per_image = []
            for i in range(10):
                Image.fromarray(ds[i][0]).save(dstdir / f'img_{i}.png')
                Image.fromarray(ds[i][1]).save(dstdir / f'mask_{i}.png')
                classes_per_image.append([ds.class_names[j] for j in np.unique(ds[i][1]) if j < 255])
            with open(dstdir / 'classes_per_image.txt', 'w') as f:
                f.write('\n'.join([str(cpi) for cpi in classes_per_image]))
            found.append(ds_name)
        except (AssertionError, FileNotFoundError, ModuleNotFoundError) as e:
            not_found.append((ds_name, str(e)))
    return found, not_found



def main(mode, ds=None, describe=False):
    assert mode in ['sam', 'dino', 'sam_embeddings']
    ds_names = get_detectron2_datasets()
    # classes to ignore:
    # others, background, unlabeled, 'background (waterbody)', 'background or trash'

    DATASETS=[
        'foodseg103_sem_seg_train',
        'mhp_v1_sem_seg_train',
        'suim_sem_seg_train',
        'zerowaste_sem_seg_train',
        'foodseg103_sem_seg_test',
        'mhp_v1_sem_seg_test',
        'suim_sem_seg_test',
        'zerowaste_sem_seg_test',
        'atlantis_sem_seg_train',
        'cwfid_sem_seg_train',
        'kvasir_instrument_sem_seg_train',
        'atlantis_sem_seg_test',
        'cwfid_sem_seg_test',
        'kvasir_instrument_sem_seg_test',
        'isaid_sem_seg_train',  
        'isaid_sem_seg_val',  
        'deepcrack_sem_seg_train',
        'deepcrack_sem_seg_test',
        'corrosion_cs_sem_seg_train', 
        'corrosion_cs_sem_seg_test', 
        #'chase_db1_sem_seg_test', 'cryonuseg_sem_seg_test', 'cub_200_sem_seg_test',  'dark_zurich_sem_seg_val', 'deepcrack_sem_seg_test', 'dram_sem_seg_test', 

        # 'isaid_sem_seg_val', 
        # 'paxray_sem_seg_test_bones', 'paxray_sem_seg_test_diaphragm', 'paxray_sem_seg_test_lungs', 'paxray_sem_seg_test_mediastinum', 'pst900_sem_seg_test', 

        # 'worldfloods_sem_seg_test_irrg',
        # 'ndd20_sem_seg_test', 'mypascalvoc_sem_seg_test', 'mysbd_sem_seg_test', 'mygrabcut_sem_seg_test'
        ]
    assert (ds is None) or ds in DATASETS 
    DATASETS = DATASETS if ds is None else [ds]

    ds_names = sorted([ds_name for ds_name in ds_names if ds_name in DATASETS])

    if describe:
        found, not_found = describe_show_found(ds_names)
        print('='*80)
        print('found:', found)
        print('='*80)
        print('not found:', not_found)
        print('='*80)
    else:
        found = ds_names

    # processing cwfid_sem_seg_test <- INCOMPLETE SAM
    complete, incomplete = [], []
    # process masks
    for ds_name in found:
        print(f'checking {mode}...', ds_name)
        ds = TorchvisionDataset(ds_name, transform=to_numpy, mask_transform=to_numpy)
        srcdir = Path(os.path.join(datasets_path, f'precomputed/{ds_name}/{mode}'))
        ratio_completed = len(os.listdir(srcdir)) / len(ds) if srcdir.exists() else 0
        print('completed', ratio_completed*100, '%')
        if ratio_completed == 1:
            complete.append(ds_name)
        else:
            incomplete.append(ds_name)
    print('='*80)
    print('complete:', complete)
    print('incomplete:', incomplete)

    print('great!')

if __name__ == '__main__':
    from fire import Fire
    Fire(main)

