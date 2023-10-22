from pathlib import Path
import os
os.environ['DETECTRON2_DATASETS'] = "/gpfsscratch/rech/chl/uyz17rc/cvpr/data"
import tqdm
import shutil
import cProfile
import numpy as np
import torch
from PIL import Image

from IISS.large_experiment import extract_masks_single
from IISS.compute_features import compute_features_list
from mess.datasets.TorchvisionDataset import TorchvisionDataset, get_detectron2_datasets, get_class_names
from config import datasets_path


def precompute_for_dataset(torchvision_dataset, dstdir, reset=False, dev=False, dino=True, sam=True, overwrite=False, return_if_dir_exists=True):
    ndigits = len(str(len(torchvision_dataset)))
    dstdir = Path(dstdir)
    if dstdir.exists():
        if reset:
            print('resetting precomputed data...')
            shutil.rmtree(dstdir)
        elif return_if_dir_exists:
            print('precomputed data already exists, returning...')
            return
    (dstdir).mkdir(parents=True, exist_ok=True)

    # start profiling
    pr = cProfile.Profile()
    pr.enable()

    for i in tqdm.tqdm(range(len(torchvision_dataset))):
        if sam:
            dstfile = dstdir / f'sam_masks_{str(i).zfill(ndigits)}.npy'
            if dstfile.exists() and not overwrite:
                continue
            img, mask = torchvision_dataset[i]
            sam_masks = extract_masks_single(img)
            np.save(dstfile, sam_masks)
        if dino:
            dstfile = dstdir / f'img_features_{str(i).zfill(ndigits)}.npy'
            if dstfile.exists() and not overwrite:
                continue
            img, mask = torchvision_dataset[i]
            img_features = compute_features_list([img])[0]
            np.save(dstfile, img_features)
        if i == 3:
            # save the profile
            pr.disable()
            pr.dump_stats('mess_experiments_iis_precompute_masks.prof')
            del pr
            if dev:
                print('dev run, breaking...')
                break
    
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

def main(mode, dev=False):
    assert mode in ['sam', 'dino', 'dry']
    ds_names = get_detectron2_datasets()

    TEST_DATASETS=['atlantis_sem_seg_test', 'chase_db1_sem_seg_test', 'corrosion_cs_sem_seg_test', 'cryonuseg_sem_seg_test', 'cub_200_sem_seg_test', 'cwfid_sem_seg_test', 'dark_zurich_sem_seg_val', 'deepcrack_sem_seg_test', 'dram_sem_seg_test', 'foodseg103_sem_seg_test', 'isaid_sem_seg_val', 'kvasir_instrument_sem_seg_test', 'mhp_v1_sem_seg_test', 'paxray_sem_seg_test_bones', 'paxray_sem_seg_test_diaphragm', 'paxray_sem_seg_test_lungs', 'paxray_sem_seg_test_mediastinum', 'pst900_sem_seg_test', 'suim_sem_seg_test', 'worldfloods_sem_seg_test_irrg', 'zerowaste_sem_seg_test']

    ds_names = sorted([ds_name for ds_name in ds_names if ds_name in TEST_DATASETS])

    found, not_found = [], []
    for ds_name in ds_names:
        try:
            ds = TorchvisionDataset(ds_name, lambda x: x)
            print('Sample of dataset', ds_name, 'of size', len(ds))
            describe(ds[0], level=1)
            found.append(ds_name)
        except (AssertionError, FileNotFoundError, ModuleNotFoundError) as e:
            not_found.append((ds_name, str(e)))

    print('='*80)
    print('found:', found)
    print('='*80)
    print('not found:', not_found)
    print('='*80)

    # processing cwfid_sem_seg_test <- INCOMPLETE SAM
    if mode == 'sam':
        # process masks
        for ds_name in found:
            print('SAM processing', ds_name)
            ds = TorchvisionDataset(ds_name, transform=to_numpy, mask_transform=to_numpy)
            precompute_for_dataset(ds, os.path.join(datasets_path,  f'precomputed/{ds_name}/sam'), reset=False, dev=dev, dino=False, sam=True, overwrite=False, return_if_dir_exists=False)

    if mode == 'dino':
        # process features
        for ds_name in found:
            print('DINO processing', ds_name)
            ds = TorchvisionDataset(ds_name, transform=to_numpy, mask_transform=to_numpy)
            precompute_for_dataset(ds, os.path.join(datasets_path, f'precomputed/{ds_name}/dino'), reset=False, dev=dev, dino=True, sam=False, overwrite=False, return_if_dir_exists=False)


    print('great!')

if __name__ == '__main__':
    from fire import Fire
    Fire(main)
