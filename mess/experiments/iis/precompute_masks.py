from pathlib import Path
import os
from config import datasets_path
os.environ['DETECTRON2_DATASETS'] = str(datasets_path)
import tqdm
import shutil
import cProfile
import numpy as np
import torch
from PIL import Image
import cv2

from IISS.extract_masks import extract_masks_single, get_embedding_sam
from IISS.compute_features import compute_features_list
from mess.datasets.TorchvisionDataset import TorchvisionDataset, get_detectron2_datasets
from IISS.project_masks import project_masks
from mess.prepare_all_datasets import DATASETS


def precompute_for_dataset(torchvision_dataset, dstdir, mode, reset=False, dev=False, overwrite=False, return_if_dir_exists=True):
    assert mode in ['sam', 'dino', 'sam_embeddings', 'dinosam']
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
        if mode == 'sam':
            dstfile = dstdir / f'sam_masks_{str(i).zfill(ndigits)}.npy'
            if dstfile.exists() and not overwrite:
                continue
            img, mask = torchvision_dataset[i]
            while img.shape[0] > 1000:
                try:
                    if img.shape[0] > 3000:
                        raise RuntimeError
                    sam_masks = extract_masks_single(img)
                    break
                except RuntimeError:
                    # downsample to fit memory
                    img = cv2.resize(img, dsize=(img.shape[0]//2, img.shape[1]//2), interpolation=cv2.INTER_LINEAR)
                    mask = cv2.resize(mask, dsize=(mask.shape[0]//2, mask.shape[1]//2), interpolation=cv2.INTER_NEAREST)
                    torch.cuda.empty_cache()
                    if img.shape[0] < 100:
                        raise RuntimeError('your image got way too small!')
            np.save(dstfile, sam_masks)
        if mode == 'dinosam':
            dstfile = dstdir / f'dinosam_feats_{str(i).zfill(ndigits)}.npy'
            if dstfile.exists() and not overwrite:
                continue
            # handle sam masks
            sammasks_dstfile = dstdir / f'sam_masks_{str(i).zfill(ndigits)}.npy'
            if sammasks_dstfile.exists() and not overwrite:  # laod existing masks
                sam_masks = np.load(sammasks_dstfile, allow_pickle=True)
            else:  # compute and save masks
                img, mask = torchvision_dataset[i]
                sam_masks = extract_masks_single(img)
                np.save(sammasks_dstfile, sam_masks)
            # handle dino features
            feats_dstfile = dstdir / f'img_feats_{str(i).zfill(ndigits)}.npy'
            if feats_dstfile.exists() and not overwrite:
                img_features = np.load(feats_dstfile, allow_pickle=True)
            else:
                img, mask = torchvision_dataset[i]  # maybe repeated, but it's ok
                img_features = compute_features_list([img])[0]
                np.save(feats_dstfile, img_features)
            # combine
            dinosam_feats = project_masks(sam_masks, img_features)
            np.save(dstfile, dinosam_feats)
        if mode == 'dino':
            dstfile = dstdir / f'img_features_{str(i).zfill(ndigits)}.npy'
            if dstfile.exists() and not overwrite:
                continue
            img, mask = torchvision_dataset[i]
            img_features = compute_features_list([img])[0]
            np.save(dstfile, img_features)
        if mode == 'sam_embeddings':
            dstfile = dstdir / f'sam_embedding_{str(i).zfill(ndigits)}.npy'
            if dstfile.exists() and not overwrite:
                continue
            img, mask = torchvision_dataset[i]
            embedding = get_embedding_sam(img)
            np.save(dstfile, embedding)
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

def main(mode, dev=False, ds=None):
    assert mode in ['sam', 'dino', 'describe', 'sam_embeddings', 'dinosam']
    ds_names = get_detectron2_datasets()

    assert (ds is None) or ds in DATASETS 
    DATASETS = DATASETS if ds is None else [ds]

    ds_names = sorted([ds_name for ds_name in ds_names if ds_name in DATASETS])

    if mode == 'describe':
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

    # process masks
    for ds_name in ds_names:
        print(f'{mode} processing', ds_name)
        ds = TorchvisionDataset(ds_name, transform=to_numpy, mask_transform=to_numpy)
        precompute_for_dataset(ds, os.path.join(datasets_path,  f'precomputed/{ds_name}/{mode}'), reset=False, dev=dev, mode=mode, overwrite=False, return_if_dir_exists=False)
    print('great!')

if __name__ == '__main__':
    from fire import Fire
    Fire(main)

