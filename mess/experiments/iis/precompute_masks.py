from pathlib import Path
import os
from config import datasets_path
os.environ['DETECTRON2_DATASETS'] = str(datasets_path)
import tqdm
import shutil
import cProfile
import numpy as np
import torch
import time
from PIL import Image
import cv2

from IISS.extract_masks import extract_masks_single, get_embedding_sam
from IISS.compute_features import compute_features_list
from mess.datasets.TorchvisionDataset import TorchvisionDataset, get_detectron2_datasets
from IISS.project_masks import project_masks, project_masks_faster


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
            if img.shape[0] > 1000:
                while True:
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
            else:
                sam_masks = extract_masks_single(img)
            np.save(dstfile, sam_masks)
        if mode == 'dinosam':
            dstfile = dstdir / f'dinosam_feats_{str(i).zfill(ndigits)}.npy'
            if dstfile.exists() and not overwrite:
                continue
            # handle sam masks
            sammasks_dstfile = dstdir.parent / 'sam' / f'sam_masks_{str(i).zfill(ndigits)}.npy'
            if sammasks_dstfile.exists() and not overwrite:  # laod existing masks
                sam_masks = np.load(sammasks_dstfile, allow_pickle=True)
            else:
                raise RuntimeError('you should precompute masks')
            # handle dino features
            feats_dstfile = dstdir.parent / 'dino' / f'img_features_{str(i).zfill(ndigits)}.npy'
            if feats_dstfile.exists() and not overwrite:
                img_features = np.load(feats_dstfile, allow_pickle=True)
            else:
                raise RuntimeError('you should precompute features')
            # combine
            # st = time.time()
            # dinosam_feats = project_masks(sam_masks, img_features)
            # print('project_masks took', time.time() - st, 'seconds')
            # st = time.time()
            dinosam_feats = project_masks_faster(sam_masks, img_features)
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

    DATASETS=[
            'foodseg103_sem_seg_test', 
            'foodseg103_sem_seg_train', 
            'mhp_v1_sem_seg_test',
            'mhp_v1_sem_seg_train',
            'suim_sem_seg_test', 
            'suim_sem_seg_train', 
            'zerowaste_sem_seg_test', 
            'zerowaste_sem_seg_train', 
            'corrosion_cs_sem_seg_test', 
            'corrosion_cs_sem_seg_train', 
            'atlantis_sem_seg_train',
            'atlantis_sem_seg_test',
            'cwfid_sem_seg_train',
            'cwfid_sem_seg_test',
            'kvasir_instrument_sem_seg_train',
            'kvasir_instrument_sem_seg_test',
            'isaid_sem_seg_train',  
            'isaid_sem_seg_val',  
            'deepcrack_sem_seg_train',
            'deepcrack_sem_seg_test',
           # 'atlantis_sem_seg_test', 'chase_db1_sem_seg_test', 'cryonuseg_sem_seg_test', 'cub_200_sem_seg_test', 'dark_zurich_sem_seg_val', 'deepcrack_sem_seg_test', 'dram_sem_seg_test', 
           # 'paxray_sem_seg_test_bones', 'paxray_sem_seg_test_diaphragm', 'paxray_sem_seg_test_lungs', 'paxray_sem_seg_test_mediastinum', 'pst900_sem_seg_test', 'worldfloods_sem_seg_test_irrg', 'ndd20_sem_seg_test', 'mypascalvoc_sem_seg_test', 'mysbd_sem_seg_test', 'mygrabcut_sem_seg_test'
           ]
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

