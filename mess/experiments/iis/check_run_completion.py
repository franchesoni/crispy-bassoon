from pathlib import Path
import os
import ast
from config import datasets_path
os.environ['DETECTRON2_DATASETS'] = str(datasets_path)
import numpy as np
import torch
from PIL import Image

from mess.datasets.TorchvisionDataset import TorchvisionDataset, get_detectron2_datasets

def to_numpy(x):
    return np.array(x)

def main(run_path, ds=None):
    ds_names = get_detectron2_datasets()
    run_path = Path(run_path)

    TEST_DATASETS=['atlantis_sem_seg_test', 'chase_db1_sem_seg_test', 'corrosion_cs_sem_seg_test', 'cryonuseg_sem_seg_test', 'cub_200_sem_seg_test', 'cwfid_sem_seg_test', 'dark_zurich_sem_seg_val', 'deepcrack_sem_seg_test', 'dram_sem_seg_test', 'foodseg103_sem_seg_test', 'isaid_sem_seg_val', 'kvasir_instrument_sem_seg_test', 'mhp_v1_sem_seg_test', 'paxray_sem_seg_test_bones', 'paxray_sem_seg_test_diaphragm', 'paxray_sem_seg_test_lungs', 'paxray_sem_seg_test_mediastinum', 'pst900_sem_seg_test', 'suim_sem_seg_test', 'worldfloods_sem_seg_test_irrg', 'zerowaste_sem_seg_test', 'ndd20_sem_seg_test', 'mypascalvoc_sem_seg_test', 'mysbd_sem_seg_test', 'mygrabcut_sem_seg_test']
    assert (ds is None) or ds in TEST_DATASETS 
    TEST_DATASETS = TEST_DATASETS if ds is None else [ds]

    ds_names = sorted([ds_name for ds_name in ds_names if ds_name in TEST_DATASETS])

    complete, uncomplete = [], []
    for ds_name in ds_names:
        print('='*80)
        ds = TorchvisionDataset(ds_name, transform=to_numpy, mask_transform=to_numpy)
        metrics_file = run_path / ds_name / f'{ds_name}.json'
        if not metrics_file.exists():
            print(f'no metrics for ds {ds_name}')
            continue
        with open(metrics_file, 'r') as f:
            metrics = ast.literal_eval(f.read())
        ratio_completed = len(metrics) / len(ds) 
        print(ds_name, 'completed', ratio_completed*100, '%')
        if ratio_completed == 1:
            complete.append(ds_name)
        else:
            uncomplete.append(ds_name)
    print('='*80)
    print('complete:', complete)
    print('uncomplete:', uncomplete)

    print('great!')

if __name__ == '__main__':
    from fire import Fire
    Fire(main)

