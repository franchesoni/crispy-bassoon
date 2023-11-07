import os
from config import datasets_path
os.environ['DETECTRON2_DATASETS'] = str(datasets_path)

from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from mess.datasets.TorchvisionDataset  import TorchvisionDataset, get_detectron2_datasets


detectron_datasets = get_detectron2_datasets()

our_datasets = ['kvasir_instrument_sem_seg_test', 'kvasir_instrument_sem_seg_train', 'cub_200_sem_seg_test', 'cub_200_sem_seg_train', 'suim_sem_seg_test', 'suim_sem_seg_train', 'chase_db1_sem_seg_test', 'chase_db1_sem_seg_train', 'pst900_sem_seg_test', 'pst900_sem_seg_train', 'mhp_v1_sem_seg_train', 'mhp_v1_sem_seg_test', 'paxray_sem_seg_test_lungs', 'paxray_sem_seg_test_mediastinum', 'paxray_sem_seg_test_bones', 'paxray_sem_seg_test_diaphragm', 'paxray_sem_seg_train_lungs', 'paxray_sem_seg_train_mediastinum', 'paxray_sem_seg_train_bones', 'paxray_sem_seg_train_diaphragm', 'worldfloods_sem_seg_test_irrg', 'worldfloods_sem_seg_train_irrg', 'corrosion_cs_sem_seg_test', 'corrosion_cs_sem_seg_train', 'isaid_sem_seg_val', 'isaid_sem_seg_train', 'foodseg103_sem_seg_test', 'foodseg103_sem_seg_train', 'atlantis_sem_seg_test', 'atlantis_sem_seg_train', 'deepcrack_sem_seg_test', 'deepcrack_sem_seg_train', 'zerowaste_sem_seg_test', 'zerowaste_sem_seg_train', 'cwfid_sem_seg_test', 'cwfid_sem_seg_train']

def normto255(x):
    return ((x - x.min()) / (x.max() - x.min()) * 255).astype(np.uint8)

def save_visualization(dstdir):
    dstdir = Path(dstdir)
    count = 0
    for ds_name in our_datasets:
        print('-'*20)
        try:
            ds = TorchvisionDataset(ds_name, lambda x: x)
            (dstdir / ds_name).mkdir(parents=True, exist_ok=True)
            print('class names:', ds.class_names)
            print(ds_name, 'len:', len(ds))
            sample = ds[0]
            print('first sample', len(sample), sample[0].size, sample[1].shape)
            sample = ds[len(ds)-1]
            print('last sample', len(sample), sample[0].size, sample[1].shape)
            for ind, sample in enumerate(ds):
                if ind == 10:
                    break
                if not (dstdir / ds_name / f'{ind}_mask.png').exists():
                    img, mask_tensor = sample
                    mask = Image.fromarray(normto255(np.array(mask_tensor[0])))
                    img.save(dstdir / ds_name / f'{ind}_img.png')
                    mask.save(dstdir / ds_name / f'{ind}_mask.png')
                    print(f'saved image {ind}', end='\r')
                else:
                    print(f'skipping img {ind}', end='\r')

            print(ds_name, 'is ok')
            count += 1
        except Exception as e:
            print('<'*30)
            print(ds_name, 'is broken')
            print('failed with exception', e)
            print('>'*30)

    print('our datasets:', len(our_datasets), 'total datasets:', count, 'half is', count // 2)

if __name__ == '__main__':
    from fire import Fire
    Fire(save_visualization)
