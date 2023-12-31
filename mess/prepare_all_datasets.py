# Description: This script prepares all datasets
# Usage: python mess/prepare_all_datasets.py --dataset_dir datasets

import os
import argparse
from detectron2.data import DatasetCatalog
from prepare_datasets import (
    prepare_mhp_v1,
    prepare_foodseg,
    prepare_atlantis,
    prepare_isaid,
    prepare_worldfloods,
    prepare_kvasir_instrument,
    prepare_paxray,
    prepare_pst900,
    prepare_corrosion_cs,
    prepare_deepcrack,
    prepare_zerowaste,
    prepare_suim,
    prepare_cub_200,
    prepare_cwfid,
    # prepare_ndd20,
    # prepare_mypascalvoc,
    # prepare_mysbd,
    # prepare_mygrabcut,
)

if __name__ == '__main__':
    # parser to get dataset directory
    parser = argparse.ArgumentParser(description='Prepare datasets')
    parser.add_argument('--dataset_dir', type=str, default='datasets', help='dataset directory')
    parser.add_argument('--stats', action='store_true', help='Only show dataset statistics')
    args = parser.parse_args()

    # set dataset directory and register datasets
    os.environ['DETECTRON2_DATASETS'] = args.dataset_dir
    os.makedirs(args.dataset_dir, exist_ok=True)
    import datasets

    # prepare datasets
    dataset_dict = {
        'mhp_v1_sem_seg_test': prepare_mhp_v1,  # natural images
        'mhp_v1_sem_seg_train': prepare_mhp_v1,
        'foodseg103_sem_seg_test': prepare_foodseg,  # food
        'foodseg103_sem_seg_train': prepare_foodseg,
        'suim_sem_seg_train': prepare_suim,  # underwater
        'suim_sem_seg_test': prepare_suim,
        'zerowaste_sem_seg_train': prepare_zerowaste,  # industrial, waste sorting
        'zerowaste_sem_seg_test': prepare_zerowaste,
        'corrosion_cs_sem_seg_train': prepare_corrosion_cs,  # materials
        'corrosion_cs_sem_seg_test': prepare_corrosion_cs,
        'atlantis_sem_seg_train': prepare_atlantis,  # infrastructure / landscapes
        'atlantis_sem_seg_test': prepare_atlantis,
        'cwfid_sem_seg_train': prepare_cwfid,  # agricultural
        'cwfid_sem_seg_test': prepare_cwfid,
        'kvasir_instrument_sem_seg_train': prepare_kvasir_instrument,  # medical
        'kvasir_instrument_sem_seg_test': prepare_kvasir_instrument,
        'isaid_sem_seg_train': prepare_isaid,  # remote sensing
        'isaid_sem_seg_val': prepare_isaid,
        'deepcrack_sem_seg_train': prepare_deepcrack,
        'deepcrack_sem_seg_test': prepare_deepcrack,  # geological

        # 'worldfloods_sem_seg_test_irrg': prepare_worldfloods,
        # 'paxray_sem_seg_test_lungs': prepare_paxray,
        # 'pst900_sem_seg_test': prepare_pst900,
        # 'cub_200_sem_seg_test': prepare_cub_200,

        # 'ndd20_sem_seg_test': prepare_ndd20,
        # 'mypascalvoc_sem_seg_test': prepare_mypascalvoc,
        # 'mysbd_sem_seg_test': prepare_mysbd,
        # 'mygrabcut_sem_seg_test': prepare_mygrabcut,
    }

    # print status of datasets
    print('Dataset: Status')
    for dataset_name in dataset_dict.keys():
        try:
            status = f'{len(DatasetCatalog.get(dataset_name))} images'
        except FileNotFoundError:
            status = 'Not found'
        except AssertionError:
            status = 'Not found'
        print(f'{dataset_name:50s} {status}')

    if args.stats:
        exit()

    for dataset_name, prepare_dataset in dataset_dict.items():
        # check if dataset is already prepared
        try:
            prepared = len(DatasetCatalog.get(dataset_name)) != 0
        except FileNotFoundError:
            prepared = False
        except AssertionError:
            prepared = False

        if prepared:
            print(f'\n{dataset_name} already prepared')
        else:
            # prepare dataset
            print(f'\nPreparing {dataset_name}')
            try:
                prepare_dataset.main()
            except Exception as e:
                print(f'Error while preparing {dataset_name}: \n{e}')

    # print status of datasets
    print('\nFinished preparing datasets')
    print('\nDataset: Status')
    for dataset_name in dataset_dict.keys():
        try:
            status = f'{len(DatasetCatalog.get(dataset_name))} images'
        except FileNotFoundError:
            status = 'Not found'
        except AssertionError:
            status = 'Not found'
        print(f'{dataset_name:50s} {status}')
