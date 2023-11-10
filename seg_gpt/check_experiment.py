"""
Main loop to run the experiment of how SEG-GPT performance changes with the number of shots.
This code runs only for one seed. In order to run for multiple seeds, launch (if possible parallel) multiple instances of this script with different seeds.
"""
# set detectron datasets, you're supposed to run this from crispy-bassoon
# and you're also supposed to run this using python -m seg_gpt.experiment
import os
from config import datasets_path
os.environ['DETECTRON2_DATASETS'] = str(datasets_path)
from pathlib import Path
import shutil
import argparse
import sys
import typing as T

import numpy as np
import tqdm

# TODO: Replace this with the correct path to the SegGPT repository
sys.path.append("seg_gpt/Painter/SegGPT/SegGPT_inference")

sys.path.append("..")
from mess.datasets.TorchvisionDataset import TorchvisionDataset

# TODO: Adjust the maximum number of shots depending on GPU size. For a quick try, use as NUMBER_OF_SHOTS = [MAXIMUM_NUMBER] and check if there's no CUDA error
CLASSES_TO_IGNORE = [
    "background",
    "others",
    "unlabeled",
    "background (waterbody)",
    "background or trash",
]


def get_class_img_mapping(dataset: TorchvisionDataset, values_to_ignore: T.List):
    """
    Returns a mapping that stores which images are available for each class.
    """
    print('getting class img mapping...')
    class_img_mapping = {}
    for ix, (_, mask) in tqdm.tqdm(enumerate(dataset), total=len(dataset)):
        for value in np.unique(mask):
            if value in values_to_ignore:
                continue
            if value not in class_img_mapping:
                class_img_mapping[value] = [ix]
            class_img_mapping[value].append(ix)
    return class_img_mapping


if __name__ == "__main__":
    print('%'*160)
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    args = parser.parse_args()
    DATASETS=[
        ('foodseg103_sem_seg_train',
        'foodseg103_sem_seg_test'),
        ('mhp_v1_sem_seg_train',
        'mhp_v1_sem_seg_test'),
        ('suim_sem_seg_train',
        'suim_sem_seg_test'),
        ('zerowaste_sem_seg_train',
        'zerowaste_sem_seg_test'),
        ('atlantis_sem_seg_train',
        'atlantis_sem_seg_test'),
        ('cwfid_sem_seg_train',
        'cwfid_sem_seg_test'),
        ('kvasir_instrument_sem_seg_train',
        'kvasir_instrument_sem_seg_test'),
        ('isaid_sem_seg_train',  
        'isaid_sem_seg_val'),
        ('deepcrack_sem_seg_train',
        'deepcrack_sem_seg_test'),
        ('corrosion_cs_sem_seg_train', 
        'corrosion_cs_sem_seg_test', )
    ]
    complete, incomplete = [], []
    for train_dataset, test_dataset in DATASETS:
        print('='*30)
        print(f"Train dataset: {train_dataset}, test dataset: {test_dataset}")
        seed_dirs = os.listdir(Path(args.results_dir) / test_dataset)
        print('seeds:', seed_dirs)
        for seed in seed_dirs:
            print('-'*20)
            print(f"Seed: {seed}")
            # Create results dir
            results_dir =  Path(args.results_dir) / test_dataset / seed

            # Try to load existing results if not resetting
            if os.path.isfile(os.path.join(results_dir, "metrics.npy")):
                metrics_per_shot = np.load(os.path.join(results_dir, "metrics.npy"), allow_pickle=True).item()
                print(f"shots {metrics_per_shot.keys()}")
                print(f"classes first {len(metrics_per_shot[list(metrics_per_shot.keys())[0]].keys())}, last {len(metrics_per_shot[list(metrics_per_shot.keys())[-1]].keys())}")
            else:
                print(f"OUPS: Results directory {results_dir} does not exist!")

