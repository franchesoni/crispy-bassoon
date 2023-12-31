
# run python mess/prepare_datasets/prepare_foodseg.py

import os
import tqdm
import numpy as np
from pathlib import Path
from PIL import Image


def download_dataset(ds_path):
    """
    Downloads the dataset
    """
    # Dataset page: https://github.com/LARC-CMU-SMU/FoodSeg103-Benchmark-v1
    print('Downloading dataset...')
    filesdir = ds_path / 'files'
    filesdir.mkdir(parents=True)
    # Downloading zip
    os.system('wget https://research.larc.smu.edu.sg/downloads/datarepo/FoodSeg103.zip -P ' + str(filesdir))

def extract_dataset(ds_path):
    filesdir = ds_path / 'files'
    os.system(f'unzip -P LARCdataset9947 {filesdir / "FoodSeg103.zip"} -d ' + str(ds_path))


def main():
    dataset_dir = Path(os.getenv('DETECTRON2_DATASETS', 'datasets'))
    ds_path = dataset_dir / 'FoodSeg103'
    if not ds_path.exists():
        download_dataset(ds_path)
        extract_dataset(ds_path)

    assert ds_path.exists(), f'Dataset not found in {ds_path}'

    print(f'Saved images and masks of {ds_path.name} dataset')


if __name__ == '__main__':
    main()
