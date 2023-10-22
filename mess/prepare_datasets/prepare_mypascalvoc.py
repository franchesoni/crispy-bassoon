
# run python mess/prepare_datasets/prepare_<DATASET NAME>.py

import os
import tqdm
import numpy as np
from pathlib import Path
from PIL import Image
import torchvision


def download_dataset(ds_path):
    """
    Downloads the dataset
    """
    print('Downloading dataset...')
    ds = torchvision.datasets.VOCSegmentation(root=ds_path, year='2012', image_set='val', download=True)

def main():
    dataset_dir = Path(os.getenv('DETECTRON2_DATASETS', 'datasets'))
    ds_path = dataset_dir / 'mypascalvoc'
    if not ds_path.exists():
        download_dataset(ds_path)


    assert ds_path.exists(), f'Dataset not found in {ds_path}'

    for split in ['test']:
        # TODO: Change if other directories are required
        # create directories
        img_dir = ds_path / 'images_detectron2' / split
        anno_dir = ds_path / 'annotations_detectron2' / split
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(anno_dir, exist_ok=True)

        ds = torchvision.datasets.VOCSegmentation(root=ds_path, year='2012', image_set='val', download=False)

        max_samples = len(ds)
        n_digits = len(str(max_samples))
        print('Saving images and masks...')
        for sample_ind in tqdm.tqdm(range(max_samples)):
            img, mask = ds[sample_ind]

            img = img.convert('RGB')
            img.save(img_dir / (str(sample_ind).zfill(n_digits)+'.png'))

            # Save mask
            mask.save(anno_dir / (str(sample_ind).zfill(n_digits)+'.png'))

        print(f'Saved {split} images and masks of {ds_path.name} dataset')


if __name__ == '__main__':
    main()
