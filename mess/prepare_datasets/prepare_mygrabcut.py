
# run python mess/prepare_datasets/prepare_<DATASET NAME>.py

import os
import tqdm
import gdown
import numpy as np
from pathlib import Path
from PIL import Image


def download_dataset(ds_path):
    """
    Downloads the dataset
    """
    print('Downloading dataset...')
    # Downloading zip
    os.system('wget https://github.com/saic-vul/fbrs_interactive_segmentation/releases/download/v1.0/GrabCut.zip')
    os.system('unzip GrabCut.zip -d ' + str(ds_path))
    os.system('rm GrabCut.zip')


def main():
    dataset_dir = Path(os.getenv('DETECTRON2_DATASETS', 'datasets'))
    ds_path = dataset_dir / 'mygrabcut'
    if not ds_path.exists():
        download_dataset(ds_path)

    assert ds_path.exists(), f'Dataset not found in {ds_path}'

    for split in ['test']:
        img_dir = ds_path / 'images_detectron2' / split
        anno_dir = ds_path / 'annotations_detectron2' / split
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(anno_dir, exist_ok=True)

        original_images_dir = ds_path / 'GrabCut' / 'data_GT' 
        for img_path in tqdm.tqdm(os.listdir(original_images_dir)):
            # Copy image
            img = Image.open(original_images_dir / img_path)
            img = img.convert('RGB')
            img.save(img_dir / str((original_images_dir / img_path).stem + '.png'))

            id = (original_images_dir / img_path).stem
            mask = ((np.array(Image.open(ds_path / 'GrabCut' / 'boundary_GT' / f'{id}.bmp')) > 0 ) * 1).astype(np.uint8)

            # Save mask
            Image.fromarray(mask).save(anno_dir / f'{id}.png')

        print(f'Saved {split} images and masks of {ds_path.name} dataset')


if __name__ == '__main__':
    main()
