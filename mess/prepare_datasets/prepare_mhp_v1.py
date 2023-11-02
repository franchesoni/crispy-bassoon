
# run python mess/prepare_datasets/prepare_mhp_v1.py

import os
import tqdm
import gdown
import numpy as np
from pathlib import Path
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def download_dataset(dataset_dir):
    """
    Downloads the dataset
    """
    print('Downloading dataset...')
    filesdir = dataset_dir / 'files'
    filesdir.mkdir()
    # Download from Google Drive
    gdown.download(f"https://drive.google.com/uc?export=download&confirm=pbef&id=1hTS8QJBuGdcppFAr_bvW2tsD9hW_ptr5", str(filesdir / 'LV-MHP-v1.zip'))

def extract_dataset(dataset_dir):
    filesdir = dataset_dir / 'files'
    os.system(f'unzip {filesdir / "LV-MHP-v1.zip"} -d ' + str(dataset_dir))

def prepare_mhp(dataset_dir):
    print('preparing mhp dataset...')
    ds_path = dataset_dir / 'LV-MHP-v1'
    assert ds_path.exists(), f'Dataset not found in {ds_path}'
    if (ds_path / 'was_prepared').exists():
        print('dataset already prepared!')
        return
    for split in ['train', 'test']:
        # create directories
        img_dir = ds_path / 'images_detectron2' / split
        anno_dir = ds_path / 'annotations_detectron2' / split
        img_dir.mkdir(parents=True, exist_ok=True)
        anno_dir.mkdir(parents=True, exist_ok=True)

        with open(ds_path / f'{split}_list.txt', 'r') as f:
            ids = f.read().splitlines()
        ids = [id.split('.')[0] for id in ids]
        if split == 'train':
            ids = ids[:3000]  # as in original mess code

        for id in tqdm.tqdm(ids):
            # Move image
            img = Image.open(ds_path / 'images' / f'{id}.jpg')
            img = img.convert('RGB')
            img.save(img_dir / f'{id}.jpg')

            # Load all masks
            mask_paths = list((ds_path / 'annotations').glob(f'{id}*.png'))
            mask = np.stack([np.array(Image.open(mask_path)) for mask_path in mask_paths])
            mask = Image.fromarray(mask.max(axis=0).astype(np.uint8))
            # Save mask
            mask.save(anno_dir / f'{id}.png')

        print(f'Saved {split} images and masks of {ds_path.name} dataset')
    os.system(f"touch {ds_path / 'was_prepared'}")



def main():
    dataset_dir = Path(os.getenv('DETECTRON2_DATASETS', 'datasets'))
    ds_path = dataset_dir / 'LV-MHP-v1'
    if not ds_path.exists():
        download_dataset(dataset_dir)
        extract_dataset(dataset_dir)

    assert ds_path.exists(), f'Dataset not found in {ds_path}'

    prepare_mhp(dataset_dir)
    # # create directories
    # for split in ['train', 'test']:
    #     os.makedirs(ds_path / 'images_detectron2' / split, exist_ok=True)
    #     os.makedirs(ds_path / 'annotations_detectron2' / split, exist_ok=True)

    # with open(ds_path / 'test_list.txt', 'r') as f:
    #     test_ids = f.read().splitlines()
    # with open(ds_path / 'train_list.txt', 'r') as f:
    #     train_ids = f.read().splitlines()
    # # using 3000 images for training, 1000 for validation, similar to the paper.
    # train_ids = train_ids[:3000]

    # for img_path in tqdm.tqdm(sorted((ds_path / 'images').glob('*.jpg'))):
    #     id = img_path.stem
    #     if img_path.name in test_ids:
    #         split = 'test'
    #     elif img_path.name in train_ids:
    #         split = 'train'

    #     # Move image
    #     img = Image.open(img_path)
    #     img = img.convert('RGB')
    #     img.save(ds_path / 'images_detectron2' / split / img_path.name)

    #     # Load all masks
    #     mask_paths = (ds_path / 'annotations').glob(f'{id}*.png')
    #     mask = np.stack([np.array(Image.open(mask_path)) for mask_path in mask_paths])
    #     # Combine masks
    #     mask = Image.fromarray(mask.max(axis=0).astype(np.uint8))
    #     # Save mask
    #     assert mask.mode == 'L', f'Expected mask to be in L mode, got {mask.mode}'
    #     assert mask.size == img.size, f'Expected mask and image to have the same size, got {mask.size} and {img.size}'
    #     mask.save(ds_path / 'annotations_detectron2' / split / f'{id}.png')

    # print(f'Saved train, val, and test images of {ds_path.name} dataset')


if __name__ == '__main__':
    main()
