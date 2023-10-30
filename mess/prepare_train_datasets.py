import os
from pathlib import Path

import tqdm
import rasterio
import numpy as np
from PIL import Image


def prepare_worldfloods(dataset_dir):
    print('preparing worldfloods dataset...')
    # normalization values for Sentinel 2 bands from:
    # https://gitlab.com/frontierdevelopmentlab/disaster-prevention/cubesatfloods/-/blob/master/data/worldfloods_dataset.py
    SENTINEL2_NORMALIZATION = np.array([
        [3787.0604973, 2634.44474043],
        [3758.07467509, 2794.09579088],
        [3238.08247208, 2549.4940614],
        [3418.90147615, 2811.78109878],
        [3450.23315812, 2776.93269704],
        [4030.94700446, 2632.13814197],
        [4164.17468251, 2657.43035126],
        [3981.96268494, 2500.47885249],
        [4226.74862547, 2589.29159887],
        [1868.29658114, 1820.90184704],
        [399.3878948,  761.3640411],
        [2391.66101119, 1500.02533014],
        [1790.32497137, 1241.9817628]], dtype=np.float32)

    def get_tiles(input, h_size=1024, w_size=1024, padding=0):
        input = np.array(input)
        h, w = input.shape[:2]
        tiles = []
        for i in range(0, h, h_size):
            for j in range(0, w, w_size):
                tile = input[i:i + h_size, j:j + w_size]
                if tile.shape[:2] == [h_size, w_size]:
                    tiles.append(tile)
                else:
                    # padding
                    if len(tile.shape) == 2:
                        # Mask (2 channels, padding with ignore_value)
                        padded_tile = np.ones((h_size, w_size), dtype=np.uint8) * padding
                    else:
                        # RGB (3 channels, padding usually 0)
                        padded_tile = np.ones((h_size, w_size, tile.shape[2]), dtype=np.uint8) * padding
                    padded_tile[:tile.shape[0], :tile.shape[1]] = tile
                    tiles.append(padded_tile)
        return tiles




    ds_path = dataset_dir / 'WorldFloods'
    assert ds_path.exists(), f'Dataset not found in {ds_path}.'

    for split in ['train', 'test']:
        # create directories
        img_dir = ds_path / 'images_detectron2' / split
        anno_dir = ds_path / 'annotations_detectron2' / split
        os.makedirs(anno_dir, exist_ok=True)
        for dir in ['rgb', 'irrg']:
            os.makedirs(img_dir / dir, exist_ok=True)

        for img_path in tqdm.tqdm(sorted((ds_path / 'worldfloods_v1_0_sample' / split / 'S2').glob('*.tif'))):
            id = img_path.stem
            # Open image
            with rasterio.open(img_path) as s2_rst:
                img = s2_rst.read()

            # normalize image by mean + one standard deviation (keeps 84% of the information)
            norm_img = img.transpose(1, 2, 0) / SENTINEL2_NORMALIZATION.sum(axis=1)
            norm_img = (np.clip(norm_img, 0, 1) * 255).astype(np.uint8)

            # RGB image
            rgb = norm_img[:, :, [3, 2, 1]]
            rgb_tiles = get_tiles(rgb, padding=0)
            for i, tile in enumerate(rgb_tiles):
                Image.fromarray(tile).save(img_dir / 'rgb' / f'{id}_{i}.png')

            # IRRG image
            irrg = norm_img[:, :, [7, 3, 1]]
            irrg_tiles = get_tiles(irrg, padding=0)
            for i, tile in enumerate(irrg_tiles):
                Image.fromarray(tile).save(img_dir / 'irrg' / f'{id}_{i}.png')

            # Open mask
            mask = np.array(Image.open(str(img_path).replace('S2', 'gt')))
            # Move ignore class to value 255
            for old, new in ((0, 255), (1, 0), (2, 1), (3, 2)):
                mask[(mask == old)] = new
            # Save mask
            mask_tiles = get_tiles(mask, padding=255)
            for i, tile in enumerate(mask_tiles):
                Image.fromarray(tile).save(anno_dir / f'{id}_{i}.png')

        print(f'Saved {split} images and masks of {ds_path.name} dataset')




def prepare_zerowaste(dataset_dir):
    print('preparing zerowaste dataset...')
    ds_path = dataset_dir / 'zerowaste-f'
    assert ds_path.exists(), f'Dataset not found in {ds_path}'
    print('No preparation needed')



def prepare_everything(detectron2_datasets_path):
    dsdir = Path(os.getenv('DETECTRON2_DATASETS', 'datasets')) if detectron2_datasets_path is None else Path(detectron2_datasets_path)
    dsdir.mkdir(parents=True, exist_ok=True)
    # change current directory to dsdir
    os.chdir(dsdir)
    # download
    dataset_dir = Path()  # now the directory is the same

    prepare_zerowaste(dataset_dir)
    prepare_worldfloods(dataset_dir)

if __name__ == '__main__':
    from fire import Fire
    Fire(prepare_everything)

