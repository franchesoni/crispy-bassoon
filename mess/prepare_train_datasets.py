import os
from pathlib import Path
import json

import tqdm
import rasterio
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
# Using 'inferno' color map for thermal images
inferno_colormap = plt.get_cmap('inferno')

def prepare_atlantis(dataset_dir):
    print('preparing atlantis dataset...')
    ds_path = dataset_dir / 'atlantis'
    assert ds_path.exists(), f'Dataset not found in {ds_path}'                   
    if (ds_path / 'was_prepared').exists():
        print('dataset already prepared!')
        return

    color_to_class = {c: i for i, c in enumerate(range(1, 57))}
    color_to_class[0] = 255

    for split in ['train', 'test']:
        # create directories
        img_dir = ds_path / 'images_detectron2' / split
        anno_dir = ds_path / 'annotations_detectron2' / split
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(anno_dir, exist_ok=True)

        for img_path in tqdm.tqdm((ds_path / 'images' / split).glob('*/*.jpg')):
            # Load and convert image and mask
            img = Image.open(img_path)
            img = img.convert('RGB')
            img.save(img_dir / img_path.name)

            mask = Image.open(str(img_path).replace('images', 'masks').replace('jpg', 'png'))
            # Replace grey values with class index
            mask = np.vectorize(color_to_class.get)(np.array(mask)).astype(np.uint8)
            Image.fromarray(mask).save(anno_dir / img_path.name.replace('jpg', 'png'))
        print(f'Saved images and masks of {ds_path.name} dataset for {split} split')
    os.system(f"touch {ds_path / 'was_prepared'}")



def prepare_chase(dataset_dir):
    print('preparing chase dataset...')
    ds_path = dataset_dir / 'CHASEDB1'
    assert ds_path.exists(), f'Dataset not found in {ds_path}'
    if (ds_path / 'was_prepared').exists():
        print('chase dataset already prepared!')
        return

    TRAIN_LEN = 8
    for split in ['train', 'test']:
        # create directories
        img_dir = ds_path / 'images_detectron2' / split
        anno_dir = ds_path / 'annotations_detectron2' / split
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(anno_dir, exist_ok=True)

        for i, img_path in tqdm.tqdm(enumerate(sorted((ds_path).glob('*.jpg')))):
            if (i < TRAIN_LEN and split == 'test') or (i >= TRAIN_LEN and split == 'train'):
                continue

            # Move image
            img = Image.open(img_path)
            img = img.convert('RGB')
            img.save(img_dir / img_path.name)

            # Open mask
            id = img_path.stem
            mask = Image.open(ds_path / f'{id}_1stHO.png')
            # Edit annotations
            # Binary encoding: (0, 255) -> (0, 1)
            mask = np.array(mask).astype(np.uint8)
            # Save mask
            Image.fromarray(mask).save(anno_dir / f'{id}.png')

        print(f'Saved {split} images and masks of {ds_path.name} dataset')

    os.system(f"touch {ds_path / 'was_prepared'}")


def prepare_corrosion(dataset_dir):
    print('preparing corrosion dataset...')
    ds_path = dataset_dir / 'CorrosionConditionStateClassification'
    assert ds_path.exists(), f'Dataset not found in {ds_path}'
    if (ds_path / 'was_prepared').exists():
        print('corrosion dataset already prepared!')
        return

    for split in ['Train', 'Test']:
        for mask_path in tqdm.tqdm(sorted(ds_path.glob(f'original/{split}/masks/*.png'))):
            # Open mask
            mask = np.array(Image.open(mask_path))
            # 'Portable network graphics' format, so no further processing needed
            # Save mask
            Image.fromarray(mask).save(mask_path)
        print(f'Saved images and masks of {ds_path.name} dataset for {split} split')
    os.system(f"touch {ds_path / 'was_prepared'}")

def prepare_cub200(dataset_dir):
    print('preparing cub_200 dataset...')
    ds_path = dataset_dir / 'CUB_200_2011'
    assert ds_path.exists(), f'Dataset not found in {ds_path}'
    if (ds_path / 'was_prepared').exists():
        print('cub_200 dataset already prepared!')
        return


    # read image file names
    with open(ds_path / 'images.txt', 'r') as f:
        img_files = [i.split(' ')[1] for i in f.read().splitlines()]

    # read test image list
    with open(ds_path / 'train_test_split.txt', 'r') as f:
        test_images = [not bool(int(i.split(' ')[1])) for i in f.read().splitlines()]
    
    for split in ['train', 'test']:
        # create directories
        img_dir = ds_path / 'images_detectron2' / split
        anno_dir = ds_path / 'annotations_detectron2' / split
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(anno_dir, exist_ok=True)

        if split == 'test':
            img_files_iter = np.array(img_files)[test_images]
        else:
            img_files_iter = np.array(img_files)[~np.array(test_images)]

        # iterate over all image files
        for img_file in tqdm.tqdm(img_files_iter):
            img_name = img_file.split('/')[-1]
            # Copy image
            img = Image.open(ds_path / 'images' / img_file)
            img = img.convert('RGB')
            img.save(img_dir / img_name)

            # Open mask
            img_name = img_name.replace('jpg', 'png')
            mask = Image.open(str(ds_path / 'segmentations' / img_file.replace('jpg', 'png'))).convert('L')

            # Edit annotations
            # Using majority voting from 5 labelers to get binary mask
            bin_mask = np.uint8(np.array(mask) > 128)
            # Replace mask with class index
            class_idx = int(img_file.split('.')[0])
            mask = bin_mask * class_idx
            # Save normal mask
            Image.fromarray(mask, 'L').save(anno_dir / img_name, "PNG")

        print(f'Saved {split} images and masks of {ds_path.name} dataset')
    os.system(f"touch {ds_path / 'was_prepared'}")




def prepare_cwfid(dataset_dir):

    print('preparing cwfid dataset...')
    ds_path = dataset_dir / 'cwfid'
    assert ds_path.exists(), f'Dataset not found in {ds_path}'
    if (ds_path / 'was_prepared').exists():
        print('cwfid dataset already prepared!')
        return

    train_ids = [2, 5, 6, 7, 8, 11, 12, 14, 16, 17, 18, 19, 20, 23, 24, 25, 27, 28, 31, 33, 34, 36, 37, 38, 40, 41, 42, 43,
                45, 46, 49, 50, 51, 52, 53, 55, 56, 57, 58, 59]
    test_ids = [1, 3, 4, 9, 10, 13, 15, 21, 22, 26, 28, 29, 30, 32, 35, 39, 44, 47, 48, 54, 60]
    for split in ['train', 'test']:
        img_dir = ds_path / 'images_detectron2' / split
        anno_dir = ds_path / 'annotations_detectron2' / split
        os.makedirs(anno_dir, exist_ok=True)
        os.makedirs(img_dir, exist_ok=True)

        ids = test_ids if split == 'test' else train_ids
        for id in tqdm.tqdm(ids):
            img_path = ds_path / 'images' / f'{id:03}_image.png'
            img = Image.open(img_path).save(img_dir / img_path.name)
            # get mask path
            mask_path = ds_path / 'annotations' / f'{id:03}_annotation.png'
            # Open mask
            mask = np.array(Image.open(mask_path))

            # Edit annotations
            color_to_class = {0: [0, 0, 0], 1: [255, 0, 0], 2: [0, 255, 0]}
            # Map RGB values to class index by converting to grayscale and applying a lookup table
            for class_idx, rgb in color_to_class.items():
                mask[(mask == rgb).all(axis=-1)] = class_idx
            mask = mask[:, :, 0]

            # Save mask
            Image.fromarray(mask).save(anno_dir / mask_path.name)

        print(f'Saved {split} images and masks of {ds_path.name} dataset')
    os.system(f"touch {ds_path / 'was_prepared'}")



def prepare_deepcrack(dataset_dir):
    print('preparing deepcrack dataset...')
    ds_path = dataset_dir / 'DeepCrack'
    assert ds_path.exists(), f'Dataset not found in {ds_path}'
    if (ds_path / 'was_prepared').exists():
        print('dataset already prepared!')
        return

    for split in ['train', 'test']:
        # create directories
        anno_dir = ds_path / 'annotations_detectron2' / split
        os.makedirs(anno_dir, exist_ok=True)

        for mask_path in tqdm.tqdm((ds_path / f'{split}_lab').glob('*.png')):
            # Open mask
            mask = Image.open(mask_path)
            # Edit annotations
            # Binary encoding: (0, 255) -> (0, 1)
            mask = np.uint8(np.array(mask) / 255)
            # Save mask
            Image.fromarray(mask).save(anno_dir / mask_path.name)

        print(f'Saved {split} images and masks of {ds_path.name} dataset')
    os.system(f"touch {ds_path / 'was_prepared'}")







    


def prepare_foodseg(dataset_dir):
    print('preparing foodseg dataset...')
    ds_path = dataset_dir / 'FoodSeg103'
    assert ds_path.exists(), f'Dataset not found in {ds_path}'
    if (ds_path / 'was_prepared').exists():
        print('dataset already prepared!')
        return
    os.system(f"touch {ds_path / 'was_prepared'}")


def prepare_isaid(dataset_dir):
    # iSAID dataset color to class mapping
    color_to_class = {0: [0, 0, 0],  # unlabeled
                    1: [0, 0, 63],  # ship
                    2: [0, 63, 63],  # storage_tank
                    3: [0, 63, 0],  # baseball_diamond
                    4: [0, 63, 127],  # tennis_court
                    5: [0, 63, 191],  # basketball_court
                    6: [0, 63, 255],  # Ground_Track_Field
                    7: [0, 127, 63],  # Bridge
                    8: [0, 127, 127],  # Large_Vehicle
                    9: [0, 0, 127],  # Small_Vehicle
                    10: [0, 0, 191],  # Helicopter
                    11: [0, 0, 255],  # Swimming_pool
                    12: [0, 191, 127],  # Roundabout
                    13: [0, 127, 191],  # Soccer_ball_field
                    14: [0, 127, 255],  # plane
                    15: [0, 100, 155],  # Harbor
                    }

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





    print('preparing isaid dataset...')
    ds_path = dataset_dir / 'isaid'
    assert ds_path.exists(), f'Dataset not found in {ds_path}'
    if (ds_path / 'was_prepared').exists():
        print('dataset already prepared!')
        return

    for split in ['train', 'val']:
        assert (ds_path / f'{split}_images').exists(),f'Raw {split} images not found in {ds_path / f"{split}_images"}'
        # create directories
        img_dir = ds_path / 'images_detectron2' / split
        anno_dir = ds_path / 'annotations_detectron2' / split
        img_dir.mkdir(parents=True, exist_ok=True)
        anno_dir.mkdir(parents=True, exist_ok=True)

        # Convert annotations to detectron2 format
        for mask_path in tqdm.tqdm(sorted((ds_path / f"{split}_masks" / "images").glob("*.png"))):
            file = mask_path.name
            id = file.split('_')[0]
            if len(list(anno_dir.glob(f'{id}_*.png'))):
                continue
            # Open image
            img = Image.open(ds_path / f'{split}_images' / "images" / f'{id}.png')
            # Open mask
            mask = np.array(Image.open(mask_path))
            # Map RGB values to class index by applying a lookup table
            for class_idx, rgb in color_to_class.items():
                mask[(mask == rgb).all(axis=-1)] = class_idx
            mask = mask[:, :, 0]  # remove channel dimension

            # get tiles
            img_tiles = get_tiles(img, padding=0)
            mask_tiles = get_tiles(mask, padding=255)
            # save tiles
            for i, (img_tile, mask_tile) in enumerate(zip(img_tiles, mask_tiles)):
                Image.fromarray(img_tile).save(img_dir / f'{id}_{i}.png')
                Image.fromarray(mask_tile).save(anno_dir / f'{id}_{i}.png')
        
        print(f'Saved {split} images and masks of {ds_path.name} dataset')
    os.system(f"touch {ds_path / 'was_prepared'}")


def prepare_kvasir(dataset_dir):
    print('preparing kvasir dataset...')
    ds_path = dataset_dir / 'kvasir-instrument'
    assert ds_path.exists(), f'Dataset not found in {ds_path}'
    if (ds_path / 'was_prepared').exists():
        print('dataset already prepared!')
        return
    
    for split in ['train', 'test']:
        img_dir = ds_path / 'images_detectron2' / split
        anno_dir = ds_path / 'annotations_detectron2' / split
        img_dir.mkdir(parents=True, exist_ok=True)
        anno_dir.mkdir(parents=True, exist_ok=True)

        with open(ds_path / f'{split}.txt', 'r') as f:
            ids = [line.rstrip() for line in f]

        for id in tqdm.tqdm(ids):
            img = Image.open(ds_path / f'images/{id}.jpg').convert('RGB')
            img.save(img_dir / f'{id}.png', "PNG")

            mask = Image.open(ds_path / f'masks/{id}.png')
            # 0: others, 1: instrument
            mask = np.uint8(np.array(mask)[:,:,0] / 255)
            Image.fromarray(mask).save(anno_dir / f'{id}.png', "PNG")
        
        print(f'Saved {split} images and masks of kvasir-instrument dataset')
    os.system(f"touch {ds_path / 'was_prepared'}")



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

def prepare_paxray(dataset_dir):
    print('preparing paxray dataset...')
    ds_path = dataset_dir / 'paxray_dataset'
    assert ds_path.exists(), f'Dataset not found in {ds_path}'
    if (ds_path / 'was_prepared').exists():
        print('dataset already prepared!')
        return

    with open(ds_path / 'paxray.json', 'r') as f:
        data = json.load(f)

    # binary predictions because of overlapping masks
    target_labels = {
        0: 'lungs',
        10: 'mediastinum',
        24: 'bones',
        163: 'diaphragm',
    }

    for split in ['train', 'test']:
        # create directories
        img_dir = ds_path / 'images_detectron2' / split
        os.makedirs(img_dir, exist_ok=True)
        anno_dir = ds_path / 'annotations_detectron2' / split
        for label in target_labels.values():
            (anno_dir / label).mkdir(parents=True, exist_ok=True)

        for paths in tqdm.tqdm(data[split]):
            # Copy image
            img = Image.open(ds_path / paths['image'])
            img = img.convert('RGB')
            img.save(img_dir / paths['image'][7:])

            # Open mask from .npy file
            mask = np.load(ds_path / paths['target'])
            # Save masks of each label separately for binary predictions
            for idx, label in target_labels.items():
                Image.fromarray(mask[idx].astype(np.uint8)).save(anno_dir / label / paths['image'][7:])

        print(f'Saved {split} images and masks for {", ".join(target_labels.values())} of {ds_path.name} dataset')

    os.system(f"touch {ds_path / 'was_prepared'}")



def prepare_pst(dataset_dir):
    print('preparing pst900 dataset...')
    ds_path = dataset_dir / 'PST900_RGBT_Dataset'
    assert ds_path.exists(), f'Dataset not found in {ds_path}'
    if (ds_path / 'was_prepared').exists():
        print('dataset already prepared!')
        return
    
    for split in ['train', 'test']:
        # create directories
        thermal_dir = ds_path / split / 'thermal_pseudo'
        os.makedirs(thermal_dir, exist_ok=True)

        for img_path in tqdm.tqdm((ds_path / split / 'thermal').glob('*.png')):
            # Open image
            img = Image.open(img_path)
            # Change thermal gray scale to pseudo color
            img = inferno_colormap(np.array(img)) * 255
            img = img.astype(np.uint8)[:, :, :3]
            # Save thermal pseudo color image
            Image.fromarray(img).save(thermal_dir / img_path.name)

        print(f'Saved images and masks of {ds_path.name} dataset for {split} split')
    os.system(f"touch {ds_path / 'was_prepared'}")



def prepare_suim(dataset_dir):
    print('preparing suim dataset...')
    ds_path = dataset_dir / 'SUIM'
    assert ds_path.exists(), f'Dataset not found in {ds_path}'
    if (ds_path / 'was_prepared').exists():
        print('dataset already prepared!')
        return

    class_dict = {
        (0, 0, 0): 0,  # BW
        (0, 0, 255): 1,  # HD
        (0, 255, 0): 2,  # PF
        (0, 255, 255): 3,  # WR
        (255, 0, 0): 4,  # RO
        (255, 0, 255): 5,  # RI
        (255, 255, 0): 6,  # FV
        (255, 255, 255): 7,  # SR
    }
    for split in ['train', 'test']:
        # create directories
        anno_dir = ds_path / 'annotations_detectron2' / split
        os.makedirs(anno_dir, exist_ok=True)

        subdir = 'TEST' if split == 'test' else 'train_val'
        for mask_path in tqdm.tqdm(sorted((ds_path / split / subdir / 'masks').glob('*.bmp'))):
            if (anno_dir / f'{mask_path.stem}.png').exists():
                continue  # allow resume
            # Open mask
            mask = Image.open(mask_path)
            mask = ((np.array(mask) > 127) * 255).astype(np.uint8)

            # Edit annotations using class_dict
            mask = np.apply_along_axis(lambda x: class_dict[tuple(x)], 2, mask).astype(np.uint8)

            # Save mask
            Image.fromarray(mask).save(anno_dir / f'{mask_path.stem}.png', "PNG")

        print(f'Saved {split} images and masks of {ds_path.name} dataset')
    os.system(f"touch {ds_path / 'was_prepared'}")





def prepare_worldfloods(dataset_dir):
    print('preparing worldfloods dataset...')
    ds_path = dataset_dir / 'WorldFloods'
    assert ds_path.exists(), f'Dataset not found in {ds_path}'
    if (ds_path / 'was_prepared').exists():
        print('dataset already prepared!')
        return

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
    os.system(f"touch {ds_path / 'was_prepared'}")




def prepare_zerowaste(dataset_dir):
    print('preparing zerowaste dataset...')
    ds_path = dataset_dir / 'zerowaste-f'
    assert ds_path.exists(), f'Dataset not found in {ds_path}'
    if (ds_path / 'was_prepared').exists():
        print('dataset already prepared!')
        return

    os.system(f"touch {ds_path / 'was_prepared'}")




def prepare_everything(detectron2_datasets_path):
    dsdir = Path(os.getenv('DETECTRON2_DATASETS', 'datasets')) if detectron2_datasets_path is None else Path(detectron2_datasets_path)
    dsdir.mkdir(parents=True, exist_ok=True)
    # change current directory to dsdir
    os.chdir(dsdir)
    # download
    dataset_dir = Path()  # now the directory is the same

    prepare_zerowaste(dataset_dir)
    prepare_worldfloods(dataset_dir)
    prepare_suim(dataset_dir)
    prepare_pst(dataset_dir)
    prepare_paxray(dataset_dir)
    prepare_mhp(dataset_dir)
    prepare_kvasir(dataset_dir)
    prepare_isaid(dataset_dir)
    prepare_foodseg(dataset_dir)
    prepare_deepcrack(dataset_dir)
    prepare_cwfid(dataset_dir)
    prepare_cub200(dataset_dir)
    prepare_corrosion(dataset_dir)
    prepare_chase(dataset_dir)
    prepare_atlantis(dataset_dir)

if __name__ == '__main__':
    from fire import Fire
    Fire(prepare_everything)

