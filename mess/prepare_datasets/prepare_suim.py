
# run python mess/prepare_datasets/prepare_suim.py

import os
import tqdm
import gdown
import numpy as np
from pathlib import Path
from PIL import Image

# RGB color code and object categories:
# -------------------------------------
# 000 BW: Background waterbody
# 001 HD: Human divers
# 010 PF: Plants/sea-grass
# 011 WR: Wrecks/ruins
# 100 RO: Robots/instruments
# 101 RI: Reefs and invertebrates
# 110 FV: Fish and vertebrates
# 111 SR: Sand/sea-floor (& rocks)

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


def download_dataset(ds_path):
    """
    Downloads the dataset
    """
    print('Downloading dataset...')
    filesdir = ds_path / 'files'
    filesdir.mkdir(parents=True)

    # Downloading zip
    gdown.download(id='1diN3tNe2nR1eV3Px4gqlp6wp3XuLBwDy', output=str(filesdir / 'TEST.zip'))

    # Downloading zip
    gdown.download(id='1YWjUODQWwQ3_vKSytqVdF4recqBOEe72', output=str(filesdir / 'train_val.zip'))


def extract_dataset(ds_path):
    filesdir = ds_path / 'files'
    os.system("unzip " + str(filesdir / "TEST.zip") + " -d " + str(ds_path / 'test'))
    os.system("unzip " + str(filesdir / "train_val.zip") + " -d " + str(ds_path / 'train'))


def prepare_suim(ds_path):
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
            mask = ((np.array(mask)>127) * 255).astype(np.uint8)

            # Edit annotations using class_dict
            mask = np.apply_along_axis(lambda x: class_dict[tuple(x)], 2, mask).astype(np.uint8)

            # Save mask
            Image.fromarray(mask).save(anno_dir / f'{mask_path.stem}.png', "PNG")

        print(f'Saved {split} images and masks of {ds_path.name} dataset')



def main():
    dataset_dir = Path(os.getenv("DETECTRON2_DATASETS", "datasets"))
    ds_path = dataset_dir / "SUIM"
    if not ds_path.exists():
        download_dataset(ds_path)
        extract_dataset(ds_path)

    assert ds_path.exists(), f"Dataset not found in {ds_path}"

    prepare_suim(ds_path)

if __name__ == "__main__":
    main()
