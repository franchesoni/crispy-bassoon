# run python mess/prepare_datasets/prepare_<DATASET NAME>.py

import os
import json
from pathlib import Path
import tqdm
import numpy as np
from PIL import Image, ImageDraw


class_dict = {"background": 0, "dolphin": 1, "whale": 2, "other": 3}


def get_length(dataset_path):
    ds_path = Path(dataset_path)
    all_labels = json.load(open(ds_path / "BELOW_LABELS.json"))
    rawfiles = sorted(all_labels.keys())
    return len(rawfiles)
 

# create a function that loads one image and its mask given the dataset path and the index
def load_sample(dataset_path: str, index: int):
    ds_path = Path(dataset_path)
    all_labels = json.load(open(ds_path / "BELOW_LABELS.json"))
    rawfiles = sorted(all_labels.keys())
    rawfile = rawfiles[index]
    img, mask = load_rawfile(all_labels, rawfile, ds_path / "BELOW")
    return np.array(img), np.array(mask)
    # could be made more efficient by using a prepopulated list of images and masks paths


def get_data(data):
    X = []
    Y = []
    L = []  # pre-allocate lists to fill in a for loop
    for region in data["regions"]:  # cycle through each polygon
        # get the x and y points from the dictionary
        X.append(region["shape_attributes"]["all_points_x"])
        Y.append(region["shape_attributes"]["all_points_y"])
        L.append(region["region_attributes"]["object"])
    return Y, X, L  # image coordinates are flipped relative to json coordinates


def get_mask(X, Y, nx, ny, L, class_dict):
    # get the dimensions of the image
    mask = np.zeros((nx, ny))

    for y, x, k in zip(X, Y, L):
        # the ImageDraw.Draw().polygon function we will use to create the mask
        # requires the x's and y's are interweaved, which is what the following
        # one-liner does
        polygon = np.vstack((x, y)).reshape((-1,), order="F").tolist()

        # create a mask image of the right size and infill according to the polygon
        if nx > ny:
            x, y = y, x
            img = Image.new("L", (nx, ny), 0)
        elif ny > nx:
            # x,y = y,x
            img = Image.new("L", (ny, nx), 0)
        else:
            img = Image.new("L", (nx, ny), 0)
        ImageDraw.Draw(img).polygon(polygon, outline=0, fill=1)
        # turn into a numpy array
        m = np.flipud(np.rot90(np.array(img)))
        try:
            mask[m == 1] = class_dict[k]
        except:
            mask[m.T == 1] = class_dict[k]

    return mask


def load_rawfile(all_labels, rawfile, ds_path):
    X, Y, L = get_data(all_labels[rawfile])
    img = Image.open(ds_path / all_labels[rawfile]["filename"])
    nx, ny, nz = np.shape(np.array(img))
    mask = get_mask(X, Y, nx, ny, L, class_dict)
    return img, mask



def download_dataset(ds_path):
    """
    Downloads the dataset
    """
    # TODO: Add an automated script if possible, otherwise remove code
    print('Downloading dataset...')
    # Downloading zip
    os.system('wget https://data.ncl.ac.uk/ndownloader/files/22774175')
    os.system('unzip 22774175 -d ' + str(ds_path))
    os.system('rm 22774175')


def main():
    dataset_dir = Path(os.getenv('DETECTRON2_DATASETS', 'datasets'))
    ds_path = dataset_dir / 'ndd20'
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

        max_samples = get_length(ds_path)
        n_digits = len(str(max_samples))
        for sample_ind in range(max_samples):
            img, mask = load_sample(ds_path, sample_ind)

            img = Image.fromarray(img)
            img = img.convert('RGB')
            img.save(img_dir / (str(sample_ind).zfill(n_digits)+'.png'))

            mask = np.uint8(1 * (mask > 0))
            Image.fromarray(mask).save(
                    anno_dir / (
                        str(sample_ind).zfill(n_digits)
                        +'.png'
                        )
            )

        print(f'Saved {split} images and masks of {ds_path.name} dataset')

if __name__ == '__main__':
    main()
