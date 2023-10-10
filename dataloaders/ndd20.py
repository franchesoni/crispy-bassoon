import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from pathlib import Path


class_dict = {"dolphin": 1, "whale": 2, "other": 3}


# create a function that loads one image and its mask given the dataset path and the index
def load_sample(dataset_path: str, index: int) -> tuple[np.ndarray, np.ndarray]:
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


def visualize_rawfile(dataset_path, index):
    img, mask = load_sample(dataset_path, index)
    plt.figure()
    plt.imshow(img)
    plt.figure()
    plt.imshow(mask)
