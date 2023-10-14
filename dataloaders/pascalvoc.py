import tqdm
import numpy as np
import torchvision
from config import datasets_path

# classes
classes = {
    'background': 0,
    'aeroplane': 1,
    'bicycle': 2,
    'bird': 3,
    'boat': 4,
    'bottle': 5,
    'bus': 6,
    'car': 7,
    'cat': 8,
    'chair': 9,
    'cow': 10,
    'diningtable': 11,
    'dog': 12,
    'horse': 13,
    'motorbike': 14,
    'person': 15,
    'potted plant': 16,
    'sheep': 17,
    'sofa': 18,
    'train': 19,
    'tv/monitor': 20,
    'void': 255
}

ds = torchvision.datasets.VOCSegmentation(root=datasets_path / 'pascalvoc',
                                          year='2012',
                                          image_set='train',
                                          download=False)
length = len(ds)

def load_img_only(index):
    return np.array(ds[index][0])

def get_length_and_load_ind_img_mask_fn(class_name, only_pos):
    assert class_name in classes.keys()
    class_ind = classes[class_name]
    if only_pos:
        pos_indices = []
        for i in tqdm.tqdm(range(len(ds))):
            if np.sum(np.array(ds[i][1]) == class_ind) > 0:
                pos_indices.append(i)
        indices = pos_indices
    else:
        indices = list(range(len(ds)))
    def load_ind_img_mask_fn(index):
        img, mask = ds[indices[index]]
        return indices[index], np.array(img), np.array(mask) == class_ind
    return len(indices), load_ind_img_mask_fn


