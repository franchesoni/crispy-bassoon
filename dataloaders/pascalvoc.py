from pathlib import Path
import tqdm
import functools
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
classes_of_interest = [c for c in classes if not c in ['background', 'void']]


ds = torchvision.datasets.VOCSegmentation(root=datasets_path / 'pascalvoc',
                                          year='2012',
                                          image_set='train',
                                          download=False)
length = len(ds)

def load_img_only(index):
    return np.array(ds[index][0])

def get_length_and_load_ind_img_mask_fn(class_name, only_pos, seed):
    assert class_name in classes.keys()
    class_ind = classes[class_name]

    if only_pos:
        if Path(f'dataloaders/cache/{class_ind}_{only_pos}.npy').exists():
            indices = np.load(f'dataloaders/cache/{class_ind}_{only_pos}.npy')
        else:
            pos_indices = []
            for i in tqdm.tqdm(range(len(ds))):
                if np.sum(np.array(ds[i][1]) == class_ind) > 0:
                    pos_indices.append(i)
            indices = pos_indices
            np.save(f'dataloaders/cache/{class_ind}_{only_pos}.npy', indices)
    else:
        indices = list(range(len(ds)))
    if not seed in [None, 0]:
        np.random.seed(seed)
        np.random.shuffle(indices)
    def load_ind_img_mask_fn(index):
        if index >= len(indices):
            raise IndexError(f"index {index} out of bounds for length {len(indices)}")
        img, mask = ds[indices[index]]
        return indices[index], np.array(img), np.array(mask) == class_ind
    return len(indices), load_ind_img_mask_fn

def load_img_with_masks_as_batch_fn(index):
    img, mask = ds[index]
    mask = np.array(mask)
    masks = []
    for c in classes_of_interest:
        if classes[c] in mask:
            class_mask = mask == classes[c]
            masks.append(class_mask)
        else:
            masks.append(None)
    return index, np.array(img), masks


def get_sample_fss_episode_fn(n_episodes, n_shots, class_name, only_pos, seed=None):
    """
    Create a number of few-shot episodes for a given class.
    Each episode is formed by randomly sampling a query image and then randomly sampling `shots` support images (without repetition and that don't include the query)."""
    length, load_ind_img_mask_fn = get_length_and_load_ind_img_mask_fn(class_name, only_pos, seed=None)
    seed = seed if seed else 0

    queries, supports = [], []
    for episode in range(n_episodes):
        np.random.seed(seed*1000 + episode)  # each episode corresopnds to a different seed
        support_inds = np.random.choice(np.arange(length), size=n_shots+1, replace=False)
        query_ind, support_inds = support_inds[0], support_inds[1:]
        queries.append(query_ind)
        supports.append(support_inds)

    def sample_fss_episode_fn(index):
        if index >= n_episodes:
            raise IndexError(f"index {index} out of bounds for n_episodes {n_episodes}")
        query_ind = queries[index]
        support_inds = supports[index]
        query_global_ind, query_img, query_mask = load_ind_img_mask_fn(query_ind)
        support_global_inds, support_imgs, support_masks = [], [], []
        for support_ind in support_inds:
            support_global_ind, support_img, support_mask = load_ind_img_mask_fn(support_ind)
            support_global_inds.append(support_global_ind)
            support_imgs.append(support_img)
            support_masks.append(support_mask)
        return (query_global_ind, query_img, query_mask), list(zip(support_global_inds, support_imgs, support_masks))
        
    return sample_fss_episode_fn








