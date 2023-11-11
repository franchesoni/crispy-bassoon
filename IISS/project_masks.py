from skimage.transform import resize
from copy import deepcopy
import numpy as np
import torchvision.transforms as T
import torch

"""
Here we assign a feature vector to each SAM mask.
For each mask we compute the mean of the features of the pixels in the mask.
Because the features are patch-wise, the mean is over patches weighted by their percentage of inclusion in the mask.
"""
from config import DINO_RESIZE, DINO_PATCH_SIZE

dino_resize_shape = (DINO_RESIZE[0] // DINO_PATCH_SIZE, DINO_RESIZE[1] // DINO_PATCH_SIZE)  # the size of the feature map
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eps = 1e-9

def project_masks_faster(masks, feat):
    feat = torch.tensor(feat).to(device)
    h, w, F = feat.shape
    segs = torch.tensor(np.array([mask['segmentation'] for mask in masks]), dtype=torch.float32).to(device)
    B, H, W = segs.shape

    # Resize operation - using torchvision
    resize = T.Resize(dino_resize_shape)
    rsegs = resize(segs.unsqueeze(1)).squeeze(1)  # Add and remove channel dimension as needed
    assert rsegs.shape == (B, h, w), "should resize to feat resolution"

    # Compute feature vectors
    rsegs_sum = rsegs.sum(dim=(1, 2)).reshape(B, 1, 1, 1) + eps
    masks_feat = ((feat.reshape(1, h, w, F) * rsegs.reshape(B, h, w, 1)) / rsegs_sum).sum(dim=(1, 2)).reshape(B, F)


    return masks_feat.cpu().numpy()

def project_masks(masks, feat):
    """
    For each SAM mask in the list of masks (each of which is a dict)we compute the feature vector that represents it by taking a weighted mean. The function could be made faster by vectorizing it."""
    # obtain segmentations
    segs = [mask['segmentation'] for mask in masks]
    # resize all segmentations
    rsegs = [resize(seg*1., dino_resize_shape, anti_aliasing=True, preserve_range=True) for seg in segs]
    # compute the feature vector for each segmentation
    masks_feat = [(feat * rseg[..., None] / (rseg.sum()+eps)).sum(axis=(0,1)) for rseg in rsegs]
    return masks_feat

def get_complement(mask):
    out = deepcopy(mask)
    out['segmentation'] = np.logical_not(mask['segmentation'])
    out['area'] = out['segmentation'].sum()
    return out

