from skimage.transform import resize
"""
Here we assign a feature vector to each SAM mask.
For each mask we compute the mean of the features of the pixels in the mask.
Because the features are patch-wise, the mean is over patches weighted by their percentage of inclusion in the mask.
"""
from config import DINO_RESIZE, DINO_PATCH_SIZE

dino_resize_shape = (DINO_RESIZE[0] // DINO_PATCH_SIZE, DINO_RESIZE[1] // DINO_PATCH_SIZE)  # the size of the feature map

def project_masks(masks, feat):
    """
    For each SAM mask in the list of masks (each of which is a dict)we compute the feature vector that represents it by taking a weighted mean. The function could be made faster by vectorizing it."""
    # obtain segmentations
    segs = [mask['segmentation'] for mask in masks]
    # resize all segmentations
    rsegs = [resize(seg*1., dino_resize_shape, anti_aliasing=True, preserve_range=True) for seg in segs]
    # compute the feature vector for each segmentation
    masks_feat = [(feat * rseg[..., None] / rseg.sum()).sum(dim=(0,1)) for rseg in rsegs]
    return masks_feat