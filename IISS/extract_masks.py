import numpy as np

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor
from config import device, dev, MIN_MASK_REGION_AREA

if dev:
    sam_checkpoint = "sam_vit_b_01ec64.pth"
    model_type = "vit_b"
else:
    sam_checkpoint = "sam_vit_l_0b3195.pth"
    model_type = "vit_l"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
mask_generator = SamAutomaticMaskGenerator(
    sam,
    min_mask_region_area=MIN_MASK_REGION_AREA, points_per_side=32 if not dev else 8,
    crop_n_layers=2 if not dev else 0,
    crop_n_points_downscale_factor=2 if not dev else 1
    )

def extract_masks_single(image):
    """Computes SAM masks for a given image,
    returns a list of dicts"""
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    assert image.dtype == np.uint8
    masks = mask_generator.generate(image)
    masks = remove_small_masks(masks, MIN_MASK_REGION_AREA) 
    return masks

def remove_small_masks(masks, min_mask_region_area):
    """Removes masks with area less than min_mask_region_area"""
    return [mask for mask in masks if mask['area'] >= min_mask_region_area]

def extract_masks_list(images):
    """Computes SAM masks for a list of images,
    returns a list of lists of dicts"""
    masks_per_frame = []
    for imgind, image in enumerate(images):
        print(f'extracting masks for image {imgind+1} of {len(images)}', end='\r')
        masks_per_frame.append(extract_masks_single(image))
    return masks_per_frame

sam.to(device=device)
sam_predictor = SamPredictor(sam)

# given an image we should extract the SAM features.
def get_embedding_sam(img: np.ndarray) -> dict:
    sam_predictor.set_image(img)
    embedding = {}
    embedding["original_size"] = sam_predictor.original_size
    embedding["input_size"] = sam_predictor.input_size
    embedding["features"] = sam_predictor.get_image_embedding()  # torch.Tensor
    return embedding


