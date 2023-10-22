import torch
import numpy as np
from torchvision import transforms

from config import dev, DINO_RESIZE, DINO_PATCH_SIZE


def get_dino(dino_size = 'small'):
    if dino_size == 'small':
        dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        dino = dinov2_vits14  # small dino
    elif dino_size == 'base':
        dinov2_vitb14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        dino = dinov2_vitb14  # base dino
    else:  
        raise ValueError(f'Unknown dino size: {dino_size}')
    return dino

dino = get_dino()
dino.eval()

def compute_features_list(images):
    """
    Computes DINO features for a list of images,
    we do it in a list because DINO allows for bathing.
    Outputs one feature map per image in the list.
    """
    with torch.no_grad():
        timgs = [preprocess_image_array(img, target_size=DINO_RESIZE) for img in images]
        timgs = torch.concat(timgs, dim=0)
        outs = dino.forward_features(timgs)
        feats = outs['x_norm_patchtokens']
        P = DINO_PATCH_SIZE
        B, C, H, W = timgs.shape
        Ph, Pw = H // P, W // P
        B, PhPw, F = feats.shape
        feats = feats.reshape(B, Ph, Pw, F)
    return feats    


def preprocess_image_array(image_array, target_size):
    assert image_array.dtype == np.uint8
    assert len(image_array.shape) == 3
    assert image_array.shape[2] == 3
    assert image_array.max() > 1 or image_array.max() == 0
    # Step 1: Normalize using mean and std of ImageNet dataset
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_array = (image_array / 255.0 - mean) / std

    # Step 2: Resize the image_array to the target size
    image_array = np.transpose(image_array, (2, 0, 1))  # PyTorch expects (C, H, W) format
    image_tensor = torch.tensor(image_array, dtype=torch.float32)
    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = resize_image_tensor(image_tensor, target_size=target_size)
    return image_tensor

def resize_image_tensor(image_tensor, target_size=(224, 224)):
    transform = transforms.Compose([
        transforms.Resize(target_size),
    ])
    resized_image = transform(image_tensor)
    return resized_image






