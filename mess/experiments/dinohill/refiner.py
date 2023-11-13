import numpy as np
from PIL import Image
from pathlib import Path
import time
import torch
import torchvision.transforms as transforms


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



"""Here we provide code to upsample a mask given by classifying patches of an image.
Because we use DINO, the usual pipeline is
    1. resizing the input image to (644, 644)
    2. computing the features for the image, with output shape (46,46,F)
    3. classifying the patches, with output shape (46,46,1), this might be a probability or be binary
    4. and now we do image guided upsampling, with steps:
        4.1. resize the image to (644,644,3)
        4.2. downsample the resized image by taking the mean of each patch, with output shape (46,46,3)
        4.3. compute the coefficients that take the downsampled image to the original image, with output shape (H,W,3)
        4.4. apply those coefficients to upsample the mask, with output shape (H,W)
    """

def downsample_image(image, patch_size=14):
    # Assuming image is a PyTorch tensor of shape (H, W, C) and on GPU
    # Unfold splits the image into patches and takes the mean
    # Reshape and permute are used to get the shape right
    unfolded = image.unfold(0, patch_size, patch_size).unfold(1, patch_size, patch_size)
    downsampled = torch.mean(unfolded.float(), dim=(-1,-2))
    return downsampled

def _upsample_mask2(torch_orig_img, orig_img_height, orig_img_width, resized_img, ds_mask, row_resize_factor, col_resize_factor, neighborhood_size):
    pad = neighborhood_size // 2
    assert ds_mask.shape == (46, 46)
    H, W = orig_img_height, orig_img_width

    downsampled_img = downsample_image(resized_img)
    ds_img = torch.zeros((46 + 2 * pad, 46 + 2 * pad, 3), device=device)
    ds_img[pad:-pad, pad:-pad] = downsampled_img

    # Generate grid for row and column indices
    rows = torch.arange(H, device=device) 
    cols = torch.arange(W, device=device) 
    grid_row, grid_col = torch.meshgrid(rows, cols, indexing="ij")  # H, W

    # Map positions to the padded ds image
    ds_rows = (grid_row.float() * row_resize_factor).to(torch.int64) + pad  # H, W
    ds_cols = (grid_col.float() * col_resize_factor).to(torch.int64) + pad  # H, W

    # Get neighborhoods
    neighborhoods = ds_img[(ds_rows.reshape(H,W,1) + torch.arange(-pad, pad + 1, device=device)).reshape(H,W,2*pad+1,1), ds_cols.reshape(H, W, 1, 1) + torch.arange(-pad, pad + 1, device=device)]
    N = (2 * pad + 1)**2
    neighborhoods = neighborhoods.reshape(H, W, N, 3)

    # Calculate closest indices
    target_values = torch_orig_img.reshape(H, W, 1, 3)  # Add one dimension for broadcasting
    diffs = ((neighborhoods - target_values) ** 2).sum(dim=3)  # H, W, N
    closest_indices = torch.argmin(diffs, dim=2)  # H, W
    a_values = torch.gather(neighborhoods, -2, closest_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, 3)).squeeze(-2)  # magic! H, W, 3

    def compute_w_ab_vectorized(target_values, a_values, b_values, eps=1e-3):
        # target_values: H, W, 1, 3
        # a_values: H, W, 3
        # b_values: H, W, N, 3
        num = torch.norm(target_values - b_values, dim=3)  # H, W, N
        denom = torch.norm(target_values.reshape(H, W, 3) - a_values) + num + eps
        return num / denom

    # Compute coefficients
    w_abs = compute_w_ab_vectorized(target_values, a_values, neighborhoods)  # H, W, N
    interpolated_values = w_abs.reshape(H, W, N, 1) * a_values.reshape(H, W, 1, 3) + (1 - w_abs.reshape(H, W, N, 1)) * neighborhoods  # H, W, N, 3
    losses = torch.norm(interpolated_values - target_values, dim=3)  # H, W, N
    b_indices = torch.argmin(losses, dim=2)  # H, W
    best_w_abs = torch.gather(w_abs, -1, b_indices.unsqueeze(-1)).squeeze(-1)  # check!

    coefficients = torch.stack((best_w_abs, closest_indices, b_indices), dim=2)  # H, W, 3

    # Apply coefficients to upsample the mask
    padded_ds_mask = torch.zeros((pad * 2 + 46, pad * 2 + 46), device=device)
    padded_ds_mask[pad:-pad, pad:-pad] = ds_mask

    # Upsampling
    neighborhood_masks = padded_ds_mask[(ds_rows.reshape(H,W,1) + torch.arange(-pad, pad + 1, device=device)).reshape(H,W,2*pad+1,1), ds_cols.reshape(H, W, 1, 1) + torch.arange(-pad, pad + 1, device=device)]
    neighborhood_masks = neighborhood_masks.reshape(H, W, N)

    a_indices = coefficients[..., 1].to(torch.int64)
    b_indices = coefficients[..., 2].to(torch.int64)
    w_abs = coefficients[..., 0]

    a_values = torch.gather(neighborhood_masks, -1, a_indices.unsqueeze(-1)).squeeze(-1)
    b_values = torch.gather(neighborhood_masks, -1, b_indices.unsqueeze(-1)).squeeze(-1)

    upsampled_mask = w_abs * a_values + (1 - w_abs) * b_values

    return upsampled_mask


def _upsample_mask(torch_orig_img, orig_img_height, orig_img_width, resized_img, ds_mask, row_resize_factor, col_resize_factor, neighborhood_size):
    # get some useful values and cast
    pad = neighborhood_size//2
    assert ds_mask.shape == (46,46)

    downsampled_img = downsample_image(resized_img)
    ds_img = torch.zeros((46+2*pad, 46+2*pad, 3), device=device)
    ds_img[pad:-pad,pad:-pad] = downsampled_img

    # now compute the coefficients that take the downsampled image to the original image
    # this involves finding, for each pixel in the original image, the best interpolation in the downsampled image
    coefficients = torch.empty((orig_img_height, orig_img_width, 3), device=device)
    for row in range(orig_img_height):
        print('row', row, '/', orig_img_height, end='\r')
        for col in range(orig_img_width):
            # get target value that should be interpolated
            target_value = torch_orig_img[row,col]

            # map the current position to the padded ds image
            ds_row = int(row * row_resize_factor)
            ds_col = int(col * col_resize_factor)

            # now get the neighborhood on the ds_image that's around (ds_row, ds_col)
            neighborhood = ds_img[ds_row: 2*pad+ds_row+1, ds_col: 2*pad+ds_col+1]
            neighborhood = neighborhood.reshape(-1, 3)

            diffs = ((neighborhood - target_value.reshape(1,-1))**2).sum(dim=1)
            closest_idx = torch.argmin(diffs)  
            a_index, a_value = closest_idx, neighborhood[closest_idx]
            # now compute, for each other value, the interpolation coefficient w_ab and the loss
            w_abs = compute_w_ab_vectorized(target_value, a_value, neighborhood) 
            interpolated_values = w_abs[:,None] * a_value + (1-w_abs[:,None]) * neighborhood
            losses = torch.norm(interpolated_values - target_value, dim=1)
            b_index = torch.argmin(losses, dim=0)
            best_w_ab = w_abs[b_index]
            coefficients[row,col] = torch.tensor([best_w_ab, a_index, b_index])
    padded_ds_mask = torch.zeros((pad*2+46, pad*2+46), device=device)
    padded_ds_mask[pad:-pad,pad:-pad] = ds_mask

    # now apply the coefficients to upsample the mask
    upsampled_mask = torch.zeros((orig_img_height, orig_img_width), device=device)
    for row in range(orig_img_height):
        print('row', row, '/', orig_img_height, end='\r')
        for col in range(orig_img_width):
            # map the current position to the padded ds mask
            ds_row = int(row * row_resize_factor)
            ds_col = int(col * col_resize_factor)
            neighborhood = padded_ds_mask[ds_row: 2*pad+ds_row+1, ds_col: 2*pad+ds_col+1]
            neighborhood = neighborhood.reshape(-1)
            # get coefficients
            w_ab, a_index, b_index = coefficients[row,col]
            # upsample
            upsampled_mask[row,col] = w_ab * neighborhood[int(a_index)] + (1-w_ab) * neighborhood[int(b_index)]
    return upsampled_mask
        
def compute_w_ab_vectorized(target_value, a_value, b_values, eps=1e-3):
    num = torch.norm(target_value[None] - b_values, dim=1)
    denom = torch.norm(target_value - a_value) + torch.norm(target_value[None] - b_values, dim=1) + eps
    return num / denom

def upsample_mask(ds_mask: np.ndarray, orig_img: Image, neighborhood_size=3, mode='vectorized'):
    """Guided Linear Upsampling
    `ds_mask`: binary mask with values {False, True}
    `orig_img`: original image with size (H, W)
    """
    torch_orig_img = torch.from_numpy(np.array(orig_img)).to(device)
    orig_img_height, orig_img_width = orig_img.height, orig_img.width
    row_resize_factor = (644 / orig_img_height) * (46 / 644)
    col_resize_factor = (644 / orig_img_width) * (46 / 644)
    # resize orig image to 644,644 and downsample by averaging each 14,14 patch
    resized_img = torch.from_numpy(np.array(orig_img.resize((644,644)))).to(device)
    ds_mask = torch.from_numpy(ds_mask).to(device)
    if mode=='vectorized':
        return _upsample_mask2(torch_orig_img, orig_img_height, orig_img_width, resized_img, ds_mask, row_resize_factor, col_resize_factor, neighborhood_size)
    return _upsample_mask(torch_orig_img, orig_img_height, orig_img_width, resized_img, ds_mask, row_resize_factor, col_resize_factor, neighborhood_size)


def _upsample_mask_np(np_orig_img, orig_img_height, orig_img_width, resized_img, ds_mask, row_resize_factor, col_resize_factor, neighborhood_size):
    # get some useful values and cast
    pad = neighborhood_size//2
    assert ds_mask.shape == (46,46)

    downsampled_img = np.zeros((46,46,3))
    for row in range(46):
        for col in range(46):
            for c in range(3):
                downsampled_img[row,col, c] = np.mean(resized_img[14*row:14*(row+1), 14*col:14*(col+1), c])
    ds_img = np.zeros((46+2*pad, 46+2*pad, 3))
    ds_img[pad:-pad,pad:-pad] = downsampled_img

    # now compute the coefficients that take the downsampled image to the original image
    # this involves finding, for each pixel in the original image, the best interpolation in the downsampled image
    # print('finging coefficients...')
    coefficients = np.empty((orig_img_height, orig_img_width, 3))
    for row in range(orig_img_height):
        # print('row', row, '/', orig_img_height, end='\r')
        for col in range(orig_img_width):
            # get target value that should be interpolated
            target_value = np_orig_img[row,col]

            # map the current position to the padded ds image
            ds_row = int(row * row_resize_factor)
            ds_col = int(col * col_resize_factor)

            # now get the neighborhood on the ds_image that's around (ds_row, ds_col)
            # neighborhood = ds_img[pad+ds_row-pad: pad+ds_row+pad+1, pad+ds_col-pad: pad+ds_col+pad+1].reshape(-1,3)
            # Before reshaping, make the array contiguous
            neighborhood = np.ascontiguousarray(ds_img[ds_row: 2*pad+ds_row+1, ds_col: 2*pad+ds_col+1])
            neighborhood = neighborhood.reshape(-1, 3)

            diffs = (neighborhood - target_value.reshape(1,-1))
            diffs = diffs * diffs
            diffs = diffs.sum(axis=1)
            closest_idx = np.argmin(diffs)  
            a_index, a_value = closest_idx, neighborhood[closest_idx]
            # now compute, for each other value, the interpolation coefficient w_ab and the loss
            best_loss = 1e9
            for idx in range(len(neighborhood)):
                b_value = neighborhood[idx]
                w_ab = compute_w_ab_np(target_value, neighborhood[closest_idx], b_value)
                loss = np.linalg.norm(w_ab * a_value + (1-w_ab) * b_value - target_value)
                if loss < best_loss:
                    best_loss = loss
                    b_index = idx
                    best_w_ab = w_ab
            coefficients[row,col] = np.array([best_w_ab, a_index, b_index])
    padded_ds_mask = np.zeros((pad*2+46, pad*2+46))
    padded_ds_mask[pad:-pad,pad:-pad] = ds_mask

    # now apply the coefficients to upsample the mask
    upsampled_mask = np.zeros((orig_img_height, orig_img_width))
    for row in range(orig_img_height):
        # print('row', row, '/', orig_img_height, end='\r')
        for col in range(orig_img_width):
            # map the current position to the padded ds mask
            ds_row = int(row * row_resize_factor)
            ds_col = int(col * col_resize_factor)
            neighborhood = np.ascontiguousarray(padded_ds_mask[ds_row: 2*pad+ds_row+1, ds_col: 2*pad+ds_col+1])
            neighborhood = neighborhood.reshape(-1)
            # get coefficients
            w_ab, a_index, b_index = coefficients[row,col]
            # upsample
            upsampled_mask[row,col] = w_ab * neighborhood[int(a_index)] + (1-w_ab) * neighborhood[int(b_index)]
    return upsampled_mask
        
            

def compute_w_ab_np(target_value, b_value, a_value, eps=1e-3):
    num = np.linalg.norm(target_value - b_value)
    denom = np.linalg.norm(target_value - a_value) + np.linalg.norm(target_value - b_value) + eps
    return num / denom

def upsample_mask_np(ds_mask: np.ndarray, orig_img: Image, neighborhood_size=3):
    """Guided Linear Upsampling
    `ds_mask`: binary mask with values {False, True}
    `orig_img`: original image with size (H, W)
    """
    np_orig_img = np.array(orig_img)
    orig_img_height, orig_img_width = orig_img.height, orig_img.width
    row_resize_factor = (644 / orig_img_height) * (46 / 644)
    col_resize_factor = (644 / orig_img_width) * (46 / 644)
    # resize orig image to 644,644 and downsample by averaging each 14,14 patch
    resized_img = np.array(orig_img.resize((644,644)))
    return _upsample_mask_np(np_orig_img, orig_img_height, orig_img_width, resized_img, ds_mask, row_resize_factor, col_resize_factor, neighborhood_size)





def minmaxnorm(x):
    return (x - x.min()) / (x.max() - x.min())


def main():
    grabcut_path = Path('/home/franchesoni/crispy-bassoon/refiner/GrabCut')
    image_filenames = sorted(list(grabcut_path.glob('data_GT/*')))
    mask_filenames = sorted(list(grabcut_path.glob('boundary_GT/*')))
    assert [img_f.stem == mask_f.stem for img_f, mask_f in zip(image_filenames, mask_filenames)]

    ind = 0
    img = Image.open(image_filenames[ind])  # [0, 255]
    mask = Image.open(mask_filenames[ind])  # {0, 255}
    ds_mask = np.array(mask.resize((46,46))) > 128  # {False, True}

    st = time.time()
    upsampled_mask = upsample_mask(ds_mask, img, mode='vectorized').cpu().numpy()
    print('time:', time.time() - st)

    st = time.time()
    upsampled_mask2 = upsample_mask_np(ds_mask, img)
    print('time:', time.time() - st)


    img.save('tmp/img.png')
    mask.save('tmp/mask.png')
    Image.fromarray(ds_mask).save('tmp/dsmask.png')
    Image.fromarray((upsampled_mask * 255).astype(np.uint8)).save('tmp/res.png')
    Image.fromarray((upsampled_mask2 * 255).astype(np.uint8)).save('tmp/res2.png')


if __name__ == '__main__':
    main()