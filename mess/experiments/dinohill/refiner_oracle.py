import os
from config import datasets_path
os.environ['DETECTRON2_DATASETS'] = str(datasets_path)
import ast
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import tqdm
import time
from numba import jit, prange

from mess.datasets.TorchvisionDataset import TorchvisionDataset, get_detectron2_datasets
from mess.experiments.metrics import compute_tps_fps_tns_fns, compute_global_metrics, aggregate_metrics





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


@jit(nopython=True)
def _upsample_mask(np_orig_img, orig_img_height, orig_img_width, resized_img, ds_mask, row_resize_factor, col_resize_factor, neighborhood_size):
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
    for row in prange(orig_img_height):
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
                w_ab = compute_w_ab(target_value, neighborhood[closest_idx], b_value)
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
        
            

@jit(nopython=True)
def compute_w_ab(target_value, b_value, a_value, eps=1e-3):
    num = np.linalg.norm(target_value - b_value)
    denom = np.linalg.norm(target_value - a_value) + np.linalg.norm(target_value - b_value) + eps
    return num / denom

def upsample_mask(ds_mask: np.ndarray, orig_img: Image, neighborhood_size=3):
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
    return _upsample_mask(np_orig_img, orig_img_height, orig_img_width, resized_img, ds_mask, row_resize_factor, col_resize_factor, neighborhood_size)




def norm(x):
    return (x - x.min()) / (x.max() - x.min())

def visualize(img, gt_mask, synthetic_points, dstdir='tmp'):
    """
    Save the img with the mask overlapping and the synthetic points on top.
    The synthetic points are tuples with (sample_ind, row, col, logit).
    """
    sample_ind = 0
    if not os.path.exists(dstdir):
        os.makedirs(dstdir)
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(np.clip(norm(img) + gt_mask[..., None] * np.array([1., 0., 1.])[None, None] * 0.3, 0, 1))
    for point in synthetic_points:
        sample_ind, row, col, logit = point
        # get the color according to the logit. Logit = -1 is red   Logit = 1 is blue
        if str(logit) in ['True', 'False']:
            logit = 1 if logit else -1
        color = (max(0, -logit), 0, max(0, logit))
        ax.scatter(col, row, color=color, s=100, marker='o')
    plt.savefig(os.path.join(dstdir, f'current.png'))
    plt.close()
    print('showed image', sample_ind)
    pass







def main(dstdir, ds_name):
    assert ds_name in get_detectron2_datasets(), f'{ds_name} not in {get_detectron2_datasets()}'
    print('running', ds_name)
    dstdir = Path(dstdir)
    dstdir.mkdir(exist_ok=True, parents=True)

    ds = TorchvisionDataset(ds_name, transform=np.array, mask_transform=np.array)
    class_indices, class_names = np.arange(len(ds.class_names)), ds.class_names
    values_to_ignore = [255] + [ind for ind, cls_name in zip(class_indices, class_names) if cls_name in ['background', 'others', 'unlabeled', 'background (waterbody)', 'background or trash']]

    dstfile = dstdir / f'metrics_per_class_{ds_name}.json'
    if dstfile.exists(): 
        with open (dstfile, 'r') as f:
            metrics_per_class = ast.literal_eval(f.read())
    else:
        metrics_per_class = {}
    for class_ind, class_name in zip(class_indices, class_names):
        if class_ind in values_to_ignore:
            continue
        print('class', class_name)
        if not class_name in metrics_per_class:
            metrics_per_class[class_name] = {}

        for sample_ind in range(len(ds)):
            print(sample_ind, end='\r')

            img, gt_mask = ds[sample_ind]
            class_mask = gt_mask == class_ind
            if class_mask.sum() == 0:
                metrics_per_class[class_name][sample_ind] = None
                continue

            dstimg = dstdir / f'class_{class_name}_sample_{sample_ind}.png'
            if dstimg.exists():
                upsampled_mask = np.array(Image.open(dstimg)) > 128
                if upsampled_mask.shape != gt_mask.shape:
                    # remove dstimg
                    os.remove(dstimg)
            if not dstimg.exists():
                st = time.time()
                pilimg = Image.fromarray(img)
                pilmask = Image.fromarray(class_mask)
                ds_mask = np.array(pilmask.resize((46,46)))
                upsampled_mask = upsample_mask(ds_mask, pilimg)
                print('upsampled in', time.time() - st)

                Image.fromarray((upsampled_mask * 255).astype(np.uint8)).save(dstimg)

            try:
                metrics = compute_global_metrics(*compute_tps_fps_tns_fns([upsampled_mask], [class_mask]))
            except:
                breakpoint()
            metrics_per_class[class_name][sample_ind] = metrics
        
        with open(dstdir / f'metrics_per_class_{ds_name}.json', 'w') as f:
            f.write(str(metrics_per_class).replace("'", '"'))




        

        



if __name__ == '__main__':
    from fire import Fire
    Fire(main)
    