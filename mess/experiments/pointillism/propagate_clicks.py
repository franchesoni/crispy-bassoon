import torch
import numpy as np
from torchvision import transforms
from skimage.feature import peak_local_max

DINO_SIZE = 'small'  # or 'base'
DINO_RESIZE = (644, 644)
DINO_PATCH_SIZE = 14

# for DINOv2 propagation
MIN_DISTANCE = 5
R_SCALE = 15

### RBF ### 
def get_distances_from_point(points, point):
  assert len(points.shape) == 2, 'points should be (N, F)'
  assert len(point.shape) == 1, 'point should be (F,)'
  assert point.shape[0] == points.shape[1], 'point should be (F,)'
  distances = torch.norm(points - point[None], p=2, dim=1, keepdim=False)  # (N,)
  return distances

def phi(z):
  return np.exp(-z**2 / 2)

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def compute_probs(distances, ys, d):
  """
  Distances is a list with P tensors of shape (N,)
  ys is a tensor of shape (P,) corresponding to the labels of each point as (False, True)
  d is the dimension of the feature space
  """
  distances = np.stack(distances, axis=0).T  # (N, P)
  assert len(distances.shape) == 2, 'distances should be (N, P) where P is number of points'
  P = distances.shape[1]
  r = (P ** (1 / (2 * d)))  # from learning from data, but gets the scale wrong
  r = r * distances.max() / R_SCALE  # 746*2 is the smallest (in magnitude) integer such that np.e**(-x) is 0, i.e. we scale according to the greatest distance
  print('using r =', r, f'max dist is {distances.max()}')
  alphas = phi(distances / r)  # (N, P)
  denom = alphas.sum(axis=1) + 1e-15  # (N,)
  prob = (alphas * (2 * ys[None] - 1) ).sum(axis=1) / denom.squeeze()  # (N,), convert labels to -1, 1
#   prob = sigmoid(gs)
  return prob  # (N,)






### PROPAGATION ###
def get_propagate_state_dino(dino, imgs):
    z = get_embeddings_DINOv2(dino, imgs, target_size=DINO_RESIZE)
    row_factors = [DINO_RESIZE[0] / img.shape[0] for img in imgs]
    col_factors = [DINO_RESIZE[1] / img.shape[1] for img in imgs]
    return {'z': z, 'row_factors': row_factors, 'col_factors': col_factors, 'dist_from_clicks': []}

def get_dino(dino_size=DINO_SIZE):
    if dino_size == 'small':
        dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        dino = dinov2_vits14  # small dino
    elif dino_size == 'base':
        dinov2_vitb14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        dino = dinov2_vitb14  # base dino
    else:  
        raise ValueError(f'Unknown dino size: {dino_size}')
    return dino

def get_embeddings_DINOv2(dino, imgs, target_size):
  with torch.no_grad():
    timgs = [preprocess_image_array(img, target_size=target_size) for img in imgs]
    timgs = torch.concatenate(timgs, dim=0)

    dino.eval()
    outs = dino.forward_features(timgs)

    feats = outs['x_norm_patchtokens']
    P = DINO_PATCH_SIZE 
    B, C, H, W = timgs.shape
    Ph, Pw = H // P, W // P
    B, PhPw, F = feats.shape
    feats = feats.reshape(B, Ph, Pw, F)
    return feats

def preprocess_image_array(image_array, target_size):
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

def propagate_dino(clicks, state):
    # get state and shapes
    P = len(clicks)
    z = state['dino']['z']  # also have row factos col factors
    B, pH, pW, F = z.shape
    dist_from_clicks = state['dino']['dist_from_clicks']  # (P-1, N) as a list of arrays
    rowf, colf = state['dino']['row_factors'], state['dino']['col_factors']
    sam_masks = state['sam_masks']

    # scale clicks
    sclicks = [[click[0], int(click[1] * rowf[click[0]] // DINO_PATCH_SIZE), int(click[2] * colf[click[0]] // DINO_PATCH_SIZE), click[3]] for click in clicks]
    # compute distance for the new click
    assert len(dist_from_clicks) == P - 1
    scaled_last_click = sclicks[-1] 
    # scale clicks
    # get the embedding at the click position
    z_at_click = z[scaled_last_click[0], scaled_last_click[1], scaled_last_click[2]]
    # compute distance
    new_dist = np.array(get_distances_from_point(z.reshape(-1, F), z_at_click))  # (N,)
    dist_from_clicks.append(new_dist)  # (P, N)
    state['dino']['dist_from_clicks'] = dist_from_clicks  # update state

    # compute probabilities
    Pz = compute_probs(dist_from_clicks, np.array([click[3] for click in clicks]), d=F)  # (N,)
    Pz = Pz.reshape(B, pH, pW)
    import matplotlib.pyplot as plt
    [plt.imsave(f'prob15_{i}.png', Pz[i]) for i in range(len(Pz))]

    # get certainty
    certainty = np.abs(Pz - 0.5).squeeze()

    # get local extrema
    local_extrema = []
    for frame_ind, cer_img in enumerate(certainty):
      # add some noise to break ties
      cer_img = cer_img + np.random.randn(cer_img.shape[0], cer_img.shape[1]) * np.diff(np.sort(np.unique(certainty))).min() / 10
      lmax = peak_local_max(cer_img, min_distance=MIN_DISTANCE)
      if len(lmax):
        lmax = np.concatenate([np.ones((lmax.shape[0],1), dtype=np.uint8) * frame_ind, lmax], axis=1)
        local_extrema.append(lmax)
    local_extrema = np.concatenate(local_extrema, axis=0)

    # sort according to certainties
    local_extrema_certainties = np.take(certainty.ravel(), np.ravel_multi_index(local_extrema.T, certainty.shape)) # the output array of shape (N,)
    local_extrema_certainties_indices = np.argsort(local_extrema_certainties)[::-1]  # decreasing
    local_extrema = local_extrema[local_extrema_certainties_indices]  # (N, 3) <- each row has (frame_ind, row, col)

    # filter out inputs (we're using only one click)
    sclicks_coords = np.array([c[:3] for c in sclicks])  # only one click
    is_input = np.all(sclicks_coords[None, :, :] == local_extrema[:, None, :], axis=2).any(axis=1)
    local_extrema = local_extrema[~is_input]

    # add labels for each point
    local_extrema_probs = np.take(Pz.ravel(), np.ravel_multi_index(local_extrema.T, Pz.shape)) 
    local_extrema = np.concatenate([local_extrema, local_extrema_probs[:, None]> 0.5], axis=1)
    s_F = local_extrema[np.abs(local_extrema_probs-0.5)>0.49]
    s_F = [(c[0],
        int((c[1]*DINO_PATCH_SIZE+(DINO_PATCH_SIZE/2)) / rowf[c[0]]),
        int((c[2]*DINO_PATCH_SIZE+(DINO_PATCH_SIZE/2)) / colf[c[0]]),
        c[3]
        ) for c in s_F]


    # we keep s_F and state
    shapes = [sam_masks[i][0]['segmentation'].shape for i in range(len(sam_masks))]
    pred_masks = [np.zeros(shapes[i], dtype=np.int8) for i in range(len(shapes))]
    for click in clicks + s_F:
        # update predictions
        click_frame = click[0]
        smallest_area = np.inf
        chosen = False
        for sam_ind, sam_mask_dict in enumerate(sam_masks[click_frame]):
            if sam_mask_dict['segmentation'][click[1], click[2]] == 1:
                if sam_mask_dict['area'] < smallest_area:
                    smallest_area = sam_mask_dict['area']
                    chosen_sam_ind = sam_ind
                chosen = True
        if chosen:
            chosen_sam_mask = sam_masks[click_frame][chosen_sam_ind]['segmentation']
            pred_masks[click_frame] = pred_masks[click_frame] + (chosen_sam_mask) * (2*click[3] - 1)  # add if pos, substract if neg
    # make masks boolean
    pred_masks = [(pred_mask>0).astype(bool) for pred_mask in pred_masks]
    return pred_masks, state, s_F







