import torch

def classify(masks_feat_per_image, clicks):
    """
    Uses the Nadayara-Watson estimator to classify each one of the masks given the clicks. The clicks are tuples (frame, mask_index, label), e.g. they're already associated to a mask.
    Can be made way faster by avoiding the recomputation of distances.
    """
    assert clicks == clean_clicks(clicks), "your click sequence overwrites clicks"
    feats = torch.stack([mask_feat for masks_feat in masks_feat_per_image for mask_feat in masks_feat])  # (N, F)
    clicked_feats = torch.stack([masks_feat_per_image[click[0]][click[1]] for click in clicks])  # (P, F)
    # compue distances
    distances = torch.cdist(feats[None], clicked_feats[None])[0]  # (N, P)
    # get labels
    ann_is_pos = torch.tensor([click[2] for click in clicks])  # (P,)

    # compute probabilities using radial basis function regression
    P, d = distances.shape[1], len(masks_feat_per_image[0][0].flatten())
    r = (P**(1/(2*d))) * distances.max() / 10
    alphas = torch.exp(-(distances / r)**2/2)  # (N, P)
    denom = alphas.sum(dim=1) + 1e-15  # (N,)
    logits = (alphas * (2 * ann_is_pos[None] - 1) ).sum(axis=1) / denom.squeeze()  # (N,), convert labels to -1, 1
    labels = 2 * (logits > 0) - 1
    return labels






def clean_clicks(clicks):
    """
    Removes all clicks that are overwritten later in the sequence.
    """
    coords = [(click[0], click[1]) for click in clicks]
    clicks = [clicks[i] for i in range(len(clicks)) if coords[i] not in coords[i+1:]]
    return clicks
    
