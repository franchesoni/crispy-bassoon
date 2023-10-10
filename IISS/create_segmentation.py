import numpy as np

def create_segmentation(masks_per_image, labels, clicks):
    # write classified labels to pred image
    pred_masks = []
    ind = 0
    for masks in masks_per_image:
        frame_pred = np.zeros_like(masks[0]['segmentation'], dtype=int)
        for mask in masks:
            frame_pred += mask['segmentation'] * labels[ind]
            ind += 1
        pred_masks.append(frame_pred)

    # overwrite frame_preds from clicks
    for click in clicks:
        frame_ind, mask_ind, is_pos = click
        clicked_region = masks_per_image[frame_ind][mask_ind]['segmentation']
        pred_masks[frame_ind] =  pred_masks[frame_ind] * (~clicked_region) + clicked_region * (2*is_pos - 1)
    pred_masks = [fp > 0 for fp in pred_masks]  # thresold the labels
    return pred_masks