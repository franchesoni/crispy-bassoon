import json
import ast
import os
import shutil
from pathlib import Path
import tqdm
import cProfile

import torch
import numpy as np
import matplotlib.pyplot as plt

from IISS.extract_masks import extract_masks_single
from IISS.compute_features import compute_features_list
from IISS.project_masks import project_masks, get_complement
from IISS.classify import classify, clean_clicks
from IISS.create_segmentation import create_segmentation

from metrics import compute_tps_fps_tns_fns, compute_global_metrics

def get_inter_union(y_true, y_pred):
    return np.logical_and(y_true, y_pred).sum(), np.logical_or(y_true, y_pred).sum()

def get_clicked_segment(pred_masks, dinosam_masks, gt_masks):  
    """Based on IoU, return when no more improvement is possible. Cycles could happen."""
    max_iou_improvement = -np.inf  # Initialize to negative infinity
    chosen_frame_ind, chosen_mask_ind, is_pos = None, None, None  # Initialize variables
    
    for frame_ind, fmasks in enumerate(dinosam_masks):
        pred_mask = pred_masks[frame_ind]
        gt_mask = gt_masks[frame_ind]
        current_inter, current_union = get_inter_union(gt_mask, pred_mask)
        current_iou = current_inter / current_union
        
        for mask_ind, mask in enumerate(fmasks):
            # check if this mask can give us a relevant improvement
            max_possible_new_IoU_pos = min((current_inter + mask['area']), current_union) / current_union
            max_possible_improvement_pos = max_possible_new_IoU_pos - current_iou
            max_possible_new_IoU_neg = current_inter / max(current_inter, (current_union - mask['area']))
            max_possible_improvement_neg = max_possible_new_IoU_neg - current_iou
            max_possible_improvement = max(max_possible_improvement_pos, max_possible_improvement_neg)
            if max_possible_improvement <= max_iou_improvement:
                continue

            # if not, compute the improvement it gives when clicked (pos and neg)
            seg = mask['segmentation']
            # Positive mask combination
            new_pos_pred = np.logical_or(pred_mask, seg)
            new_pos_inter, new_pos_union = get_inter_union(gt_mask, new_pos_pred)
            new_pos_iou = new_pos_inter / new_pos_union
            iou_improvement = new_pos_iou - current_iou
            
            if iou_improvement > max_iou_improvement:
                max_iou_improvement = iou_improvement
                chosen_frame_ind = frame_ind
                chosen_mask_ind = mask_ind
                is_pos = True
                
            # Negative mask combination
            new_neg_pred = np.logical_and(pred_mask, np.logical_not(seg))
            new_neg_inter, new_neg_union = get_inter_union(gt_mask, new_neg_pred)
            new_neg_iou = new_neg_inter / new_neg_union
            iou_improvement = new_neg_iou - current_iou
            
            if iou_improvement > max_iou_improvement:
                max_iou_improvement = iou_improvement
                chosen_frame_ind = frame_ind
                chosen_mask_ind = mask_ind
                is_pos = False
                
    if max_iou_improvement <= 0:
        return None
    
    return (chosen_frame_ind, chosen_mask_ind, is_pos)


def get_clicked_segment_acc(pred_masks, dinosam_masks, gt_masks):  # optimize for accuracy
    """Based on error reduction, return when no more improvement is possible. Cycles could happen."""
    max_err_reduction = -np.inf
    for frame_ind, fmasks in enumerate(dinosam_masks):
        pred_mask = pred_masks[frame_ind]
        gt_mask = gt_masks[frame_ind]
        current_err = np.logical_xor(pred_mask, gt_mask).sum()
        for mask_ind, mask in enumerate(fmasks):
            if mask['area'] <= max_err_reduction:  # can't reduce error more
                continue
            seg = mask['segmentation']
            new_pos_pred = np.logical_or(pred_mask, seg)
            new_pos_error = np.logical_xor(new_pos_pred, gt_mask).sum()
            err_reduction = current_err - new_pos_error
            if err_reduction > max_err_reduction:
                max_err_reduction = err_reduction
                chosen_frame_ind = frame_ind
                chosen_mask_ind = mask_ind
                is_pos = True
            # although I thought positive error reduction implies negative error increase, this is false, so we have to check
            new_neg_pred = np.logical_and(pred_mask, np.logical_not(seg))
            new_neg_error = np.logical_xor(new_neg_pred, gt_mask).sum()
            err_reduction = current_err - new_neg_error
            if err_reduction > max_err_reduction:
                max_err_reduction = err_reduction
                chosen_frame_ind = frame_ind
                chosen_mask_ind = mask_ind
                is_pos = False
    if max_err_reduction <= 0:
        return None
    return (chosen_frame_ind, chosen_mask_ind, is_pos)


def get_training_metrics(load_ind_img_mask_fn, precomputed_dir, n_images, runname='tmp'):
    precomputed_dir = Path(precomputed_dir)
    ndigits_pre = len(os.listdir(precomputed_dir)[0].split('_')[2].split('.')[0])
    # load global clicks
    global_clicks_path = Path(runname) / 'global_clicks.json'
    with open(global_clicks_path, 'r') as f:
        content = f.read()
    global_clicks = ast.literal_eval(content)
    clicks = []
    seed_vectors = []
    metrics_vs_t = []
    for i in tqdm.tqdm(range(len(global_clicks))):
        new_click = global_clicks[i]
        masks_feat = np.load(precomputed_dir / f'masks_feat_{str(new_click[0]).zfill(ndigits_pre)}.npy', allow_pickle=True)
        masks_feat = torch.from_numpy(np.array(masks_feat)) 
        new_seed_vector = masks_feat[new_click[1]]

        clicks.append(new_click)
        seed_vectors.append(new_seed_vector)
        del masks_feat
        ann_is_pos = torch.tensor([click[2] for click in clicks])
        
        metrics_vs_j = []
        for j in tqdm.tqdm(range(n_images), leave=False):  # iterate over images
            global_ds_ind, img, gt = load_ind_img_mask_fn(j)
            sam_masks = np.load(precomputed_dir / f'sam_masks_{str(global_ds_ind).zfill(ndigits_pre)}.npy', allow_pickle=True)
            masks_feat = np.load(precomputed_dir / f'masks_feat_{str(global_ds_ind).zfill(ndigits_pre)}.npy', allow_pickle=True)
            masks_feat = torch.from_numpy(np.array(masks_feat)) 
            labels = classify(seed_vectors, ann_is_pos, [masks_feat])
            clicks_so_far_over_stack = [(0, c[1], c[2]) for c in clicks if c[0]==global_ds_ind]
            pred_mask = create_segmentation([sam_masks], labels, clicks_so_far_over_stack)
            metdict = compute_global_metrics(*compute_tps_fps_tns_fns(pred_mask, [gt]))
            metrics_vs_j.append(metdict)
        metrics_vs_t.append(metrics_vs_j)
    # save metrics vs t at runs / runname
    with open(Path(runname) / 'metrics_vs_t.json', 'w') as f:
        f.write(str(metrics_vs_t).replace("'", '"'))



def precompute_for_dataset(ds_load_img_fn, ds_length, dstdir, reset=False):
    ndigits = len(str(ds_length))
    dstdir = Path(dstdir)
    if reset:
        shutil.rmtree(dstdir)
    else:
        if dstdir.exists():
            print('precomputed data already exists, returning...')
            return
    (dstdir).mkdir()

    # start profiling
    pr = cProfile.Profile()
    pr.enable()

    for i in tqdm.tqdm(range(ds_length)):
        img = ds_load_img_fn(i)
        sam_masks = extract_masks_single(img)
        img_features = compute_features_list([img])[0]
        masks_feat = project_masks(sam_masks, img_features)
        np.save(dstdir / f'sam_masks_{str(i).zfill(ndigits)}.npy', sam_masks)
        np.save(dstdir / f'masks_feat_{str(i).zfill(ndigits)}.npy', masks_feat)
        if i == 10:
            # save the profile
            pr.disable()
            pr.dump_stats('precompute.prof')
            del pr
    
def precompute_for_dataset_complements(ds_load_img_fn, ds_length, dstdir):
    ndigits = len(str(ds_length))
    dstdir = Path(dstdir)
    if len(list(dstdir.glob("*complement*"))):
        print('precomputed data already exists, returning...')
        return
    assert dstdir.exists()
    print('precomputing complement masks...')

    for i in tqdm.tqdm(range(ds_length)):
        img = ds_load_img_fn(i)
        sam_masks = np.load(dstdir / f'sam_masks_{str(i).zfill(ndigits)}.npy', allow_pickle=True)
        complement_sam_masks = [get_complement(mask) for mask in sam_masks]
        del sam_masks
        img_features = compute_features_list([img])[0]
        masks_feat = project_masks(complement_sam_masks, img_features)
        np.save(dstdir / f'complement_sam_masks_{str(i).zfill(ndigits)}.npy', complement_sam_masks)
        np.save(dstdir / f'complement_masks_feat_{str(i).zfill(ndigits)}.npy', masks_feat)
 




def run_experiment(load_ind_img_mask_fn, precomputed_dir, n_images, max_total_clicks, stack_size=8, clicks_per_stack=4, runname='tmp', reset=False, single_mode=True):
    """Generic experiment function for all datasets. The data handling should be done in the construction of the load_sample_fn function, which takes an index and returns an image and a mask.
    The number of images sets the range in which load_sample_fn is called.
    
    The feature extraction should be done once only and is independent of the ground truth or target."""
    if single_mode:
        assert stack_size == 1
        assert clicks_per_stack == 1
    precomputed_dir = Path(precomputed_dir)
    assert precomputed_dir.exists()
    ndigits_pre = len(os.listdir(precomputed_dir)[0].split('_')[2].split('.')[0])
    assert n_images % stack_size == 0, 'you should set a nice stack size for simplicity'  # else change the range in the for loop and also make sure the number of clicks on the last stack is correct

    ndigits = len(str(n_images))
    dstdir = Path(runname)
    try:
        dstdir.mkdir(parents=True)
    except FileExistsError:
        if reset:
            shutil.rmtree(dstdir)
            dstdir.mkdir()
            print('removed last run')
        else:
            print('run already exists and not resetting...')
            return

    seed_vectors = []
    all_clicks_so_far = []
    sweeps = 0
    (dstdir / 'metrics').mkdir(exist_ok=not reset)
    while len(all_clicks_so_far) < max_total_clicks:  # sweep through the dataset
        for stack_ind in range(n_images // stack_size):  # the stack index in the sub dataset
            # build image-stack
            subds_indices_in_stack = list(range(stack_ind * stack_size, (stack_ind + 1) * stack_size))
            imgs, gt_masks, sam_masks_per_frame, masks_feat_per_frame = [], [], [], []
            global_ds_indices_in_stack = []
            for subds_ind_in_stack in subds_indices_in_stack:
                global_ds_ind, img, gt = load_ind_img_mask_fn(subds_ind_in_stack)
                sam_masks = np.load(precomputed_dir / f'sam_masks_{str(global_ds_ind).zfill(ndigits_pre)}.npy', allow_pickle=True)
                masks_feat = np.load(precomputed_dir / f'masks_feat_{str(global_ds_ind).zfill(ndigits_pre)}.npy', allow_pickle=True)
                imgs.append(img)
                gt_masks.append(gt)
                sam_masks_per_frame.append(sam_masks)
                masks_feat_per_frame.append(masks_feat)
                global_ds_indices_in_stack.append(global_ds_ind)
            
            masks_feat_per_frame = [torch.from_numpy(np.array(masks_features)) for masks_features in masks_feat_per_frame]

            # we need to keep track of a few indices: global dataset index, indexes the images on the dataset, (ii) stack index, indexes the frames on the stack
            # we have to classify the stack based on global clicks, but for that we need the seed vectors corresponding to the global clicks and we need to modify the classify function

            if len(seed_vectors) == 0:
                pred_masks = [np.zeros_like(mask) for mask in gt_masks]  # init at 0
            else:
                # classify masks in stack
                # assert clean_clicks(all_clicks_so_far) == all_clicks_so_far, "your click sequence overwrites clicks"
                ann_is_pos = torch.tensor([click[2] for click in all_clicks_so_far])  # (P,)
                labels = classify(seed_vectors, ann_is_pos, masks_feat_per_frame)
                # paint masks
                clicks_so_far_over_stack = [(global_ds_indices_in_stack.index(c[0]), c[1], c[2]) for c in all_clicks_so_far if c[0] in global_ds_indices_in_stack]
                pred_masks = create_segmentation(sam_masks_per_frame, labels, clicks_so_far_over_stack)

            # now we make a click, save the click, classify the stack, make a new click, save the click, classify the stack, and so on, before going to a new stack. Then we save the sequence of clicks which we'll use to compute the global error. 

            # save performance with no clicks and start loop
            stack_metrics = [compute_global_metrics(*compute_tps_fps_tns_fns(pred_masks, gt_masks))]

            for ind_in_stack in range(clicks_per_stack):
                print(f'sweep {sweeps} | stack {stack_ind+1}/{n_images // stack_size} | click {ind_in_stack+1}/{clicks_per_stack}', end = '\r')
                clicked_segment = get_clicked_segment(pred_masks, sam_masks_per_frame, gt_masks)  # click the mask that reduces the error the most, (frame, mask_index, label)
                if clicked_segment is None:
                    break  # no click improves the error
                all_clicks_so_far.append((global_ds_indices_in_stack[clicked_segment[0]], clicked_segment[1], clicked_segment[2]))
                seed_vectors.append(masks_feat_per_frame[clicked_segment[0]][clicked_segment[1]])

                # classify masks in stack
                # assert clean_clicks(all_clicks_so_far) == all_clicks_so_far, "your click sequence overwrites clicks"
                ann_is_pos = torch.tensor([click[2] for click in all_clicks_so_far])  # (P,)
                labels = classify(seed_vectors, ann_is_pos, masks_feat_per_frame)
                # paint masks
                clicks_so_far_over_stack = [(global_ds_indices_in_stack.index(c[0]), c[1], c[2]) for c in all_clicks_so_far if c[0] in global_ds_indices_in_stack]
                pred_masks = create_segmentation(sam_masks_per_frame, labels, clicks_so_far_over_stack)

                stack_metdict = compute_global_metrics(*compute_tps_fps_tns_fns(pred_masks, gt_masks))
                stack_metrics.append(stack_metdict)
                assert sweeps == 0
                metrics_dstfile = f'{dstdir}/metrics/sweep_{sweeps}_stack_{str(stack_ind).zfill(ndigits)}_metrics.json' if not single_mode else f'{dstdir}/metrics/image_{str(stack_ind).zfill(ndigits)}_metrics.json'
                with open(metrics_dstfile, 'w') as f:
                    f.write(str(stack_metrics).replace("'", '"'))
            with open(dstdir / 'global_clicks.json', 'w') as f:
                f.write(str([list(c) for c in all_clicks_so_far]).replace("'", '"'))
        sweeps += 1
            

