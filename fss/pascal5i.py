import numpy as np
import tqdm
from pathlib import Path
import torch
import matplotlib.pyplot as plt
"""This experiment is few shot segmentation over the pascal dataset.
Usually the dataset is divided in training and validation folds but our method needs no training. We'll randomly sample 100 episodes for each class for 1-shot and 5-shot problems and evaluate our performance there."""

from dataloaders.pascalvoc import get_sample_fss_episode_fn, classes_of_interest
from IISS.create_segmentation import create_segmentation
from IISS.classify import classify
from metrics import compute_global_metrics, compute_tps_fps_tns_fns, aggregate_metrics

def run_fss_experiment():
    """
    Few shot segmentation experiment.
    The cls_oracle is an oracle that knows the ground truth and classifies each mask accordingly. The seg_oracle is an oracle that adds the ground truth mask in between the candidate masks. 
    """
    ndigits_pre = 4
    precomputed_dir = Path('runs/pascalvoc')
    n_episodes=100
    for pascal_class in classes_of_interest:
        for n_shots in [1, 5]:
            # get episode
            sample_fss_episode = get_sample_fss_episode_fn(n_episodes=n_episodes, n_shots=n_shots, class_name=pascal_class, only_pos=True, seed=None)
            all_metrics, cls_oracle_all_metrics, seg_oracle_all_metrics, seg_cls_oracle_all_metrics = [], [], [], []
            for episode in tqdm.tqdm(range(n_episodes)):
                # get support and query
                (query_global_ind, query_img, query_mask), support = sample_fss_episode(episode)
                # get masks
                img_indices = [query_global_ind] + [sup[0] for sup in support]
                sam_masks_per_frame = [
                    np.load(precomputed_dir / f'sam_masks_{str(img_ind).zfill(ndigits_pre)}.npy', allow_pickle=True)
                    for img_ind in img_indices
                    ]
                # get an extra sam mask to compute an oracle
                query_mask_sam_format = {'segmentation': query_mask, 'area': query_mask.sum()} 

                # compute support labels
                ann_is_pos = []
                for supp_ind, sam_masks in enumerate(sam_masks_per_frame[1:]):
                    gt_mask = support[supp_ind][2]  # gt for this support image
                    for mask_dict in sam_masks:
                        mask = mask_dict['segmentation']
                        ann_is_pos.append(np.sum(mask * gt_mask) / np.sum(mask) > 0.5)

                # compute query labels for oracle
                cls_oracle_labels = []
                gt_mask = query_mask
                for mind, mask_dict in enumerate(sam_masks_per_frame[0]):
                    mask = mask_dict['segmentation']
                    if np.sum(mask * gt_mask) / np.sum(mask) > 0.5:
                        cls_oracle_labels.append(1)
                    else:
                        cls_oracle_labels.append(-1)

                # load feat vectors of support images
                support_vectors = []
                for global_supp_ind in img_indices[1:]:
                    support_vectors += list(np.load(precomputed_dir / f'masks_feat_{str(global_supp_ind).zfill(ndigits_pre)}.npy', allow_pickle=True))
                support_vectors = torch.from_numpy(np.array(support_vectors))
                
                # load feat vectors of query image
                query_vectors = torch.from_numpy(np.load(precomputed_dir / f'masks_feat_{str(query_global_ind).zfill(ndigits_pre)}.npy', allow_pickle=True))

                # classify query
                labels = classify(seed_vectors=list(support_vectors), ann_is_pos=torch.tensor(ann_is_pos), masks_feat_per_frame=[query_vectors])

                # compute prediction 
                ## our method
                pred = create_segmentation(sam_masks_per_frame[:1], labels, [])
                ## if we classify every mask correctly
                cls_oracle_pred = create_segmentation(sam_masks_per_frame[:1], cls_oracle_labels, [])
                ## if we add the ground truth mask classified correcty
                seg_oracle_pred = create_segmentation([[query_mask_sam_format]+list(sam_masks_per_frame[0])], [1]+list(labels), [])
                ## if we add a the ground truth and classify every mask correctly 
                seg_cls_oracle_pred = create_segmentation([[query_mask_sam_format]+list(sam_masks_per_frame[0])], [1]+cls_oracle_labels, [])
            
                # compute performance
                metrics = compute_global_metrics(*compute_tps_fps_tns_fns(pred, [query_mask]))
                cls_oracle_metrics = compute_global_metrics(*compute_tps_fps_tns_fns(cls_oracle_pred, [query_mask]))
                seg_oracle_metrics = compute_global_metrics(*compute_tps_fps_tns_fns(seg_oracle_pred, [query_mask]))
                seg_cls_oracle_metrics = compute_global_metrics(*compute_tps_fps_tns_fns(seg_cls_oracle_pred, [query_mask]))

                all_metrics.append(metrics)
                cls_oracle_all_metrics.append(cls_oracle_metrics)
                seg_oracle_all_metrics.append(seg_oracle_metrics)
                seg_cls_oracle_all_metrics.append(seg_cls_oracle_metrics)


            # plot only the last episode of the experiment
            # save support images with overlapped gt_masks and query image with overlapped prediction on the tmp dir
            Path('tmp').mkdir(exist_ok=True)
            for supp_ind, (global_supp_ind, supp_img, supp_mask) in enumerate(support):
                supp_img[supp_mask] = 0.5 * supp_img[supp_mask] + 0.5 * np.array([255, 0, 255])
                supp_img = supp_img.astype(np.uint8)
                plt.imsave(f'tmp/{str(supp_ind).zfill(2)}.png', supp_img)
            # now plot the query image with the prediction in red and the ground truth in blue
            query_img[query_mask] = 0.5 * query_img[query_mask] + 0.5 * np.array([0, 0, 255])
            query_img[pred[0]] = 0.5 * query_img[pred[0]] + 0.5 * np.array([255, 0, 0])
            query_img = query_img.astype(np.uint8)
            plt.imsave(f'tmp/query.png', query_img)
        

            print(f'mIoU {n_shots}-shot {pascal_class}:',
                  aggregate_metrics(all_metrics)['avg_jacc'])
            print(f'cls oracle mIoU {n_shots}-shot {pascal_class}:',
                  aggregate_metrics(cls_oracle_all_metrics)['avg_jacc'])
            print(f'seg oracle mIoU {n_shots}-shot {pascal_class}:',
                    aggregate_metrics(seg_oracle_all_metrics)['avg_jacc'])
            print(f'seg cls oracle mIoU {n_shots}-shot {pascal_class}:',    
                    aggregate_metrics(seg_cls_oracle_all_metrics)['avg_jacc'])


if __name__ == '__main__':
    run_fss_experiment()