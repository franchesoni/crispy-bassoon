import os
from config import datasets_path
os.environ['DETECTRON2_DATASETS'] = str(datasets_path)

import ast
import shutil
from pathlib import Path
import tqdm
import numpy as np
import time
import matplotlib.pyplot as plt
import cv2

from mess.datasets.TorchvisionDataset import TorchvisionDataset, get_detectron2_datasets
import numpy as np
from metrics import compute_global_metrics, compute_tps_fps_tns_fns


def get_largest_incorrect_region(pred, gt):
    largest_incorrects = []
    for cls in [0, 1]:
        incorrect = (pred == cls) * (gt != cls)  # false neg / pos
        ret, labels_con = cv2.connectedComponents(incorrect.astype(np.uint8) * 255)
        label_unique, counts = np.unique(
            labels_con[labels_con != 0], return_counts=True
        )
        if len(counts) > 0:
            largest_incorrect = labels_con == label_unique[np.argmax(counts)]
            largest_incorrects.append(largest_incorrect)
        else:
            largest_incorrects.append(np.zeros_like(incorrect))

    largest_incorrect_cat = np.argmax([np.count_nonzero(x) for x in largest_incorrects])
    largest_incorrect = largest_incorrects[largest_incorrect_cat]
    return largest_incorrect, 1 - largest_incorrect_cat


def dt(a):
    return cv2.distanceTransform((a * 255).astype(np.uint8), cv2.DIST_L2, 0)


def click_position(largest_incorrect):
    h, w = largest_incorrect.shape

    largest_incorrect_w_boundary = np.zeros((h + 2, w + 2))
    largest_incorrect_w_boundary[1:-1, 1:-1] = largest_incorrect

    uys, uxs = np.where(largest_incorrect_w_boundary > 0)

    if uys.shape[0] == 0:
        return -1, -1

    outside_region_mask = 1 - largest_incorrect_w_boundary
    dist = dt(1 - outside_region_mask)
    dist = dist[1:-1, 1:-1]
    row, col = np.unravel_index(dist.argmax(), dist.shape)
    return row, col


def robot_click_single(pred, gt):
    incorrect_region, click_cat = get_largest_incorrect_region(pred, gt)
    row, col = click_position(incorrect_region)
    return (row, col, click_cat)





def to_numpy(x):
    return np.array(x)

def handle_dstdir(dstdir, reset, resume):
    dstdir = Path(dstdir)
    assert not (resume and reset), "you can't both resume and reset"
    if not (resume or reset):
        if dstdir.exists():
            raise FileExistsError('run already exists, you should resume or reset')
        else:
            print('creating brand new run')
            dstdir.mkdir(parents=True)
    elif reset:
        if dstdir.exists():
            shutil.rmtree(dstdir)
            print('removed last run')
        else:
            print('creating brand new run')
        dstdir.mkdir(parents=True)
    elif resume:
        if dstdir.exists():
            print('resuming last run')
        else:
            print('creating brand new run')
            dstdir.mkdir(parents=True)
    return dstdir
     
        
def main(dstdir, max_clicks_per_image=10, reset=False, resume=False, ds=None, dev=False, plot=False):
    ds_names = get_detectron2_datasets()
    TEST_DATASETS=['atlantis_sem_seg_test', 'chase_db1_sem_seg_test', 'corrosion_cs_sem_seg_test', 'cryonuseg_sem_seg_test', 'cub_200_sem_seg_test', 'cwfid_sem_seg_test', 'dark_zurich_sem_seg_val', 'deepcrack_sem_seg_test', 'dram_sem_seg_test', 'foodseg103_sem_seg_test', 'isaid_sem_seg_val', 'kvasir_instrument_sem_seg_test', 'mhp_v1_sem_seg_test', 'paxray_sem_seg_test_bones', 'paxray_sem_seg_test_diaphragm', 'paxray_sem_seg_test_lungs', 'paxray_sem_seg_test_mediastinum', 'pst900_sem_seg_test', 'suim_sem_seg_test', 'worldfloods_sem_seg_test_irrg', 'zerowaste_sem_seg_test', 'ndd20_sem_seg_test', 'mypascalvoc_sem_seg_test', 'mysbd_sem_seg_test', 'mygrabcut_sem_seg_test']
    assert (ds is None) or ds in TEST_DATASETS 
    TEST_DATASETS = TEST_DATASETS if ds is None else [ds]
    ds_names = sorted([ds_name for ds_name in ds_names if ds_name in TEST_DATASETS])

    modelname = "simpleclick"
    print("USING ", modelname.upper(), " MODEL")
    controller = load_controller()
    assert controller is not None

    for ds_name in ds_names:

        try:
            dstdir = handle_dstdir(Path(dstdir) / ds_name, reset, resume)
        except FileExistsError:
            continue

        dstjson = dstdir / f'{ds_name}.json'
        if resume:
            if dstjson.exists():
                with open(dstjson, 'r') as f:
                    metrics_for_dataset = ast.literal_eval(f.read())
            else:
                print('brand new run')
                metrics_for_dataset = {}
        else:
            metrics_for_dataset = {}

        print(f'running {ds_name}')
        ds = TorchvisionDataset(ds_name, transform=to_numpy, mask_transform=to_numpy)
        print('dataset length:', len(ds), 'already annotated:', len(metrics_for_dataset))
        class_indices, class_names = np.arange(len(ds.class_names)), ds.class_names
        n_digits = len(str(len(ds)))
        values_to_ignore = [255] + [ind for ind, cls_name in zip(class_indices, class_names) if cls_name in ['background', 'others', 'unlabeled', 'background (waterbody)', 'background or trash']]
        st, n_imgs = time.time(), 0
        for sample_ind, sample in tqdm.tqdm(enumerate(ds)):

            if resume and sample_ind in metrics_for_dataset:
                continue  # skip those that we already computed
            if dev:
                if sample_ind > 4:
                    break
            metrics_for_dataset[sample_ind] = {}

            img = sample[0]
            mask = sample[1]
            gt_masks = [mask == value for value in np.unique(mask) if value not in values_to_ignore]
            subdstdir = dstdir / f'img_{str(sample_ind).zfill(n_digits)}'
            subdstdir.mkdir()


            for mvalue_ind, value in enumerate(np.unique(mask)):
                if value in values_to_ignore:
                    continue
                gt_masks = [(mask == value)]
                class_name = ds.class_names[value].replace(' ','-').replace('(','').replace(')','').replace('/','-').replace('_', '-')
                if gt_masks[0].sum() == 0:
                    metrics_for_dataset[sample_ind][class_name] = None
                    continue
                subdstdirmask = subdstdir / f"class_{class_name}"
                subdstdirmask.mkdir()

                pred_mask = np.zeros_like(mask)
                metrics = [compute_global_metrics(*compute_tps_fps_tns_fns([pred_mask], gt_masks))]

                if plot:
                    err_mask = np.clip(
                            pred_mask[..., None] * [0, 1.0, 1.0] + gt_masks[0][..., None] * [1.0, 0, 0], 0, 1
                        )
                    plt.imsave(
                        f"{subdstdirmask}/click{str(0).zfill(2)}_mask_diff.png",
                        err_mask,
                    )
                controller.set_image(img)

                clicks = []
                for click_number in range(max_clicks_per_image):
                    imgs_to_show = []
                    names = []
                    # simulate clicks
                    click = robot_click_single(pred_mask, gt_masks[0])
                    clicks.append(click)

                    controller.add_click(click[1], click[0], click[2])
                    pred_mask = np.array(0 < controller.result_mask)

                    imgs_to_show += [pred_mask]
                    names += ["user_pred_masks"]

                    imgs_to_show += [
                            np.clip(
                                pred_mask[..., None] * [0, 1.0, 1.0] + gt_masks[0][..., None] * [1.0, 0, 0], 0, 1
                            )
                    ]
                    names += ["mask_diff"]

                    if plot:
                        # visualize everything
                        for i, im in enumerate(imgs_to_show):
                            plt.imsave(
                                f"{subdstdirmask}/click{str(click_number+1).zfill(2)}_{names[i]}.png",
                                im,
                            )
                            plt.close()

                        plt.figure()
                        plt.imshow(img)
                        if len(clicks) > 0:
                            plt.scatter(
                                [click[1] for click in clicks],
                                [click[0] for click in clicks],
                                c=[
                                    ["r", "b"][click[2]]
                                    for click in clicks
                                ],
                            )
                        plt.savefig(
                            f"{subdstdirmask}/click{str(click_number+1).zfill(2)}.png"
                        )
                        plt.close()

                    # compute metrics
                    metdict = compute_global_metrics(*compute_tps_fps_tns_fns([pred_mask], gt_masks))
                    metrics.append(metdict)
                    with open(f"{subdstdirmask}/metrics.json", "w") as f:
                        f.write(str(metrics).replace("'", '"'))
                metrics_for_dataset[sample_ind][class_name] = metrics
                with open(f"{subdstdirmask}/clicks.json", "w") as f:
                    f.write(str([list(c) for c in clicks]).replace("'", '"'))

            with open(dstdir / f'{ds_name}.json', 'w') as f:
                f.write(str(metrics_for_dataset).replace("'", '"'))

            n_imgs += 1
                
        print(f'processed {n_imgs/len(ds)*100}% of {len(ds)} in {time.time() - st} seconds')

    print('great!')

if __name__ == '__main__':
    import sys
    simpleclick_path = str(Path(__file__).parent / 'SimpleClick')
    print(f'adding {simpleclick_path} to path')
    sys.path.append(simpleclick_path)
    from clean_inference import load_controller

    from fire import Fire
    Fire(main)

