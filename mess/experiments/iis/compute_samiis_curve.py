import time
from pathlib import Path
import os
import shutil
from typing import Callable
from ast import literal_eval

import numpy as np
import matplotlib.pyplot as plt
import cv2


from segment_anything import SamPredictor
import numpy as np
from IISS.extract_masks import sam_predictor
from metrics import compute_global_metrics, compute_tps_fps_tns_fns


# we need to simulate a real click. It should be in the center of the largest falsely classified region
# define the click type which is (frame_ind, row, col, category)
class Click:
    def __init__(self, frame_ind: int, row: int, col: int, category: int):
        self.row = row
        self.col = col
        self.label = category
        self.pos = (self.row, self.col)
        self.frame_ind = frame_ind
        self.tuple = (self.frame_ind, self.row, self.col, self.label)

    def __str__(self):
        return str(self.tuple)


"""#### Extract features"""


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(
        pos_points[:, 0],
        pos_points[:, 1],
        color="green",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )
    ax.scatter(
        neg_points[:, 0],
        neg_points[:, 1],
        color="red",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )



def to_input_point_label(clicks: list[Click], N: int):
    input_points, input_labels = [], []
    for frame_ind in range(N):
        input_point = np.array(
            [(click.col, click.row) for click in clicks if click.frame_ind == frame_ind]
        )
        input_label = np.array(
            [click.label for click in clicks if click.frame_ind == frame_ind]
        )
        input_points.append(input_point)
        input_labels.append(input_label)
    return input_points, input_labels


def apply_f(
    sam_predictor,
    embeddings,
    clicks,
    mask_inputs,
):
    out_masks, out_logits = [], []
    N = len(embeddings)
    input_points, input_labels = to_input_point_label(clicks, N)
    for frame_ind, embedding in enumerate(embeddings):
        sam_predictor.original_size = embedding["original_size"]
        if len(input_points[frame_ind]) > 0:
            sam_predictor.input_size = embedding["input_size"]
            sam_predictor.features = embedding["features"]
            sam_predictor.is_image_set = True
            if mask_inputs[frame_ind] is None:
                mask_input = None
            else:
                mask_input = mask_inputs[frame_ind][None]

            masks, qualities, logits = sam_predictor.predict(
                point_coords=input_points[frame_ind],
                point_labels=input_labels[frame_ind],
                mask_input=mask_input,
                multimask_output=False,
            )  # Use mask input
            mask, logit = masks[np.argmax(qualities)], logits[np.argmax(qualities)]
            out_masks.append(mask)
            out_logits.append(logit)
        else:
            out_masks.append(np.zeros(sam_predictor.original_size))
            out_logits.append(None)
    return out_masks, out_logits


def predict_and_plot(
    sam_predictor, input_points_at_frame, input_labels_at_frame, mask_input, outname
):
    masks, qualities, logits = sam_predictor.predict(
        point_coords=input_points_at_frame,
        point_labels=input_labels_at_frame,
        mask_input=mask_input,
        multimask_output=False,
    )  # mask_input
    mask, logit = masks[np.argmax(qualities)], logits[np.argmax(qualities)]
    # debug
    plt.figure()
    show_mask(mask, plt.gca())
    show_points(input_points_at_frame, input_labels_at_frame, plt.gca())
    plt.savefig(f"{outname}.png")
    plt.close()


def propagate_dummy(clicks, pred_masks, propagator_state=None):
    return [], [], propagator_state


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

def robot_center_of_largest_single(pred, gt_mask):
    incorrect_region, click_cat = get_largest_incorrect_region(pred, gt_mask)
    row, col = click_position(incorrect_region)
    return Click(0, row, col, click_cat)

def robot_click_multiple(preds, masks):
    max_size_incorrect = -1
    for ind, (pred, gt_mask) in enumerate(zip(preds, masks)):
        incorrect_region, click_cat = get_largest_incorrect_region(pred, gt_mask)
        size_incorrect = (incorrect_region == 1).sum()
        if max_size_incorrect < size_incorrect:
            max_size_incorrect = size_incorrect
            row, col = click_position(incorrect_region)
            ret_click_cat = click_cat
            img_ind = ind
    return Click(img_ind, row, col, ret_click_cat)


def annotate_stack(
    images: list[np.ndarray],
    gt_masks: list[np.ndarray],
    iis_predictor: SamPredictor,
    embeddings: list[dict],
    robot_clicker: Callable,
    propagate_fn: Callable,
    propagator_state: dict,
    dstdir: str,
    max_clicks: int = 20,
    noplt=True,
):
    """This function is used to evaluate the performance of an annotation pipeline over a stack of images. We use the SAM API for the iis_predictor, allow for different robot clickers and for a propagator. The annotation loop is run and the metrics are computed. The intermediate masks and metrics are saved in dstdir."""

    pred_logits_masks = [None for _ in range(len(images))]
    pred_masks = [np.zeros_like(mask) for mask in gt_masks]  # init at 0
    clicks, metrics = [], [
        compute_global_metrics(*compute_tps_fps_tns_fns(pred_masks, gt_masks))
    ]

    if not noplt:
        err_masks = [
            np.clip(
                pm[..., None] * [0, 1.0, 1.0] + gt[..., None] * [1.0, 0, 0], 0, 1
            )
            for pm, gt in zip(pred_masks, gt_masks)
        ]
        for j, im in enumerate(err_masks):
            plt.imsave(
                f"{dstdir}/click{str(0).zfill(2)}_mask_diff_{str(j).zfill(2)}.png",
                im,
            )
 


    for click_number in range(max_clicks):
        imgs_to_show = []
        names = []
        # simulate clicks
        click = robot_clicker(pred_masks, gt_masks)
        clicks.append(click)

        pred_masks, pred_logits_masks = apply_f(
            iis_predictor, embeddings, clicks, mask_inputs=pred_logits_masks
        )  # predict once, this is for h_S
        imgs_to_show += [pred_masks, pred_logits_masks]
        names += ["user_pred_masks", "user_pred_logits_mask"]

        synthetic_clicks_F, synthetic_clicks_S, propagator_state = propagate_fn(
            clicks, pred_masks, propagator_state
        )  # propagate clicks in feature and space

        if len(synthetic_clicks_F) + len(synthetic_clicks_S) > 0:
            pred_masks, pred_logits_masks = apply_f(
                iis_predictor,
                embeddings,
                clicks + synthetic_clicks_F + synthetic_clicks_S,
                mask_inputs=pred_logits_masks,
            )
            imgs_to_show += [pred_masks, pred_logits_masks]
            names += ["pred_masks", "pred_logits_masks"]

        imgs_to_show += [
            [
                np.clip(
                    pm[..., None] * [0, 1.0, 1.0] + gt[..., None] * [1.0, 0, 0], 0, 1
                )
                for pm, gt in zip(pred_masks, gt_masks)
            ]
        ]
        names += ["mask_diff"]

        if not noplt:
            clicks_to_show = [clicks, synthetic_clicks_F, synthetic_clicks_S]

            # visualize everything
            for i, img in enumerate(imgs_to_show):
                for j, im in enumerate(img):
                    if im is None:
                        continue
                    plt.imsave(
                        f"{dstdir}/click{str(click_number+1).zfill(2)}_{names[i]}_{str(j).zfill(2)}.png",
                        im,
                    )
                    plt.close()

            for i, img in enumerate(images):
                plt.figure()
                plt.imshow(img)
                for typeind, clicks_of_type in enumerate(clicks_to_show):
                    clicks_of_type_and_frame = [
                        click for click in clicks_of_type if click.frame_ind == i
                    ]
                    if len(clicks_of_type_and_frame) > 0:
                        plt.scatter(
                            [click.pos[1] for click in clicks_of_type_and_frame],
                            [click.pos[0] for click in clicks_of_type_and_frame],
                            c=[
                                ["r", "b"][click.label]
                                for click in clicks_of_type_and_frame
                            ],
                            marker=["x", "o", "s"][typeind],
                            s=[100, 50, 40][typeind],
                        )
                        # print([click.label for click in clicks_of_type_and_frame])
                plt.savefig(
                    f"{dstdir}/click{str(click_number+1).zfill(2)}_clicks_{str(i).zfill(2)}.png"
                )
                plt.close()

        # compute metrics
        metdict = compute_global_metrics(*compute_tps_fps_tns_fns(pred_masks, gt_masks))
        metrics.append(metdict)
        with open(f"{dstdir}/metrics.json", "w") as f:
            f.write(str(metrics).replace("'", '"'))
    with open(f"{dstdir}/clicks.json", "w") as f:
        f.write(str([list(c.tuple) for c in clicks]).replace("'", '"'))
    return clicks, metrics



import os
from config import datasets_path
os.environ['DETECTRON2_DATASETS'] = str(datasets_path)
import ast
from PIL import Image
import shutil
from pathlib import Path
import tqdm
import numpy as np

from IISS.create_segmentation import create_segmentation
from IISS.large_experiment import get_clicked_segment
from metrics import compute_global_metrics, compute_tps_fps_tns_fns

from mess.datasets.TorchvisionDataset import TorchvisionDataset, get_detectron2_datasets

def to_numpy(x):
    return np.array(x)

def handle_dstdir(dstdir, reset, resume):
    dstdir = Path(dstdir)
    assert not (resume and reset), "you can't both resume and reset"
    if not (resume or reset):
        if dstdir.exists():
            raise FileExistsError('run already exists, you should resume or reset')
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
     
        
def main(precomputed_dir, dstdir, max_clicks_per_image=10, reset=False, resume=False, ds=None, dev=False, plot=False):
    """`precomputed_dir` is a folder where the precomputed variables are stored. The variables are stored at `precomputed_dir / ds_name / sam_embeddings / sam_embedding_000.npy` where `000` is the image index on the dataset."""
    precomputed_dir = Path(precomputed_dir)
    assert precomputed_dir.exists()

    ds_names = get_detectron2_datasets()
    TEST_DATASETS=['atlantis_sem_seg_test', 'chase_db1_sem_seg_test', 'corrosion_cs_sem_seg_test', 'cryonuseg_sem_seg_test', 'cub_200_sem_seg_test', 'cwfid_sem_seg_test', 'dark_zurich_sem_seg_val', 'deepcrack_sem_seg_test', 'dram_sem_seg_test', 'foodseg103_sem_seg_test', 'isaid_sem_seg_val', 'kvasir_instrument_sem_seg_test', 'mhp_v1_sem_seg_test', 'paxray_sem_seg_test_bones', 'paxray_sem_seg_test_diaphragm', 'paxray_sem_seg_test_lungs', 'paxray_sem_seg_test_mediastinum', 'pst900_sem_seg_test', 'suim_sem_seg_test', 'worldfloods_sem_seg_test_irrg', 'zerowaste_sem_seg_test', 'ndd20_sem_seg_test', 'mypascalvoc_sem_seg_test', 'mysbd_sem_seg_test', 'mygrabcut_sem_seg_test']
    assert (ds is None) or ds in TEST_DATASETS 
    TEST_DATASETS = TEST_DATASETS if ds is None else [ds]
    ds_names = sorted([ds_name for ds_name in ds_names if ds_name in TEST_DATASETS])

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
            embedding = np.load(precomputed_dir / ds_name / 'sam_embeddings' / f'sam_embedding_{str(sample_ind).zfill(n_digits)}.npy', allow_pickle=True).item()
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

                clicks, metrics = annotate_stack(
                    images=[img],
                    gt_masks=gt_masks,
                    iis_predictor=sam_predictor,
                    embeddings=[embedding],
                    robot_clicker=robot_click_multiple,
                    propagate_fn=propagate_dummy,
                    propagator_state={},
                    dstdir=subdstdirmask,
                    max_clicks=max_clicks_per_image,
                    noplt=not plot,
                )
                metrics_for_dataset[sample_ind][class_name] = metrics
            print(f'processed {sample_ind+1}/{len(ds)}', end='\r')
            with open(dstdir / f'{ds_name}.json', 'w') as f:
                f.write(str(metrics_for_dataset).replace("'", '"'))
            n_imgs += 1
                
        print(f'processed {n_imgs/len(ds)*100}% of {len(ds)} in {time.time() - st} seconds')

    print('great!')

if __name__ == '__main__':
    from fire import Fire
    Fire(main)

