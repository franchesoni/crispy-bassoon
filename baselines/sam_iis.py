from pathlib import Path
import os
import shutil
from typing import Callable
from ast import literal_eval

import numpy as np
import matplotlib.pyplot as plt
import cv2


from segment_anything import sam_model_registry, SamPredictor
import numpy as np
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


# given an image we should extract the SAM features.
def get_embeddings_sam(sam_predictor, images: list[np.ndarray]) -> list[dict]:
    embeddings = []
    for img in images:
        # print(f'getting embedding for frame {frame_ind}')
        sam_predictor.set_image(img)
        embedding = {}
        embedding["original_size"] = sam_predictor.original_size
        embedding["input_size"] = sam_predictor.input_size
        embedding["features"] = sam_predictor.get_image_embedding()  # torch.Tensor
        embeddings.append(embedding)
    return embeddings


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
    sam_predictor: SamPredictor,
    embeddings: list[dict],
    clicks: list[Click],
    mask_inputs: list[None] | list[np.ndarray],
) -> tuple[list[np.ndarray], list[np.ndarray] | list[None]]:
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


def run_experiment(load_sample_fn, n_images, dev, max_clicks_per_image, runname, device):
    assert n_images > 0 and n_images <= 50, "n_images must be between 1 and 50"
    N = n_images
    MAX_CLICKS = max_clicks_per_image

    noplt = False
    dstdir = f"runs/{runname}"
    dstdir = Path(dstdir)
    try:
        dstdir.mkdir(parents=True)
    except FileExistsError:
        shutil.rmtree(dstdir)
        dstdir.mkdir()
        print("removed last run")

    if dev:
        sam_checkpoint = "sam_vit_b_01ec64.pth"
        model_type = "vit_b"
    else:
        sam_checkpoint = "sam_vit_l_0b3195.pth"
        model_type = "vit_l"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    sam_predictor = SamPredictor(sam)

    for ind in range(N):
        print(f"processing {ind+1}/{N}", end="\r")
        if ind == 8:
            noplt = True
        img, mask = load_sample_fn(ind)
        imgs, gt_masks = [img], [mask]
        subdstdir = dstdir / f"{str(ind).zfill(3)}"
        Path(subdstdir).mkdir()
        embeddings = get_embeddings_sam(sam_predictor, imgs)
        annotate_stack(
            images=imgs,
            gt_masks=gt_masks,
            iis_predictor=sam_predictor,
            embeddings=embeddings,
            robot_clicker=robot_click_multiple,
            propagate_fn=propagate_dummy,
            propagator_state={},
            dstdir=subdstdir,
            max_clicks=MAX_CLICKS,
            noplt=noplt,
        )

def run_main_plot_experiment(load_img_with_masks_as_batch_fn, n_images, dev, max_clicks_per_image, runname, device):
    dstdir = Path(runname)
    n_digits = len(str(n_images))
    try:
        dstdir.mkdir(parents=True)
    except FileExistsError:
        shutil.rmtree(dstdir)
        dstdir.mkdir()
        print("removed last run")

    if dev:
        sam_checkpoint = "sam_vit_b_01ec64.pth"
        model_type = "vit_b"
    else:
        sam_checkpoint = "sam_vit_l_0b3195.pth"
        model_type = "vit_l"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    sam_predictor = SamPredictor(sam)

    for ind in range(n_images):
        print(f"processing {ind+1}/{n_images}", end="\r")
        _, img, gt_masks = load_img_with_masks_as_batch_fn(ind)
        subdstdir = dstdir / f"img{str(ind).zfill(n_digits)}"
        Path(subdstdir).mkdir()
        embeddings = get_embeddings_sam(sam_predictor, [img])
        for maskind, gt_mask in enumerate(gt_masks):
            if gt_mask is None:
                continue
            subdstdirmask = subdstdir / f"class{str(maskind).zfill(2)}"
            subdstdirmask.mkdir()
            annotate_stack(
                images=[img],
                gt_masks=[gt_mask],
                iis_predictor=sam_predictor,
                embeddings=embeddings,
                robot_clicker=robot_click_multiple,
                propagate_fn=propagate_dummy,
                propagator_state={},
                dstdir=subdstdirmask,
                max_clicks=max_clicks_per_image,
                noplt=False,
            )


def get_curves_for_main_plot(runname, max_total_clicks, stack_size, clicks_per_stack, class_number):
    runname = Path(runname)
    assert Path(runname).exists()
    # for each class, we need to compute the curve
    # we need to aggrupate in stacks
    # make some clicks per stack
    # and continue until we get to max_total_clicks
    imgs_dirs = sorted(os.listdir(runname))
    sweep = 0
    n_clicks_so_far = 0
    while n_clicks_so_far < max_total_clicks:  # sweep
        stack_starts_at = 0
        while stack_starts_at < len(imgs_dirs) - stack_size:
            offset = 0
            stack = []
            # go through the images finding the subdirs call `classXX` that correspond to the class
            while len(stack) < stack_size:
                imgdir = imgs_dirs[stack_starts_at + offset]
                if (runname / imgdir / f"class{str(class_number).zfill(2)}").exists():
                    stack.append(imgdir)
                offset += 1
            print('created stack!')
            # get metrics in stack
            metrics_in_stack = []
            for imgdir in stack:
                with open(runname / imgdir / f"class{str(class_number).zfill(2)}" / 'metrics.json', 'r') as f:
                    metrics = literal_eval(f.read())
                    metrics_in_stack.append(metrics)
            print('got metrics in stack!')
            # now get the curve for the stack
            best_curve_metrics = get_best_iis_curve(metrics_in_stack, max_clicks=clicks_per_stack)
            print('got curve for stack!')

            stack_starts_at += stack_size
            print('well done!')
        sweep += 1

from metrics import aggregate_metrics
def get_best_iis_curve(metrics_in_stack, total_clicks=None):
    '''metrics_in_stack is a list of lists of metrics corresponding to the frame ind and the click ind respectively'''
    n_frames = len(metrics_in_stack)  # e.g. 16
    max_clicks = n_frames * len(metrics_in_stack[0])-1 if total_clicks is None else total_clicks  # e.g. 4

    clicks_per_image = [0 for _ in range(n_frames)]  # e.g. [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    metrics = [aggregate_metrics(metrics_in_stack, clicks_per_image)]
    for click_n in range(n_frames * max_clicks):
        print(f'finding click {click_n+1} of {n_frames*max_clicks}', end='\r')
        best_global_iou = -np.inf  #metrics[-1]['jacc']
        for ind in range(n_frames):  # try clicks on all the images 
            if clicks_per_image[ind] == max_clicks:
                continue  # already maxed out clicks on this image
            # else try to click on this image
            hypothetical_clicks_per_image = clicks_per_image.copy()
            hypothetical_clicks_per_image[ind] += 1
            hypothetical_metrics = aggregate_metrics(metrics_in_stack, hypothetical_clicks_per_image)
            if hypothetical_metrics['jacc'] > best_global_iou:
                best_global_iou = hypothetical_metrics['jacc']
                best_ind = ind
        if best_global_iou == -np.inf:
            break  # no click can be placed, exit
        clicks_per_image[best_ind] += 1
        metrics.append(aggregate_metrics(metrics_in_stack, clicks_per_image))
    return metrics




def get_curves_for_main_plot_oneimageatatime(runname, ds_length, max_total_clicks, class_number, seed, load_ind_img_mask_fn):
    """Simply load the results in the order given by the loading function and repeat if needed."""
    runname = Path(runname)
    assert Path(runname).exists()
    class_index = class_number - 1  # rm background as class
    # get all curves (one per image)
    all_metrics = []  
    for click_n in range(ds_length):
        current_ind, _, _ = load_ind_img_mask_fn(click_n)
        with open(runname / f"img{str(current_ind).zfill(4)}" / f"class{str(class_index).zfill(2)}" / 'metrics.json', 'r') as f:
            metrics = literal_eval(f.read())
        all_metrics.append(metrics)

    # compute initial metrics
    curve = [aggregate_metrics(all_metrics, [0 for _ in range(ds_length)])]  
    # make clicks
    for click_n in range(max_total_clicks):
        sweep = click_n // ds_length
        ds_ind = click_n % ds_length
        new_agg = aggregate_metrics(all_metrics, [sweep+1 for _ in range(ds_ind+1)] + [sweep for _ in range(ds_length-ds_ind-1)])
        curve.append(new_agg)

    # now save curve to runname / curve.json
    (runname / f'curves_seed{seed}').mkdir(exist_ok=True)
    with open(runname / f'curves_seed{seed}' / f"curve_class{str(class_index).zfill(2)}.json", 'w') as f:
        f.write(str(curve).replace("'", '"'))


        
