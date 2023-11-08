"""
Main loop to run the experiment of how SEG-GPT performance changes with the number of shots.
This code runs only for one seed. In order to run for multiple seeds, launch (if possible parallel) multiple instances of this script with different seeds.
"""
import argparse
import os
import sys
import tempfile
import typing as T

# TODO: Replace this with the correct path to the datasets
os.environ["DETECTRON2_DATASETS"] = "/mnt/adisk/franchesoni/messdata"


import numpy as np
import torch
import tqdm
from PIL import Image
from skimage.morphology import binary_dilation, binary_erosion

# TODO: Replace this with the correct path to the SegGPT repository
sys.path.append("/home/emasquil/workspace/Painter/SegGPT/SegGPT_inference")
import models_seggpt
from seggpt_engine import inference_image

sys.path.append("..")
from mess.datasets.TorchvisionDataset import TorchvisionDataset
from seg_gpt.metrics import compute_global_metrics, compute_tps_fps_tns_fns

# TODO: Adjust the maximum number of shots depending on GPU size. For a quick try, use as NUMBER_OF_SHOTS = [MAXIMUM_NUMBER] and check if there's no CUDA error
NUMBER_OF_SHOTS = [i for i in range(1, 6)]
CLASSES_TO_IGNORE = [
    "background",
    "others",
    "unlabeled",
    "background (waterbody)",
    "background or trash",
]


def get_class_img_mapping(dataset: TorchvisionDataset, values_to_ignore: T.List):
    """
    Returns a mapping that stores which images are available for each class.
    """
    class_img_mapping = {}
    for ix, (_, mask) in enumerate(dataset):
        for value in np.unique(mask):
            if value in values_to_ignore:
                continue
            if value not in class_img_mapping:
                class_img_mapping[value] = [ix]
            class_img_mapping[value].append(ix)
    return class_img_mapping


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dataset", default="foodseg103_sem_seg_train")
    parser.add_argument("--test-dataset", default="foodseg103_sem_seg_test")
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    # Create results dir
    try:
        os.makedirs(args.results_dir)
    except FileExistsError:
        raise FileExistsError(f"Results directory {args.results_dir} already exists.")

    # Prepare model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = getattr(models_seggpt, "seggpt_vit_large_patch16_input896x448")()
    model.seg_type = "instance"
    checkpoint = torch.load(
        "/home/emasquil/workspace/Painter/SegGPT/SegGPT_inference/seggpt_vit_large.pth"
    )
    _ = model.load_state_dict(checkpoint["model"], strict=False)
    model.eval()
    model.to(device)
    print("Model loaded")

    # Load datasets
    train_ds = TorchvisionDataset(
        args.train_dataset,
        transform=lambda x: np.array(x),
        mask_transform=lambda x: np.array(x),
    )
    test_ds = TorchvisionDataset(
        args.test_dataset,
        transform=lambda x: np.array(x),
        mask_transform=lambda x: np.array(x),
    )
    print("Datasets loaded")

    # Get mapping that stores which images are available for each class
    class_indices, class_names = (
        np.arange(len(train_ds.class_names)),
        train_ds.class_names,
    )
    values_to_ignore = [255] + [
        ind
        for ind, cls_name in zip(class_indices, class_names)
        if cls_name
        in [
            "background",
            "others",
            "unlabeled",
            "background (waterbody)",
            "background or trash",
        ]
    ]
    class_img_mapping = get_class_img_mapping(train_ds, values_to_ignore)
    test_class_img_mapping = get_class_img_mapping(test_ds, values_to_ignore)

    # Set random seed
    np.random.seed(args.seed)
    # Iterate over all shots
    metrics_per_shot = {}
    for shot in tqdm.tqdm(NUMBER_OF_SHOTS):
        print(f"Shot {shot}")

        # Iterate over all classes
        metrics_per_class = {}
        for class_ind in tqdm.tqdm(list(class_img_mapping.keys())):
            print(f"Class {class_ind} ({class_names[class_ind]})")

            # Skip if not enough images
            if len(class_img_mapping[class_ind]) < shot:
                print(
                    "Not enough images to sample from. Skipping this shot for this class."
                )
                continue

            # Iterate over all images in the test dataset
            metrics_per_image = []
            for img_ind in tqdm.tqdm(test_class_img_mapping[class_ind]):
                with tempfile.TemporaryDirectory() as tmpdir:
                    # Get test image and mask
                    test_img, test_masks = test_ds[img_ind]
                    test_mask = test_masks == class_ind
                    Image.fromarray(test_img).save(f"{tmpdir}/query.png")

                    # Get support images and masks
                    support_img_inds = np.random.choice(
                        class_img_mapping[class_ind], size=shot, replace=False
                    )
                    support_img_list, support_mask_list = [], []
                    for n, ind in enumerate(support_img_inds):
                        support_img, support_masks = train_ds[ind]
                        support_mask = support_masks == class_ind
                        Image.fromarray(support_img).save(f"{tmpdir}/supp_img_{n}.png")
                        Image.fromarray(support_mask).save(
                            f"{tmpdir}/supp_mask_{n}.png"
                        )
                        support_img_list.append(f"{tmpdir}/supp_img_{n}.png")
                        support_mask_list.append(f"{tmpdir}/supp_mask_{n}.png")

                    # Perform inference
                    inference_image(
                        model,
                        device,
                        f"{tmpdir}/query.png",
                        support_img_list,
                        support_mask_list,
                        f"{tmpdir}/pred.png",
                    )

                    # compute prediction (hack)
                    imgi = np.array(Image.open(f"{tmpdir}/query.png").convert("RGB"))
                    imgo = np.array(Image.open(f"{tmpdir}/pred.png").convert("RGB"))
                    pred = binary_erosion(
                        binary_dilation(
                            np.abs(imgo / 255 - imgi / 255 * 0.4).max(axis=2) > 0.01
                        )
                    )

                # Compute metrics
                metrics = compute_global_metrics(
                    *compute_tps_fps_tns_fns([pred], [test_mask])
                )
                metrics_per_image.append(metrics)

            # Store metrics
            metrics_per_class[class_ind] = metrics_per_image
        # Store metrics
        metrics_per_shot[shot] = metrics_per_class

    # Save metrics
    np.save(os.path.join(args.results_dir, "metrics.npy"), metrics_per_shot)