import argparse
import os
import sys
import tempfile
import typing as T

os.environ["DETECTRON2_DATASETS"] = "/mnt/adisk/franchesoni/messdata"


import detectron2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from skimage.morphology import binary_dilation, binary_erosion
import tqdm

sys.path.append("/home/emasquil/workspace/Painter/SegGPT/SegGPT_inference")
import models_seggpt
import skimage.io
from seggpt_engine import inference_image, inference_video

sys.path.append("/home/emasquil/workspace/crispy-bassoon")
from seg_gpt.metrics import compute_global_metrics, compute_tps_fps_tns_fns

from mess.datasets.TorchvisionDataset import TorchvisionDataset, get_detectron2_datasets

NUMBER_OF_SHOTS = [i for i in range(1, 8)]
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
    parser.add_argument("--use-existent-mappings", action="store_true")

    # Prepare model
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
        parser.parse_args().train_dataset,
        transform=lambda x: np.array(x),
        mask_transform=lambda x: np.array(x),
    )
    test_ds = TorchvisionDataset(
        parser.parse_args().test_dataset,
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
    if not parser.parse_args().use_existent_mappings:
        class_img_mapping = get_class_img_mapping(train_ds, values_to_ignore)
        test_class_img_mapping = get_class_img_mapping(test_ds, values_to_ignore)
        # Save mappings
        np.save("class_img_mapping.npy", class_img_mapping)
        np.save("test_class_img_mapping.npy", test_class_img_mapping)
    else:
        class_img_mapping = np.load("class_img_mapping.npy", allow_pickle=True).item()
        test_class_img_mapping = np.load(
            "test_class_img_mapping.npy", allow_pickle=True
        ).item()

    # Repeat the experiment for 3 seeds
    seeds = [0, 1, 2]
    for seed in seeds:
        np.random.seed(seed)
        # Iterate over all available classes
        metrics_per_class = {}
        for class_ind in tqdm.tqdm(list(class_img_mapping.keys())[:1]):
            print(f"Class {class_ind} ({class_names[class_ind]})")

            # Iterate over all available shots
            metrics_per_shot = {}
            for shot in tqdm.tqdm(NUMBER_OF_SHOTS):
                print(f"Shot {shot}")
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
                            Image.fromarray(support_img).save(
                                f"{tmpdir}/supp_img_{n}.png"
                            )
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
                        imgi = np.array(
                            Image.open(f"{tmpdir}/query.png").convert("RGB")
                        )
                        imgo = np.array(Image.open(f"{tmpdir}/pred.png").convert("RGB"))
                        pred = binary_erosion(
                            binary_dilation(
                                np.abs(imgo / 255 - imgi / 255 * 0.4).max(axis=2) > 0.01
                            )
                        )

                        # # Create 2x2 plot with query image, gt mask, pred.png and pred
                        # fig, axs = plt.subplots(2, 2)
                        # axs[0, 0].imshow(test_img)
                        # axs[0, 1].imshow(test_mask)
                        # axs[1, 0].imshow(imgo)
                        # axs[1, 1].imshow(pred)
                        # plt.savefig(f"plot{img_ind}.png")

                        # Compute metrics
                        metrics = compute_global_metrics(
                            *compute_tps_fps_tns_fns([pred], [test_mask])
                        )
                        metrics_per_image.append(metrics)

                # Store metrics
                metrics_per_shot[shot] = metrics_per_image

            # Store metrics
            metrics_per_class[class_names[class_ind]] = metrics_per_shot

        # Save metrics
        np.save(f"metrics_{str(seed)}.npy", metrics_per_class)
