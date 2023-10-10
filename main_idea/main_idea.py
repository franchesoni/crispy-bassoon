"""Generates the results over the four images for two clicks taken from SAM IIS for our method and the SAM IIS method."""

from pathlib import Path
import shutil
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from dataloaders.ndd20 import load_sample
from baselines.sam_iis import run_experiment as run_sam_iis_experiment
from baselines.sam_iis import click_position
from IISS.run_experiment import run_experiment as run_iiss_experiment
from config import device, dev
from metrics import get_global_click_metrics_from_iis

def main():
    datapath = Path('main_idea/images') 
    def load_sample_fn(index):
        return load_sample(datapath, index)

    # run the two experiments. What we need is to take the best first two clicks for SAM IIS and for IISS. Then we need to show the clicks in the images (maybe as triangles / squares). Then we need to show the error maps obtained. For this we need 1. the first two clicks for SAM IIS, 2. the first two clicks for IISS, 3. the error maps for the first two clicks of SAM IIS, 4. the error maps for IISS.

    # run sam iis 
    run_sam_iis_experiment(load_sample_fn, 4, dev, 2, 'main_idea_sam_iis', device)
    metrics, click_seq = get_global_click_metrics_from_iis('runs/main_idea_sam_iis', 2)
    print(click_seq[1])
    # first clicks at images [1, 0, 0, 1]

    # run IISS second
    run_iiss_experiment(load_sample_fn, 4, max_total_clicks=2, runname='main_idea_iiss')
    # first clicks at [(3, 24, True), (1, 9, False)]

    # visualize images and clicks
    shutil.rmtree('runs/main_idea_vis')
    Path('runs/main_idea_vis').mkdir()
    # get the click positions
    gt_mask_1 = np.array(Image.open('runs/main_idea_iiss/sam_masks/gt_00.png').convert('L')) > 100
    click_iis_1 = click_position(gt_mask_1)
    gt_mask_4 = np.array(Image.open('runs/main_idea_iiss/sam_masks/gt_03.png').convert('L')) > 100
    click_iis_2 = click_position(gt_mask_4)

    selected_mask_1 = np.array(Image.open('runs/main_idea_iiss/sam_masks/mask_03_024.png').convert('L')) > 100
    click_iiss_1 = click_position(selected_mask_1)
    selected_mask_2 = np.array(Image.open('runs/main_idea_iiss/sam_masks/mask_01_009.png').convert('L')) > 100
    click_iiss_2 = click_position(selected_mask_2)
    
    # show the images and overlay the clicks if needed
    plt.figure()
    plt.imshow(Image.open('runs/main_idea_iiss/sam_masks/img_00.png'), interpolation='nearest')
    plt.plot(click_iis_1[1], click_iis_1[0], 'bo')
    plt.axis('off')
    plt.grid('off')
    plt.savefig('runs/main_idea_vis/img_00.png', bbox_inches='tight', pad_inches=0)

    plt.figure()
    plt.imshow(Image.open('runs/main_idea_iiss/sam_masks/img_01.png'), interpolation='nearest')
    plt.plot(click_iiss_2[1], click_iiss_2[0], 'rx', markersize=10)
    plt.axis('off')
    plt.grid('off')
    plt.savefig('runs/main_idea_vis/img_01.png', bbox_inches='tight', pad_inches=0)

    plt.figure()
    plt.imshow(Image.open('runs/main_idea_iiss/sam_masks/img_02.png'), interpolation='nearest')
    plt.axis('off')
    plt.grid('off')
    plt.savefig('runs/main_idea_vis/img_02.png', bbox_inches='tight', pad_inches=0)

    plt.figure()
    plt.imshow(Image.open('runs/main_idea_iiss/sam_masks/img_03.png'), interpolation='nearest')
    plt.plot(click_iiss_1[1], click_iiss_1[0], 'ro', markersize=10)
    plt.plot(click_iis_2[1], click_iis_2[0], 'bo')
    plt.axis('off')
    plt.grid('off')
    plt.savefig('runs/main_idea_vis/img_03.png', bbox_inches='tight', pad_inches=0)







if __name__ == '__main__':
    main()