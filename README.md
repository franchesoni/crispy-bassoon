# crispy-bassoon

**setup**
you can follow the instructions in `mess/setup_env.sh`, for manual install do a mix of that and:
you might also need to update numpy from 1.21 to 1.25
```
pip install --upgrade pip
pip install git+https://github.com/facebookresearch/segment-anything.git
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth 
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
pip install opencv-python scikit-image
```

**guide to run the experiments in the supercomputer**
`module load pytorch-gpu/py3/1.10.1` (ii) download sam model. DINO is downloaded via torchvision hub. You have to run dino with internet connection at least once to download the weights 
`bash mess/env_setup.sh`

follow the instructions on mess repo to download the datasets. Some won't be downloaded. DRAM needs to be processed by hand with a local installation of rar.




# structure
- `baselines`
    - `sam_iis.py`: Implements SAM in the interactive segmentation (IIS) regime. For a group of images we annotate each one individually until $N$ clicks per image. The clicks are placed in the "center" of the largest wrongly classified component. `robot_click_multiple` will take the predictions and the ground truth masks, compute the falsely classified regions, take the largest one in the dataset and place a click there. 
- `dataloaders`: where the dataset handling is implemented
    - `ndd20.py`: `load_sample` from this dataset. Only take underwater images.
- `IISS`: where our method lives
    - see `run_experiment` to understand how we annotate an image stack. To compute the next click we use `get_clicked_segment` where the click is placed on the segment that reduces the error the most. How to implement this interaction in practice is another question answered on the paper (involving inclusions and centers).
- `main_idea`: the code to compute the figures for the "main idea" figure of the paper
- `config.py`: global configuration options
- `metrics.py`: classification metrics, acc and IoU both for individual images and stacks.


# Datasets
ADE20k following https://github.com/CSAILVision/placeschallenge/tree/master/instancesegmentation
PASCAL VOC (use torchvision datasets)
