device = 'cuda'
from pathlib import Path
datasets_path = Path('/home/franchesoni/adisk')

MIN_MASK_REGION_AREA = 100  # SAM's default, do not change
dev = False
# dev for SAM is 8x8, vit_b, and one layer

# DINO config
DINO_RESIZE = (644, 644)  # from the recommended segmentation resize in the paper but adapted to the patch size 
DINO_PATCH_SIZE = 14