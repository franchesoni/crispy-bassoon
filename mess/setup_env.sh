# run script with
# bash mess/setup_env.sh

# Create new environment "mess"
source ~/miniconda3/etc/profile.d/conda.sh
conda create --name mess -y python=3.8
conda activate mess

# Install PyTorch with CUDA 11.3
conda install -y pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
# Install Detectron2
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html

pip install gdown
pip install rasterio
pip install pandas
pip install Pillow==9.5 scikit-image opencv-python
pip install git+https://github.com/facebookresearch/segment-anything.git
