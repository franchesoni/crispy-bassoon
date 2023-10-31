import os
from pathlib import Path

import gdown

"""This script downloads all the datasets that have a direct public download link and have training data. Train and test splits are preferred, if one is missing, val is used instead, if the two are missing, then we have no split and the dataset should be discarded. For chase, the train/test split is built at preparation. 
- CryoNuSeg doesn't have a split, therefore it's discarded.
- DRAM doesn't have annotations for training data, therefore it's discarded.
- BDD100k doesn't have a public download link, therefore it's discarded.
- Dark Zurich train images don't have labels, therefore it's discarded.
- floodnet doesn't have a public download link, therefore it's discarded.
- postdam doesn't have a public download link, therefore it's discarded.
- uavid doesn't have a public download link, therefore it's discarded.
We end up with 17 datasets.
"""


def download_atlantis(dataset_dir):
    ds_path = dataset_dir / 'atlantis'
    if ds_path.exists():
        print('Dataset already downloaded')
        return
    # Dataset page: https://github.com/smhassanerfani/atlantis.git
    print('Downloading dataset...')
    # Downloading github repo
    os.system(f'git clone https://github.com/smhassanerfani/atlantis.git atlantisGit')
    os.system('mv atlantisGit/atlantis ' + str(ds_path))
    os.system('rm -Rf atlantisGit')

def download_chase(dataset_dir):
    ds_path = dataset_dir / 'CHASEDB1'
    if ds_path.exists():
        print('Dataset already downloaded')
        return
    print('Downloading dataset...')
    # Downloading zip
    os.system('wget https://staffnet.kingston.ac.uk/~ku15565/CHASE_DB1/assets/CHASEDB1.zip')
    os.system('unzip CHASEDB1.zip -d ' + str(ds_path))
    os.system('rm CHASEDB1.zip')

    ######################################
    ######################################
def download_corrosion(dataset_dir):
    ds_path = dataset_dir / 'CorrosionConditionStateClassification'
    if ds_path.exists():
        print('Dataset already downloaded')
        return
    print('Downloading dataset...')
    # Downloading zip
    os.system('wget https://figshare.com/ndownloader/files/31729733')
    os.system('unzip 31729733 -d ' + str(dataset_dir))
    os.system('rm 31729733')

    ######################################
    ######################################

# deleted CryoNuSeg. CryoNuSeg doesn't have a training set 

    ######################################
    ######################################

def download_cub(dataset_dir):
    ds_path = dataset_dir / "CUB_200_2011"
    if ds_path.exists():
        print('Dataset already downloaded')
        return
    # Downloading data
    os.system("wget https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz")
    os.system(f"tar -xvzf CUB_200_2011.tgz -C {dataset_dir}")
    os.system(f"rm CUB_200_2011.tgz")

    # Downloading segmentation masks
    os.system("wget https://data.caltech.edu/records/w9d68-gec53/files/segmentations.tgz")
    os.system(f"tar -xvzf segmentations.tgz -C {dataset_dir / 'CUB_200_2011'}")
    os.system(f"rm segmentations.tgz")
    os.system("rm attributes.txt")

    ######################################
    ######################################
def download_cwfid(dataset_dir):
    ds_path = dataset_dir / 'cwfid'
    if ds_path.exists():
        print('Dataset already downloaded')
        return
    # Dataset page: https://github.com/cwfid/dataset.git
    print('Downloading dataset...')
    # Downloading dataset from git repo
    os.system('git clone https://github.com/cwfid/dataset.git')
    os.system('mv dataset ' + str(ds_path))


    ######################################
    ######################################
def download_deepcrack(dataset_dir):
    ds_path = dataset_dir / 'DeepCrack'
    if ds_path.exists():
        print('Dataset already downloaded')
        return
    print('Downloading dataset...')
    # Downloading git repo with zip
    os.system('git clone https://github.com/yhlleo/DeepCrack.git DeepCrackGit')
    os.system('unzip DeepCrackGit/dataset/DeepCrack.zip -d ' + str(ds_path))
    os.system('rm -Rf DeepCrackGit')

    ######################################
    ######################################
def download_foodseg(dataset_dir):
    ds_path = dataset_dir / 'FoodSeg103'
    # Dataset page: https://github.com/LARC-CMU-SMU/FoodSeg103-Benchmark-v1
    if ds_path.exists():
        print('Dataset already downloaded')
        return
    print('Downloading dataset...')
    # Downloading zip
    os.system('wget https://research.larc.smu.edu.sg/downloads/datarepo/FoodSeg103.zip')
    os.system('unzip -P LARCdataset9947 FoodSeg103.zip -d ' + str(ds_path))
    os.system('rm FoodSeg103.zip')



    ######################################
    ######################################
def download_isaid(dataset_dir):
    ds_path = dataset_dir / 'isaid'
    if ds_path.exists():
        print('Dataset already downloaded')
        return
    print('Downloading dataset...')
    # Download from Google Drive
    # Download val images https://drive.google.com/drive/folders/1RV7Z5MM4nJJJPUs6m9wsxDOJxX6HmQqZ?usp=share_link4
    gdown.download_folder(id='1RV7Z5MM4nJJJPUs6m9wsxDOJxX6HmQqZ', output=str(ds_path))
    os.system(f'unzip {ds_path / "part1.zip"} -d {ds_path / "val_images"}')
    os.system(f'rm {ds_path / "part1.zip"}')
    # Download val mask https://drive.google.com/drive/folders/1jlVr4ClmeBA01IQYx7Aq3Scx2YS1Bmpb
    gdown.download_folder(id='1jlVr4ClmeBA01IQYx7Aq3Scx2YS1Bmpb', output=str(ds_path))
    os.system(f'unzip {ds_path / "images.zip"} -d {ds_path / "val_masks"}')
    os.system(f'rm {ds_path / "images.zip"}')

    # train images and masks
    gdown.download_folder(id='1MvSH7sNaY4p4lhwAU_BG3y7zth6-rtrD', output=str(ds_path))
    os.system(f'unzip {ds_path / "part1.zip"} -d {ds_path / "train_images"}')
    os.system(f'unzip {ds_path / "part2.zip"} -d {ds_path / "train_images"}')
    os.system(f'unzip {ds_path / "part3.zip"} -d {ds_path / "train_images"}')
    os.system(f'rm {ds_path / "part1.zip"}')
    os.system(f'rm {ds_path / "part2.zip"}')
    os.system(f'rm {ds_path / "part3.zip"}')
    gdown.download(id='1YLjZ1cmA9PH3OfzMF-eq6T-O9FTGvSrx', output=str(ds_path / 'train_masks.zip'))
    os.system(f'unzip {ds_path / "train_masks.zip"} -d {ds_path / "train_masks"}')
    os.system(f'rm {ds_path / "train_masks.zip"}')
    os.system(f'rm -rf {ds_path / "1"}')
    

# https://drive.google.com/drive/folders/1MvSH7sNaY4p4lhwAU_BG3y7zth6-rtrD?usp=sharing
# https://drive.google.com/file/d/1pEmwJtugIWhiwgBqOtplNUtTG2T454zn/view?usp=drive_link, https://drive.google.com/file/d/1JBWCHdyZOd9ULX0ng5C9haAt3FMPXa3v/view?usp=drive_link, https://drive.google.com/file/d/1BlaGYNNEKGmT6OjZjsJ8HoUYrTTmFcO2/view?usp=drive_link

# semantic masks
# https://drive.google.com/file/d/1YLjZ1cmA9PH3OfzMF-eq6T-O9FTGvSrx/view?usp=drive_link





    ######################################
    ######################################
def download_kvasir(dataset_dir):
    ds_path = dataset_dir / "kvasir-instrument"
    if ds_path.exists():
        print('Dataset already downloaded')
        return
    print('Downloading dataset...')
    os.system("wget https://datasets.simula.no/downloads/kvasir-instrument.zip")
    os.system("unzip kvasir-instrument.zip -d " + str(dataset_dir))
    os.system("rm kvasir-instrument.zip")

    print('Creating images directory...')
    os.system(f"tar -xvzf {ds_path}/images.tar.gz -C {ds_path}")
    os.system(f"tar -xvzf {ds_path}/masks.tar.gz -C {ds_path}")
    os.system(f"rm {ds_path}/images.tar.gz")
    os.system(f"rm {ds_path}/masks.tar.gz")


    ######################################
    ######################################
def download_mhp(dataset_dir):
    ds_path = dataset_dir / 'LV-MHP-v1'
    if ds_path.exists():
        print('Dataset already downloaded')
        return
    print('Downloading dataset...')
    # Download from Google Drive
    gdown.download(f"https://drive.google.com/uc?export=download&confirm=pbef&id=1hTS8QJBuGdcppFAr_bvW2tsD9hW_ptr5")
    os.system('unzip LV-MHP-v1.zip -d ' + str(dataset_dir))
    os.system('rm LV-MHP-v1.zip')

    ######################################
    ######################################
def download_paxray(dataset_dir):
    ds_path = dataset_dir / 'paxray_dataset'
    if ds_path.exists():
        print('Dataset already downloaded')
        return
    print('Downloading dataset...')
    # Download from Google Drive
    # https://drive.google.com/file/d/19HPPhKf9TDv4sO3UV-nI3Jhi4nCv_Zyc/view?usp=share_link
    gdown.download(f"https://drive.google.com/uc?export=download&confirm=pbef&id=19HPPhKf9TDv4sO3UV-nI3Jhi4nCv_Zyc")
    os.system('unzip paxray_dataset.zip -d ' + str(ds_path))
    os.system('rm paxray_dataset.zip')

    ######################################
    ######################################
def download_pst900(dataset_dir):
    ds_path = dataset_dir / 'PST900_RGBT_Dataset'
    if ds_path.exists():
        print('Dataset already downloaded')
        return
    print('Downloading dataset...')
    # Download from Google Drive
    # link: https://drive.google.com/open?id=1hZeM-MvdUC_Btyok7mdF00RV-InbAadm
    gdown.download("https://drive.google.com/uc?export=download&confirm=pbef&id=1hZeM-MvdUC_Btyok7mdF00RV-InbAadm")
    os.system('unzip PST900_RGBT_Dataset.zip -d ' + str(dataset_dir))
    os.system('rm PST900_RGBT_Dataset.zip')

    ######################################
    ######################################
def download_suim(dataset_dir):
    ds_path = dataset_dir / "SUIM"
    if ds_path.exists():
        print('Dataset already downloaded')
        return
    print('Downloading dataset...')
    ds_path.mkdir(parents=True, exist_ok=True)
    # Downloading zip
    gdown.download(id='1diN3tNe2nR1eV3Px4gqlp6wp3XuLBwDy')
    os.system("unzip TEST.zip -d " + str(ds_path / 'test'))
    os.system("rm TEST.zip")

    # Downloading zip
    gdown.download(id='1YWjUODQWwQ3_vKSytqVdF4recqBOEe72')
    os.system("unzip train_val.zip -d " + str(ds_path / 'train'))
    os.system("rm train_val.zip")

    ######################################
    ######################################
def download_worldfloods(dataset_dir):
    ds_path = dataset_dir / 'WorldFloods'
    if ds_path.exists():
        print('Dataset already downloaded')
        return
    print('Downloading dataset...')
    # Download from Google Drive
    gdown.download(id='11O6aKZk4R6DERIx32o4mMTJ5dtzRRKgV')
    os.system(f"unzip worldfloods_v1_0_sample.zip -d {str(ds_path)}")
    os.system("rm worldfloods_v1_0_sample.zip") 

    ######################################
    ######################################
def download_zerowaste(dataset_dir):
    ds_path = dataset_dir / 'zerowaste-f'
    if ds_path.exists():
        print('Dataset already downloaded')
        return
    print('Downloading dataset...')
    # Downloading zip
    os.system('wget https://zenodo.org/record/6412647/files/zerowaste-f-final.zip')
    os.system('unzip zerowaste-f-final.zip -d ' + str(ds_path))
    os.system('rm zerowaste-f-final.zip')

def download_everything(detectron2_datasets_path):
    dsdir = Path(os.getenv('DETECTRON2_DATASETS', 'datasets')) if detectron2_datasets_path is None else Path(detectron2_datasets_path)
    dsdir.mkdir(parents=True, exist_ok=True)

    # change current directory to dsdir
    os.chdir(dsdir)
    # download
    dataset_dir = Path()  # now the directory is the same
    download_zerowaste(dataset_dir)
    download_worldfloods(dataset_dir)
    download_suim(dataset_dir)
    download_pst900(dataset_dir)
    download_paxray(dataset_dir)
    download_mhp(dataset_dir)
    download_kvasir(dataset_dir)
    download_isaid(dataset_dir)
    download_foodseg(dataset_dir)
    download_deepcrack(dataset_dir)
    download_cwfid(dataset_dir)
    download_cub(dataset_dir)
    download_corrosion(dataset_dir)
    download_chase(dataset_dir)
    download_atlantis(dataset_dir)



if __name__ == '__main__':
    from fire import Fire
    Fire(download_everything)
