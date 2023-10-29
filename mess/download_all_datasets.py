import os
from pathlib import Path

import gdown

def download_everything(detectron2_datasets_path):
    # dataset_dir = Path(os.getenv('DETECTRON2_DATASETS', 'datasets'))
    dataset_dir = Path(detectron2_datasets_path)

    ######################################
    ######################################
    """
    Downloads the dataset
    """
    ds_path = dataset_dir / 'atlantis'
    if not os.path.exists(ds_path):
        ds_path.mkdir(parents=True)
    # Dataset page: https://github.com/smhassanerfani/atlantis.git
    print('Downloading dataset...')
    # Downloading github repo
    os.system(f'git clone https://github.com/smhassanerfani/atlantis.git {ds_path}')

    ######################################
    ######################################
    """
    Downloads the dataset
    """
    ds_path = dataset_dir / 'CHASEDB1'
    print('Downloading dataset...')
    # Downloading zip
    os.system('wget https://staffnet.kingston.ac.uk/~ku15565/CHASE_DB1/assets/CHASEDB1.zip')
    os.system('unzip CHASEDB1.zip -d ' + str(ds_path))
    os.system('rm CHASEDB1.zip')

    ######################################
    ######################################
    ds_path = dataset_dir / 'Corrosion Condition State Classification'
    """
    Downloads the dataset
    """
    print('Downloading dataset...')
    # Downloading zip
    #
    os.system('wget https://figshare.com/ndownloader/files/31729733')
    os.system('unzip 31729733 -d ' + str(dataset_dir))
    os.system('rm 31729733')

    ######################################
    ######################################
    ds_path = dataset_dir / 'CryoNuSeg'
    print('Downloading dataset...')
    # Download from Google Drive
    # Folder: https://drive.google.com/drive/folders/1dgtO_mCcR4UNXw_4zK32NlakbAvnySck
    gdown.download("https://drive.google.com/uc?export=download&confirm=pbef&id=1Or8qSpwLx77ZcWFqOKCKd3upwTUvb0U6")
    gdown.download("https://drive.google.com/uc?export=download&confirm=pbef&id=1WHork0VjF1PTye1xvCTtPtly62uHF72J")
    os.makedirs(ds_path, exist_ok=True)
    os.system('unzip Final.zip -d ' + str(ds_path))
    os.system('unzip masks.zip -d ' + str(ds_path))
    os.system('rm Final.zip')
    os.system('rm masks.zip')

    ######################################
    ######################################
    ds_path = dataset_dir / "CUB_200_2011"
    # Downloading data
    os.system("wget https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz")
    os.system(f"tar -xvzf CUB_200_2011.tgz -C {dataset_dir}")
    os.system(f"rm CUB_200_2011.tgz")

    # Downloading segmentation masks
    os.system("wget https://data.caltech.edu/records/w9d68-gec53/files/segmentations.tgz")
    os.system(f"tar -xvzf segmentations.tgz -C {dataset_dir / 'CUB_200_2011'}")
    os.system(f"rm segmentations.tgz")

    ######################################
    ######################################
    ds_path = dataset_dir / 'cwfid'
    """
    Downloads the dataset
    """
    # Dataset page: https://github.com/cwfid/dataset.git
    print('Downloading dataset...')
    # Downloading dataset from git repo
    os.system('git clone https://github.com/cwfid/dataset.git')
    os.system('mv dataset ' + str(ds_path))

    ######################################
    ######################################
    ds_path = dataset_dir / 'Dark_Zurich'
    ds_path.mkdir(parents=True, exist_ok=True)
    """
    Downloads the dataset
    """
    # Dataset page: https://www.trace.ethz.ch/publications/2019/GCMA_UIoU/
    print('Downloading dataset...')
    # Downloading zip
    os.system('wget https://data.vision.ee.ethz.ch/csakarid/shared/GCMA_UIoU/Dark_Zurich_val_anon.zip')
    os.system('unzip Dark_Zurich_val_anon.zip -d ' + str(ds_path / 'val'))
    os.system('rm Dark_Zurich_val_anon.zip')

    os.system('wget https://data.vision.ee.ethz.ch/csakarid/shared/GCMA_UIoU/Dark_Zurich_train_anon.zip')
    os.system('unzip Dark_Zurich_train_anon.zip -d ' + str(ds_path / 'train'))
    os.system('rm Dark_Zurich_train_anon.zip')

 

    ######################################
    ######################################
    ds_path = dataset_dir / 'DeepCrack'
    """
    Downloads the dataset
    """
    print('Downloading dataset...')
    # Downloading git repo with zip
    os.system('git clone https://github.com/yhlleo/DeepCrack.git')
    os.system('unzip DeepCrack/dataset/DeepCrack.zip -d ' + str(ds_path))
    os.system('rm -R DeepCrack')

    ######################################
    ######################################
    ds_path = dataset_dir / 'DRAM_processed'
    """
    Downloads the dataset
    """
    # Dataset page: https://faculty.runi.ac.il/arik/site/artseg/Dram-Dataset.html
    print('Downloading dataset...')
    # Downloading zip
    os.system('wget https://faculty.runi.ac.il/arik/site/artseg/DRAM_processed.zip')
    os.system('unzip DRAM_processed.zip -d ' + str(ds_path))
    os.system(f'cd {ds_path} && unrar x DRAM_processed.rar')
    os.system('rm DRAM_processed.zip')
    os.system('rm ' + str(ds_path / 'DRAM_processed.rar'))

    ######################################
    ######################################
    ds_path = dataset_dir / 'FoodSeg103'
    """
    Downloads the dataset
    """
    # Dataset page: https://github.com/LARC-CMU-SMU/FoodSeg103-Benchmark-v1
    print('Downloading dataset...')
    # Downloading zip
    os.system('wget https://research.larc.smu.edu.sg/downloads/datarepo/FoodSeg103.zip')
    os.system('unzip -P LARCdataset9947 FoodSeg103.zip -d ' + str(ds_path))
    os.system('rm FoodSeg103.zip')

    ######################################
    ######################################
    ds_path = dataset_dir / 'isaid'
    print('Downloading dataset...')
    # Download from Google Drive
    # Download val images https://drive.google.com/drive/folders/1RV7Z5MM4nJJJPUs6m9wsxDOJxX6HmQqZ?usp=share_link4
    gdown.download_folder(id='1RV7Z5MM4nJJJPUs6m9wsxDOJxX6HmQqZ', output=str(ds_path))
    os.system(f'unzip {ds_path / "part1.zip"} -d {ds_path / "val"}')
    os.system(f'rm {ds_path / "part1.zip"}')
    # Download val mask https://drive.google.com/drive/folders/1jlVr4ClmeBA01IQYx7Aq3Scx2YS1Bmpb
    gdown.download_folder(id='1jlVr4ClmeBA01IQYx7Aq3Scx2YS1Bmpb', output=str(ds_path))
    os.system(f'unzip {ds_path / "images.zip"} -d {ds_path / "raw_val"}')
    os.system(f'rm {ds_path / "images.zip"}')

    ######################################
    ######################################
    ds_path = dataset_dir / "kvasir-instrument"
    """
    Downloads the dataset
    """
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
    ds_path = dataset_dir / 'LV-MHP-v1'
    """
    Downloads the dataset
    """
    print('Downloading dataset...')
    # Download from Google Drive
    gdown.download(f"https://drive.google.com/uc?export=download&confirm=pbef&id=1hTS8QJBuGdcppFAr_bvW2tsD9hW_ptr5")
    os.system('unzip LV-MHP-v1.zip -d ' + str(dataset_dir))
    os.system('rm LV-MHP-v1.zip')

    ######################################
    ######################################
    ds_path = dataset_dir / 'paxray_dataset'
    """
    Downloads the dataset
    """
    print('Downloading dataset...')
    # Download from Google Drive
    # https://drive.google.com/file/d/19HPPhKf9TDv4sO3UV-nI3Jhi4nCv_Zyc/view?usp=share_link
    gdown.download(f"https://drive.google.com/uc?export=download&confirm=pbef&id=19HPPhKf9TDv4sO3UV-nI3Jhi4nCv_Zyc")
    os.system('unzip paxray_dataset.zip -d ' + str(ds_path))
    os.system('rm paxray_dataset.zip')

    ######################################
    ######################################
    ds_path = dataset_dir / 'PST900_RGBT_Dataset'
    """
    Downloads the dataset
    """
    print('Downloading dataset...')
    # Download from Google Drive
    # link: https://drive.google.com/open?id=1hZeM-MvdUC_Btyok7mdF00RV-InbAadm
    gdown.download("https://drive.google.com/uc?export=download&confirm=pbef&id=1hZeM-MvdUC_Btyok7mdF00RV-InbAadm")
    os.system('unzip PST900_RGBT_Dataset.zip -d ' + str(dataset_dir))
    os.system('rm PST900_RGBT_Dataset.zip')

    ######################################
    ######################################
    ds_path = dataset_dir / "SUIM"
    """
    Downloads the dataset
    """
    print('Downloading dataset...')
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
    ds_path = dataset_dir / 'WorldFloods'
    print('Downloading dataset...')
    # Download from Google Drive
    # https://drive.google.com/drive/folders/1Bp1FXppikOpQrgth2lu5WjpYX7Lb2qOW?usp=share_link
    gdown.download_folder(id='1Bp1FXppikOpQrgth2lu5WjpYX7Lb2qOW', output=str(ds_path / 'test'))

    ######################################
    ######################################
    ds_path = dataset_dir / 'zerowaste-f'
    """
    Downloads the dataset
    """
    print('Downloading dataset...')
    # Downloading zip
    os.system('wget https://zenodo.org/record/6412647/files/zerowaste-f-final.zip')
    os.system('unzip zerowaste-f-final.zip -d ' + str(ds_path))
    os.system('rm zerowaste-f-final.zip')


if __name__ == '__main__':
    from fire import Fire
    Fire(download_everything)