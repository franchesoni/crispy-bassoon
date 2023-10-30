import os
from pathlib import Path

import gdown

def dummy(detectron2_datasets_path):
    # dataset_dir = Path(os.getenv('DETECTRON2_DATASETS', 'datasets'))
    dataset_dir = Path(detectron2_datasets_path)
    # change current directory to dataset_dir
    os.chdir(dataset_dir)

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

    # train images
    gdown.download_folder(id='1MvSH7sNaY4p4lhwAU_BG3y7zth6-rtrD', output='tmp')

# https://drive.google.com/drive/folders/1MvSH7sNaY4p4lhwAU_BG3y7zth6-rtrD?usp=sharing
# https://drive.google.com/file/d/1pEmwJtugIWhiwgBqOtplNUtTG2T454zn/view?usp=drive_link, https://drive.google.com/file/d/1JBWCHdyZOd9ULX0ng5C9haAt3FMPXa3v/view?usp=drive_link, https://drive.google.com/file/d/1BlaGYNNEKGmT6OjZjsJ8HoUYrTTmFcO2/view?usp=drive_link

# semantic masks
# https://drive.google.com/file/d/1YLjZ1cmA9PH3OfzMF-eq6T-O9FTGvSrx/view?usp=drive_link





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

# wget --header="Host: storage.googleapis.com" --header="User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36 Edg/118.0.2088.69" --header="Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7" --header="Accept-Language: en,es-419;q=0.9,es;q=0.8,fr;q=0.7,pt-BR;q=0.6,pt;q=0.5" --header="Cookie: _ga=GA1.3.2046005398.1697120548; _ga_L31B3DX5TC=GS1.1.1697120548.1.1.1697120586.0.0.0" --header="Connection: keep-alive" "https://storage.googleapis.com/cos-osf-prod-files-us/9a90c143c7cc7642ada45f607b74fb45fb498c981d1064893f1bbc4b8b1b49fe?response-content-disposition=attachment%3B%20filename%3D%22files_used_for_testing.txt%22%3B%20filename%2A%3DUTF-8%27%27files_used_for_testing.txt&GoogleAccessId=files-us%40cos-osf-prod.iam.gserviceaccount.com&Expires=1698671096&Signature=FKdGRNfZK9w%2BOp%2F%2BUxtx3LM%2BJWpWOj9H2bPjdOZ0GQ2kKw6YftynexTwJQBYcnbOtBB4hXeC76km3leZoyv3eCMKKOMkVqvwTl%2BJ2%2FDeICMHnT0D3ZwyzBrlFDQmngrAiJWn8cO4u3nUZzukOpbRBZk04T7WtnRbs2fuzGVn0N5cdpuWICFuyoJCfb6isCwmrT61LtN%2FifrEqZjCQn3tuVPKRtTyXVVs8Z6msNBWG6%2BtGtPRWkYxPWVQL%2BwmfYyLTLCYOhMcJ5KvT24ENi3S9sfbIYAEkQohjS8vdveEd9Is3kcv8sBQa3vs0WpBUmjZ8ST0nkP70yMmjw55I9qcgQ%3D%3D" -c -O 'files_used_for_testing.txt'

# wget --header="Host: storage.googleapis.com" --header="User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36 Edg/118.0.2088.69" --header="Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7" --header="Accept-Language: en,es-419;q=0.9,es;q=0.8,fr;q=0.7,pt-BR;q=0.6,pt;q=0.5" --header="Cookie: _ga=GA1.3.2046005398.1697120548; _ga_L31B3DX5TC=GS1.1.1697120548.1.1.1697120586.0.0.0" --header="Connection: keep-alive" "https://storage.googleapis.com/cos-osf-prod-files-us/42f066dabc92d071e9a7418c1817d0a760e7f998d483a6729b627ba6356b3851?response-content-disposition=attachment%3B%20filename%3D%22files_used_for_training.txt%22%3B%20filename%2A%3DUTF-8%27%27files_used_for_training.txt&GoogleAccessId=files-us%40cos-osf-prod.iam.gserviceaccount.com&Expires=1698671177&Signature=psQUhQQgHlEJWiMDfPMVpD3pr7vlVOccpU%2FXDn44Ub%2B5g1s2aaXgeHOOmnPlw2e%2FD9%2BGmDb9WUrbzNw4fS9ylr4%2Br5A%2BdslgiH7irxNyjUfwZ8dBkjCynExLJHAT%2BWT9uFNtbSzE0fARicy17T9g%2F2yxvCex3%2Fk6eIbsb99wySHMRULfi7X5vZYgkwopSADZ5%2FFXDKSpGN4QGZLLhYBZ%2F0bSXzNA8R87HxQ7AvNy%2FPH%2FbLsLtn58nkqVUs8ML42UNZngaeLyCvIGg9ITHyZjaSYZYlL%2FCpRg4MkKpztdDFeC8RR%2BLgCea%2BHMm8qPaTFCgPyOhY8E4jGztUeuWN5D0w%3D%3D" -c -O 'files_used_for_training.txt'

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
    dataset_dir = Path(os.getenv('DETECTRON2_DATASETS', 'datasets')) if detectron2_datasets_path is None else Path(detectron2_datasets_path)

    # change current directory to dataset_dir
    os.chdir(dataset_dir)
    # download
    download_zerowaste(dataset_dir)



if __name__ == '__main__':
    from fire import Fire
    Fire(download_everything)