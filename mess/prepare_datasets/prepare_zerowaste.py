
# run python mess/prepare_datasets/prepare_zerowaste.py

import os
from pathlib import Path


def download_dataset(ds_path):
    """
    Downloads the dataset
    """
    print('Downloading dataset...')
    filesdir = ds_path / 'files'
    filesdir.mkdir(parents=True)
    # Downloading zip
    os.system('wget https://zenodo.org/record/6412647/files/zerowaste-f-final.zip -P ' + str(filesdir))

def extract_dataset(ds_path):
    filepath = ds_path / 'files' / 'zerowaste-f-final.zip'
    os.system(f'unzip {str(filepath)} -d ' + str(ds_path))



def main():
    dataset_dir = Path(os.getenv('DETECTRON2_DATASETS', 'datasets'))
    ds_path = dataset_dir / 'zerowaste-f'
    if not ds_path.exists():
        download_dataset(ds_path)
        extract_dataset(ds_path)

    assert ds_path.exists(), f'Dataset not found in {ds_path}'

    print(f'Saved images and masks of {ds_path.name} dataset')


if __name__ == '__main__':
    main()
