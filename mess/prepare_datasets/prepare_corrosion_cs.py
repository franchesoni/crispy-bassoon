
# run python mess/prepare_datasets/prepare_corrosion_cs.py

import os
from pathlib import Path


# Classes:
# 0: background
# 1: Fair
# 2: Poor
# 3: Severe

def download_dataset(dataset_dir):
    """
    Downloads the dataset
    """
    print('Downloading dataset...')
    filesdir = dataset_dir / 'files'
    filesdir.mkdir(parents=True)
    # Downloading zip
    os.system(f'wget https://figshare.com/ndownloader/files/31729733 -P {filesdir}')

def extract_dataset(dataset_dir):
    filesdir = dataset_dir / 'files'
    os.system(f'unzip {str(filesdir / "31729733")} -d ' + str(dataset_dir))
    src_dir = dataset_dir / 'Corrosion Condition State Classification'
    out_dir = dataset_dir 
    command = f'mv "{src_dir}"/* "{out_dir}"/'
    os.system(command)
    os.system(f'rm -rf "{src_dir}"')

def prepare_corrosion(ds_path):
    pass  # I don't see the point of doing the following
    # for split in ['Train', 'Test']:
    #     # Edit test masks
    #     for mask_path in tqdm.tqdm(sorted(ds_path.glob(f'original/{split}/masks/*.png'))):
    #         # Open mask
    #         mask = np.array(Image.open(mask_path))
    #         # 'Portable network graphics' format, so no further processing needed
    #         # Save mask
    #         Image.fromarray(mask).save(mask_path)
    # print(f'Saved images and masks of {ds_path.name} dataset')



def main():
    dataset_dir = Path(os.getenv('DETECTRON2_DATASETS', 'datasets'))
    ds_path = dataset_dir / 'Corrosion Condition State Classification'.strip()
    if not ds_path.exists():
        download_dataset(ds_path)
        extract_dataset(ds_path)

    assert ds_path.exists(), f'Dataset not found in {ds_path}'

    prepare_corrosion(ds_path)


if __name__ == '__main__':
    main()
