from config import datasets_path
import os
os.environ['DETECTRON2_DATASETS'] = str(datasets_path)
from pathlib import Path
import numpy as np
from mess.datasets.TorchvisionDataset import TorchvisionDataset

train_dataset_names = [
        'foodseg103_sem_seg_train',
        'mhp_v1_sem_seg_train',
        'suim_sem_seg_train',
        'zerowaste_sem_seg_train',
        'atlantis_sem_seg_train',
        'cwfid_sem_seg_train',
        'kvasir_instrument_sem_seg_train',
        'isaid_sem_seg_train',  
        'deepcrack_sem_seg_train',
        'corrosion_cs_sem_seg_train', 


]

class_names_to_ignore = [
            "background",
            "others",
            "unlabeled",
            "background (waterbody)",
            "background or trash",
        ]

def main(runspath):
    runspath = Path(runspath)
    completed = []
    for dataset_name in train_dataset_names:
        ds = TorchvisionDataset(dataset_name, transform=np.array)
        class_names = ds.class_names
        print('='*80)
        print(f"dataset: {dataset_name}")
        complete = True  # assume complete
        for seed in [0,1,2]:
            print(f'seed {seed}:\n')
            res_path = runspath / dataset_name / f'results_seed_{seed}.json'
            if not res_path.exists():
                complete = False
                print(f"missing results for seed {seed}, breaking...")
                continue
            n_completed_classes = len(class_names)
            for class_name in class_names:
                if class_name in class_names_to_ignore:
                    continue
                vectors_path = runspath / dataset_name / f'seed_vectors_{class_name}_seed_{seed}.npy'
                clicks_path = runspath / dataset_name / f'clicks_so_far_{class_name}_seed_{seed}.npy'
                if not vectors_path.exists() or not clicks_path.exists():
                    complete = False
                    #print(f"missing vectors or clicks for class {class_name}, seed {seed}, breaking...")
                    n_completed_classes -= 1
            print(f'completed {n_completed_classes} / {len(class_names)} classes')
        if complete:
            print(f"dataset {dataset_name} is complete")
            completed.append(dataset_name)
    print('%'*80)
    print('completed:', completed)
    print('incomplete:', [ds for ds in train_dataset_names if ds not in completed])
        
            



if __name__ == '__main__':
    from fire import Fire
    Fire(main)
    
