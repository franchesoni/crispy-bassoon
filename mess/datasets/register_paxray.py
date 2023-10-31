
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg
from detectron2.utils.colormap import colormap

LUNGS = ('others', 'lung')
MEDIANSTINUM = ('others', 'mediastinum')
BONES = ('others', 'bones')
DIAPHRAGM = ('others', 'diaphragm')

def register_dataset(root):
    ds_name = 'paxray'
    root = os.path.join(root, 'paxray_dataset')

    for split, image_dirname, sem_seg_dirname, class_names in [
        ('test_lungs', 'images_detectron2/test', 'annotations_detectron2/test/lungs', LUNGS),
        ('test_mediastinum', 'images_detectron2/test', 'annotations_detectron2/test/mediastinum', MEDIANSTINUM),
        ('test_bones', 'images_detectron2/test', 'annotations_detectron2/test/bones', BONES),
        ('test_diaphragm', 'images_detectron2/test', 'annotations_detectron2/test/diaphragm', DIAPHRAGM),

        ('train_lungs', 'images_detectron2/train', 'annotations_detectron2/train/lungs', LUNGS),
        ('train_mediastinum', 'images_detectron2/train', 'annotations_detectron2/train/mediastinum', MEDIANSTINUM),
        ('train_bones', 'images_detectron2/train', 'annotations_detectron2/train/bones', BONES),
        ('train_diaphragm', 'images_detectron2/train', 'annotations_detectron2/train/diaphragm', DIAPHRAGM),

    ]:
        image_dir = os.path.join(root, image_dirname)
        gt_dir = os.path.join(root, sem_seg_dirname)
        full_name = f'{ds_name}_sem_seg_{split}'
        DatasetCatalog.register(
            full_name,
            lambda x=image_dir, y=gt_dir: load_sem_seg(
                y, x, gt_ext='png', image_ext='png'
            ),
        )
        MetadataCatalog.get(full_name).set(
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type='sem_seg',
            ignore_label=255,
            stuff_classes=class_names,
            stuff_colors=colormap(rgb=True),
            classes_of_interest=[1],
            background_class=0,
        )


_root = os.getenv('DETECTRON2_DATASETS', 'datasets')
register_dataset(_root)
