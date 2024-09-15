
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
import numpy as np
import os
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
register_coco_instances("my_dataset_train", {}, "CoCo\\train.json", "CoCo\\train_img")
register_coco_instances("my_dataset_val", {}, "CoCo\\val.json", "CoCo\\val_img")
if __name__ == '__main__':
    from detectron2.engine import DefaultTrainer

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("my_dataset_train",)
    cfg.DATASETS.TEST = ("my_dataset_val",)
    cfg.TEST.EVAL_PERIOD = 100  # Evaluate the model after every 100 iterations (you can adjust this)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 2  # Adjust batch size based on your GPU memory
    cfg.SOLVER.BASE_LR = 0.0001  # pick a good learning rate
    cfg.SOLVER.MAX_ITER = 5000   # Number of iterations to train
    cfg.SOLVER.STEPS = []        # Do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # RoIHead batch size, can increase if you have enough memory
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
    import yaml
    config_yaml_path = "config.yaml"
    with open(config_yaml_path, 'w') as file:
        yaml.dump(cfg, file)