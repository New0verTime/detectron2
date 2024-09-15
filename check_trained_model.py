from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2 import model_zoo
import os, random, cv2
from detectron2.data.datasets import register_coco_instances

register_coco_instances("my_dataset_train", {}, "CoCo\\train.json", "CoCo\\train_img")
im = cv2.imread("CoCo/val_img/labelled_data2/00025.png")

cfg = get_cfg()
cfg.merge_from_file("config.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
cfg.MODEL.WEIGHTS = "output/model_final.pth"
predictor = DefaultPredictor(cfg)
outputs = predictor(im)
v = Visualizer(im[:, :, ::-1], metadata=MetadataCatalog.get("my_dataset_train"), scale=0.8)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
cv2.imshow("abc",out.get_image()[:, :, ::-1])
cv2.waitKey(0)
