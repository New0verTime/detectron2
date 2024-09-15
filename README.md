# Detectron2

## Files and descriptions
- CoCo: Coco format training and validation data
- output: The output after the training phase
- check_coco_annotation.py: View the training data in Coco format that is converted from Labelme format
- check_trained_model.py: View the trained model
- train.py: Train model
- demo.py: Use the trained model to infer from video. Open cmd, run 
python demo.py --config-file config.yaml --video-input video/tokyo2.mp4 --confidence-threshold 0.6 --output video-output.mkv --opts MODEL.WEIGHTS output/model_final.pth
