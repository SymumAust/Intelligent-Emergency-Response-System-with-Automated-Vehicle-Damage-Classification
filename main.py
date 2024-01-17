from ultralytics import YOLO

model = YOLO('yolov8n-cls.pt')  # load a pretrained model (recommended for training)


results = model.train(data='G:\PR\dataset', epochs=500, imgsz=100)