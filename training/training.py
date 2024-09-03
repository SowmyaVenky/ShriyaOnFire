from ultralytics import YOLO

model = YOLO("yolov8n.yaml")
# results = model.train(data="coco_dataset.yaml", epochs=5)

results = model.train(data="fire_dataset.yaml", epochs=5, imgsz=512)