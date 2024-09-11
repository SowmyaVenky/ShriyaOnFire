from ultralytics import YOLO
import os
import cv2

ROOT_PATH = '/home/sowmyavenky/ShriyaOnFire/training/runs/detect/train/weights/'

BEST_PT = os.path.join(ROOT_PATH, 'best.pt')

# Load the YOLOv8 model
model = YOLO(BEST_PT)

# Export the model to TFLite format
model.export(format="tflite")  # creates 'yolov8n_float32.tflite'

imgdir = os.path.join('../', 'image_extractions', 'roboflowdata', 'train', 'images', '0e25edb4-5c37-40d8-80fc-cb1e4a64408c_frame1750_jpg.rf.3e67bc85c5fcebcf9daa7c9b4f723fd5.jpg')

# Load the exported TFLite model
TFLITE_PATH = os.path.join(ROOT_PATH, 'best_saved_model/')
tflite_model = YOLO(TFLITE_PATH)

# from ndarray
im2 = cv2.imread(imgdir)
results = tflite_model.predict(source=im2) 

# View results
for r in results:
    print(r.boxes)  # print the Boxes object containing the detection bounding boxes