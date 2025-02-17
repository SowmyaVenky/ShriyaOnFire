import cv2
from PIL import Image
import os
from ultralytics import YOLO

model = YOLO("/home/sowmyavenky/ShriyaOnFire/training/runs/detect/train/weights/last.pt")
imgdir = os.path.join('../', 'image_extractions', 'roboflowdata', 'train', 'images', '0e25edb4-5c37-40d8-80fc-cb1e4a64408c_frame1750_jpg.rf.3e67bc85c5fcebcf9daa7c9b4f723fd5.jpg')

# from ndarray
im2 = cv2.imread(imgdir)
results = model.predict(source=im2) 

# View results
for r in results:
    print(r.boxes)  # print the Boxes object containing the detection bounding boxes

