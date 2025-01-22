import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import os

ROOT_PATH = '/home/sowmyavenky/ShriyaOnFire/training/runs/detect/train/weights/'

BEST_PT = os.path.join(ROOT_PATH, 'best.pt')

# Load the YOLOv8 model
model = YOLO(BEST_PT)

cap = cv2.VideoCapture(0)

cap.set(3, 640)
cap.set(4, 480)

while cap.isOpened():
    ret, frame1 = cap.read()
    frame = cv2.resize(frame1, (640, 480)) 
    
    results = model.predict(source=frame) 

    for r in results:        
      annotator = Annotator(frame)
        
      boxes = r.boxes
      for box in boxes:
        b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
        c = box.cls
        annotator.box_label(b, model.names[int(c)], color=(0, 255, 0),txt_color=(0, 0, 255),)
          
    img = annotator.result()  
    cv2.imshow('YOLO V8 Detection', img)   
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
