import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import os

#ROOT_PATH = '/home/sowmyavenky/ShriyaOnFire/training/runs/detect/train/weights/'
#VIDEO_PATH = '/home/sowmyavenky/ShriyaOnFire/testing_videos/'

ROOT_PATH = 'C:/Venky/Shriya_on_Fire/ShriyaOnFire/training/runs/detect/train2/weights/'
TEST_VIDEO_PATH = 'C:/Venky/Shriya_on_Fire/ShriyaOnFire/testing_videos/'
VIDEO_PATH = 'C:/Venky/Shriya_on_Fire/ShriyaOnFire/videos/'

BEST_PT = os.path.join(ROOT_PATH, 'best.pt')

TEST_VIDEO = os.path.join(VIDEO_PATH, 'gas_2.mp4')
# TEST_VIDEO = os.path.join(TEST_VIDEO_PATH, 'pump_stove_demo.mp4')
# TEST_VIDEO = os.path.join(TEST_VIDEO_PATH, 'pump_stove_demo1.mp4')


# Load the YOLOv8 model
model = YOLO(BEST_PT)
print("Opening " + TEST_VIDEO)
cap = cv2.VideoCapture(TEST_VIDEO)

cap.set(3, 640)
cap.set(4, 480)

while cap.isOpened():
    ret, frame1 = cap.read()
    if  frame1 is None:
      break
    
    # frame = frame1
    frame = cv2.resize(frame1, (640, 480)) 
    #cv2.imshow('YOLO', frame)
    
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
