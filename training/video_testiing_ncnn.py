import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

## Webpage for Raspberry pi 5 
# https://docs.ultralytics.com/guides/raspberry-pi/#how-do-i-set-up-ultralytics-yolov8-on-a-raspberry-pi-without-using-docker

model = YOLO("C:/Venky/Shriya_on_Fire/ShriyaOnFire/training/runs/detect/train/weights/best_ncnn_model/")
# model = YOLO("/home/sowmyavenky/ShriyaOnFire/training/runs/detect/train/weights/last.pt")

cap = cv2.VideoCapture('C:\\Venky\\Shriya_on_Fire\\ShriyaOnFire\\videos\\homam1.mp4')
# cap = cv2.VideoCapture('C:\\Venky\\Shriya_on_Fire\\ShriyaOnFire\\videos\\homam2.mp4')

# cap = cv2.VideoCapture('C:\\Venky\\Shriya_on_Fire\\ShriyaOnFire\\testing_videos\\fire_testing1.mp4')
# cap = cv2.VideoCapture('C:\\Venky\\Shriya_on_Fire\\ShriyaOnFire\\testing_videos\\fire_testing2.mp4')
# cap = cv2.VideoCapture('C:\\Venky\\Shriya_on_Fire\\ShriyaOnFire\\testing_videos\\fire_testing3.mp4')

# cap = cv2.VideoCapture('/home/sowmyavenky/ShriyaOnFire/videos/homam1.mp4')

# cap = cv2.VideoCapture('C:\\Venky\\Shriya_on_Fire\\ShriyaOnFire\\testing_videos\\fire_testing4.mp4')
# cap = cv2.VideoCapture('C:\\Venky\\Shriya_on_Fire\\ShriyaOnFire\\testing_videos\\fire_testing5.webm')
# cap = cv2.VideoCapture('C:\\Venky\\Shriya_on_Fire\\ShriyaOnFire\\testing_videos\\fire_testing6.mp4')
# cap = cv2.VideoCapture('C:\\Venky\\Shriya_on_Fire\\ShriyaOnFire\\testing_videos\\fire_testing7.mp4')
# cap = cv2.VideoCapture('C:\\Venky\\Shriya_on_Fire\\ShriyaOnFire\\testing_videos\\fire_testing8.mp4')
# cap = cv2.VideoCapture('C:\\Venky\\Shriya_on_Fire\\ShriyaOnFire\\testing_videos\\fire_testing9.mp4')

cap.set(3, 640)
cap.set(4, 480)

while cap.isOpened():
    ret, frame1 = cap.read()
    
    if  frame1 is None:
      break
        
    frame = cv2.resize(frame1, (640, 480)) 
    # cv2.imshow('YOLO', frame)
    
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
