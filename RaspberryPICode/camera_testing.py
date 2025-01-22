import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import os

cap = cv2.VideoCapture(0)

cap.set(3, 640)
cap.set(4, 480)

while cap.isOpened():
    ret, frame1 = cap.read()
    frame = cv2.resize(frame1, (640, 480)) 
    cv2.imshow('YOLO V8 Detection', frame)   
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
