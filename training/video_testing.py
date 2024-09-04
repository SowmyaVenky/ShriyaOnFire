import cv2
from ultralytics import YOLO

model = YOLO("/home/sowmyavenky/ShriyaOnFire/training/runs/detect/train/weights/last.pt")

cap = cv2.VideoCapture('/home/sowmyavenky/ShriyaOnFire/videos/homam1.mp4')
while cap.isOpened():
    ret, frame = cap.read()
    cv2.imshow('YOLO', frame)
    
    results = model.predict(source=frame) 
    for r in results:
      print(r.boxes)  # print the Boxes object containing the detection bounding boxes    

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
