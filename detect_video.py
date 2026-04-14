import cv2
from ultralytics import YOLO
model=YOLO('yolov8m.pt')

cap = cv2.VideoCapture("videos/demo.mp4")

while True:
    ret,frame=cap.read()
    if not ret:
        break
    results=model(frame)
    annotated_frame=results[0].plot()
    cv2.imshow("Detection",annotated_frame)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()