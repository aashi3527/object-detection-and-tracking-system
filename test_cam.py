import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera not opening")
else:
    print("Camera opened")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Frame not received")
        break

    cv2.imshow("Test", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()