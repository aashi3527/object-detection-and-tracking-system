from turtle import width

import cv2
import time
from scipy.__config__ import show
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

def process_video(input_path, output_path,show=False):

    model = YOLO("yolov8n.pt")
    tracker = DeepSort(max_age=30)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("error:cannot open the video")
        exit()
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_path, fourcc, 20, (640, 480))

    line_y=200
    counted_ids = set()
    count=0

    while True:
        ret, frame = cap.read()

        if not ret:
            print("video ended")
            break

        frame = cv2.resize(frame, (640, 480))

        results = model(frame, imgsz=320, conf=0.5, verbose=False)[0]

        detections = []

        for box in results.boxes:
            cls = int(box.cls[0])
            x1, y1, x2, y2 = box.xyxy[0]
            conf = float(box.conf[0])

            detections.append(([x1, y1, x2-x1, y2-y1], conf, "object"))

        tracks = tracker.update_tracks(detections, frame=frame)

        cv2.line(frame, (0,line_y), (640,line_y), (255,0,0),2)

        for track in tracks:
            if not track.is_confirmed():
                continue

            l, t, w, h = track.to_ltrb()
            track_id = track.track_id
        
            centre_y=int(t+h/2)
            if centre_y>line_y and track_id not in counted_ids:
                counted_ids.add(track_id)
                count+=1

            cv2.rectangle(frame, (int(l), int(t)), (int(l+w), int(t+h)), (0,255,0), 2)
            cv2.putText(frame, f"ID {track_id}", (int(l), int(t)-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            cv2.putText(frame, f"Count: {count}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            out.write(frame)
            if show:
                cv2.imshow("Tracking", frame)

                if cv2.waitKey(1) == 27:
                    break
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    return output_path
if __name__ == "__main__":
    input_video = "videos/demo.mp4"
    output_video = "output_demo.mp4"

    process_video(input_video, output_video, show=True)