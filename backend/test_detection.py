import cv2
import numpy as np
import urllib.request
from tracker import ObjectTracker

def test_real_detection():
    print("Testing with a real image of a person (consecutive frames)...")
    url = "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/bus.jpg"
    image_path = "sample_bus.jpg"
    try:
        print(f"Downloading sample image from {url}...")
        urllib.request.urlretrieve(url, image_path)
        print("Download complete.")
    except Exception as e:
        print(f"Failed to download sample image: {e}")
        return

    tracker = ObjectTracker()
    frame = cv2.imread(image_path)
    if frame is None:
        print("Failed to read downloaded image.")
        return
        
    print(f"Image shape: {frame.shape}")
    print("Running tracking pipeline multiple times to confirm tracks...")
    
    try:
        # Feed the frame 4 times to let tracks transition from Tentative to Confirmed
        for i in range(1, 5):
            print(f"\n--- Processing Frame {i} ---")
            annotated_frame, tracks, counts = tracker.process_frame(frame.copy())
            print(f"Frame {i} processed successfully.")
            print(f"Confirmed tracks: {len(tracks)}")
            print(f"Class counts: {counts}")
            for t in tracks:
                print(f"  Track ID: {t['track_id']}, Class: {t['class_name']}, BBox: {t['bbox']}")
        print("\nAll frames processed without crashing!")
    except Exception as e:
        print("\n!!! PIPELINE CRASHED !!!")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_real_detection()
