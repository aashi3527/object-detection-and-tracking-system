import cv2
import numpy as np
from tracker import ObjectTracker

def test_tracker():
    print("Testing YOLO & DeepSORT initialization...")
    # Initialize ObjectTracker. This will also download yolov8n.pt if it's not present.
    tracker = ObjectTracker()
    print("Tracker loaded successfully!")
    
    # Create a dummy frame (a black 640x360 BGR image)
    dummy_frame = np.zeros((360, 640, 3), dtype=np.uint8)
    
    # Draw a white box in the middle to simulate an object
    cv2.rectangle(dummy_frame, (150, 150), (250, 250), (255, 255, 255), -1)
    
    print("Processing mock frame through tracking pipeline...")
    # Process the frame: detection -> tracking -> annotations
    annotated_frame, tracks, counts = tracker.process_frame(dummy_frame)
    print("Mock frame processed successfully!")
    print(f"Active tracks data: {tracks}")
    print(f"Active counts by class: {counts}")
    print("Environment test passed successfully!")

if __name__ == "__main__":
    test_tracker()
