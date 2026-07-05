import cv2
import numpy as np
import time
import os
from tracker import ObjectTracker

def run_verification():
    video_path = r"C:\Users\HP\object_tracking_system\input_demo.mp4"
    output_image_path = r"C:\Users\HP\.gemini\antigravity\brain\4c88fc58-7bbf-405c-b4b9-c7543c707341\annotated_demo_frame.jpg"
    
    print(f"Opening demo video: {video_path}")
    if not os.path.exists(video_path):
        print(f"ERROR: Video file not found at {video_path}")
        return
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("ERROR: OpenCV could not open the video file.")
        return
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video metadata: {width}x{height} @ {fps:.2f} FPS | Total frames: {total_frames}")
    
    # Initialize our tracker
    tracker = ObjectTracker(tracker_max_age=25)
    
    frame_count = 0
    start_time = time.time()
    unique_ids = set()
    saved_snapshot = False
    
    # We will process the first 120 frames (approx. 4-5 seconds of video)
    max_process_frames = 120
    print(f"Processing first {max_process_frames} frames...")
    
    while cap.isOpened() and frame_count < max_process_frames:
        ret, frame = cap.read()
        if not ret:
            print("Reached end of video stream or failed to read frame.")
            break
            
        frame_count += 1
        
        # Downscale for tracking (same as React frontend)
        target_width = 640
        aspect_ratio = height / width
        target_height = int(target_width * aspect_ratio)
        frame_resized = cv2.resize(frame, (target_width, target_height))
        
        # Run tracking pipeline
        # allowed_classes = None tracks everything, or we can filter for vehicles
        allowed_classes = ['car', 'truck', 'bus', 'motorcycle', 'person']
        annotated_frame, tracks, counts = tracker.process_frame(
            frame_resized, 
            allowed_classes=allowed_classes, 
            conf_threshold=0.35,
            show_trails=True
        )
        
        # Record unique track IDs
        for t in tracks:
            unique_ids.add(t['track_id'])
            
        # Print status every 30 frames
        if frame_count % 30 == 0:
            print(f"Frame {frame_count}/{max_process_frames} | Active in frame: {len(tracks)} | Cumulative tracks: {len(unique_ids)}")
            
        # Save a snapshot on frame 75 (where vehicles are likely fully entered and tracked)
        # or if we have at least 2 active tracks
        if not saved_snapshot and (frame_count == 75 or (frame_count > 30 and len(tracks) >= 2)):
            print(f"Saving annotated snapshot at frame {frame_count} to: {output_image_path}")
            # Ensure parent dir exists
            os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
            cv2.imwrite(output_image_path, annotated_frame)
            saved_snapshot = True
            
    end_time = time.time()
    cap.release()
    
    elapsed = end_time - start_time
    processing_fps = frame_count / elapsed
    print("\n--- Verification Summary ---")
    print(f"Frames processed: {frame_count}")
    print(f"Time taken: {elapsed:.2f} seconds")
    print(f"Average processing speed: {processing_fps:.2f} FPS")
    print(f"Total unique vehicles tracked: {len(unique_ids)}")
    print(f"Cumulative Track IDs logged: {sorted(list(unique_ids))}")
    print("Verification completed successfully!")

if __name__ == "__main__":
    run_verification()
