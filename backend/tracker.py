import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

class ObjectTracker:
    """
    A unified class that combines YOLOv8 (for object detection) and DeepSORT (for object tracking).
    
    RESUME EXPLANATION OF CONCEPTS USED HERE:
    1. YOLOv8: A single-stage deep learning object detector that divides frames into grids 
       and predicts bounding boxes, class labels, and probabilities in a single pass.
    2. DeepSORT (Simple Online and Realtime Tracking with a Deep Association Metric):
       - Kalman Filter: Predicts the object's next position using a linear constant velocity model.
       - Hungarian Algorithm: Solves the bipartite matching problem to pair existing Kalman predicted tracks 
         with new YOLO detections.
       - Deep Embedder (MobileNet): Extracts deep visual features (embeddings) of the detected objects. 
         This allows the tracker to maintain object IDs even when objects are briefly blocked (occluded).
    """
    def __init__(self, model_name="yolov8n.pt", confidence_threshold=0.35, tracker_max_age=25):
        # Initialize YOLOv8 model (auto-detects and uses CUDA if a GPU is available)
        print(f"Loading YOLO model: {model_name}...")
        self.model = YOLO(model_name)
        
        # Get class names dictionary from YOLO
        self.class_names = self.model.names
        
        # Initialize the DeepSORT tracker
        # We use the MobileNet embedder because it is lightweight, runs fast on CPU, and is accurate.
        self.tracker = DeepSort(
            max_age=tracker_max_age, # Delete track if object is not seen for 25 consecutive frames
            n_init=3,                # Number of consecutive detections before a track is 'confirmed'
            nms_max_overlap=1.0,     # Non-max suppression threshold
            max_cosine_distance=0.3, # Threshold for cosine distance matching (deep embeddings)
            embedder="mobilenet",    # Visual appearance feature extractor
            half=True,               # Use FP16 for embedder speedup
            bgr=True                 # OpenCV uses BGR images
        )
        
        self.confidence_threshold = confidence_threshold
        
        # History of track coordinates to draw visual trails/paths
        self.track_history = {} # track_id -> list of (x, y) tuples (centers)
        self.max_history_len = 30 # Maintain last 30 frames of path trail

    def get_available_classes(self):
        """Returns the dictionary of all detectable classes for selection in the frontend."""
        return self.class_names

    def process_frame(self, frame, allowed_classes=None, conf_threshold=None, show_trails=True):
        """
        Processes a single video frame:
        1. Runs YOLOv8 object detection on the frame.
        2. Formats and filters detections for DeepSORT.
        3. Updates DeepSORT tracks (Kalman Filter prediction & Hungarian matching).
        4. Draws bounding boxes, IDs, and path history.
        5. Returns the annotated frame and tracking metadata.
        """
        if conf_threshold is None:
            conf_threshold = self.confidence_threshold

        # Step 1: Run YOLOv8 Object Detection
        # verbose=False reduces console clutter
        results = self.model(frame, verbose=False)[0]
        
        # Format detections for DeepSORT: List of [ [x, y, w, h], confidence, class_id ]
        # DeepSORT expects the bounding box to be in [left, top, width, height] format.
        deepsort_detections = []
        
        for r in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r
            
            # Filter detections by confidence threshold
            if score < conf_threshold:
                continue
                
            class_name = self.class_names[int(class_id)]
            
            # If a class filter is active, skip classes not in the list
            if allowed_classes is not None and class_name not in allowed_classes:
                continue
                
            w = x2 - x1
            h = y2 - y1
            
            # Append detection in format: ( [left, top, w, h], confidence, class_name )
            deepsort_detections.append(([int(x1), int(y1), int(w), int(h)], score, class_name))
            
        # Step 2: Update DeepSORT Tracks
        # tracker.update_tracks pairs new detections with predicted tracks
        tracks = self.tracker.update_tracks(deepsort_detections, frame=frame)
        
        active_counts = {}
        processed_tracks_data = []

        # Step 3: Draw tracks and trails on the frame
        for track in tracks:
            # Skip tracks that are not confirmed (too new) or became inactive
            if not track.is_confirmed():
                continue
                
            track_id = track.track_id
            class_name = track.det_class
            
            # Get bounding box in [left, top, width, height] format
            ltrb = track.to_ltrb() # Convert to [left, top, right, bottom]
            x1, y1, x2, y2 = map(int, ltrb)
            
            # Track statistics counter
            active_counts[class_name] = active_counts.get(class_name, 0) + 1
            
            # Calculate the center point of the bounding box
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            
            # Update path history for trail visualization
            if track_id not in self.track_history:
                self.track_history[track_id] = []
            self.track_history[track_id].append((center_x, center_y))
            
            # Keep history length within bounds
            if len(self.track_history[track_id]) > self.max_history_len:
                self.track_history[track_id].pop(0)
                
            # Draw path trail (only if enabled)
            if show_trails:
                history = self.track_history[track_id]
                for i in range(1, len(history)):
                    thickness = int(np.sqrt(self.max_history_len / float(i + 1)) * 2)
                    # Soft, subtle blue-gray color for trails (BGR: 200, 170, 140)
                    cv2.line(frame, history[i - 1], history[i], (200, 170, 140), thickness)
                
            # Draw Bounding Box (use a soft, pastel color based on the track ID)
            color_b = int((hash(str(track_id)) * 40) % 80) + 160
            color_g = int((hash(str(track_id)) * 60) % 80) + 140
            color_r = int((hash(str(track_id)) * 80) % 80) + 100
            color = (color_b, color_g, color_r)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label with ID and Class Name
            label = f"#{track_id} {class_name}"
            # Draw small background box for label text to make it readable
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - 20), (x1 + label_w, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            
            # Store metadata to send back to frontend
            processed_tracks_data.append({
                "track_id": track_id,
                "class_name": class_name,
                "bbox": [x1, y1, x2, y2]
            })

        # Cleanup track history for deleted tracks to save memory
        current_track_ids = {track.track_id for track in tracks if track.is_confirmed()}
        for deleted_id in list(self.track_history.keys()):
            if deleted_id not in current_track_ids:
                del self.track_history[deleted_id]

        return frame, processed_tracks_data, active_counts
