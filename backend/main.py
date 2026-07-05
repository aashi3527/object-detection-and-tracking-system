import base64
import json
import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from tracker import ObjectTracker

app = FastAPI(title="SentinelTrack AI Backend", description="YOLO + DeepSORT Object Tracking Server")

# Allow CORS so that the React frontend (running on another port) can communicate with this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global tracker instance
# Uses yolov8n.pt by default (lightweight, runs fast on CPU)
tracker = ObjectTracker(model_name="yolov8n.pt")

@app.get("/")
def read_root():
    return {"status": "running", "model": "YOLOv8n + DeepSORT"}

@app.get("/api/classes")
def get_classes():
    """Returns the list of classes YOLO can detect so the frontend can display them."""
    return {"classes": tracker.get_available_classes()}

@app.websocket("/ws/track")
async def websocket_tracking_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time tracking.
    Expects client to stream frames as Base64-encoded JPEG strings, 
    and returns annotated frames and tracking metadata.
    """
    await websocket.accept()
    print("Client connected to tracking WebSocket")
    
    try:
        while True:
            # Receive data from frontend
            # Expects JSON: { "image": "data:image/jpeg;base64,...", "settings": { "conf": 0.35, "classes": [...] } }
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # 1. Parse Image Base64 Data
            image_data = message.get("image")
            if not image_data:
                continue
                
            # Strip metadata prefix if present (e.g., "data:image/jpeg;base64,")
            if "," in image_data:
                header, image_data = image_data.split(",", 1)
                
            try:
                # Convert base64 string back to binary JPEG bytes
                binary_data = base64.b64decode(image_data)
                np_arr = np.frombuffer(binary_data, dtype=np.uint8)
                # Decode image bytes to OpenCV format (BGR)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            except Exception as e:
                print(f"Error decoding frame: {e}")
                await websocket.send_text(json.dumps({"error": "Failed to decode frame"}))
                continue
                
            if frame is None:
                await websocket.send_text(json.dumps({"error": "Decoded frame is null"}))
                continue

            # 2. Parse Tracker Settings
            settings = message.get("settings", {})
            conf_threshold = settings.get("confidence", 0.35)
            allowed_classes = settings.get("classes", None) # None means detect all classes
            show_trails = settings.get("show_trails", True)
            
            # 3. Run Object Detection & DeepSORT Tracking
            # Modifies the frame by drawing bounding boxes and paths
            annotated_frame, tracks_data, active_counts = tracker.process_frame(
                frame, 
                allowed_classes=allowed_classes, 
                conf_threshold=conf_threshold,
                show_trails=show_trails
            )
            
            # 4. Encode Annotated Frame back to Base64
            _, buffer = cv2.imencode('.jpg', annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            encoded_image = base64.b64encode(buffer).decode('utf-8')
            
            # 5. Send results back to React frontend
            response = {
                "image": f"data:image/jpeg;base64,{encoded_image}",
                "tracks": tracks_data,
                "counts": active_counts
            }
            await websocket.send_text(json.dumps(response))
            
    except WebSocketDisconnect:
        print("Client disconnected from tracking WebSocket")
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        try:
            await websocket.close()
        except:
            pass

if __name__ == "__main__":
    import uvicorn
    # Start the server on port 8000
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
