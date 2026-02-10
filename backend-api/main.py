import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import json
import asyncio

app = FastAPI()

# Configure CORS to allow requests from the frontend
origins = [
    "http://localhost",
    "http://localhost:3000",  # React development server
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the YOLOv8 model globally once
model = YOLO('yolov8n.pt')

# In-memory storage for tracking and counting state per WebSocket connection
# In a real-world application, consider more robust state management
class ConnectionState:
    def __init__(self):
        self.tracked_objects = {}
        self.counted_ids = set()
        self.total_vehicles_counted = 0
        self.frame_width = 0
        self.frame_height = 0
        self.line_y = 0

connection_states = {} # Store states per client_id if multiple clients are expected

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Vehicle Detection Backend API!"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await websocket.accept()
    print(f"Client {client_id} connected")
    
    # Initialize state for this client
    state = ConnectionState()
    connection_states[client_id] = state

    try:
        while True:
            data = await websocket.receive_text()
            # Expecting base64 encoded image data
            
            # Decode base64 image data
            # Example: data:image/jpeg;base64,...
            header, encoded = data.split(",", 1)
            image_bytes = base64.b64decode(encoded)
            
            # Convert bytes to numpy array
            nparr = np.frombuffer(image_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                print("Failed to decode frame")
                continue
            
            # Update frame dimensions and counting line if not set
            if state.frame_width == 0:
                state.frame_height, state.frame_width, _ = frame.shape
                state.line_y = int(state.frame_height * 0.5)

            # Run YOLOv8 inference on the frame with tracking
            # We're using stream=True here, but persist=True is crucial for continuous tracking
            results = model(frame, tracker='bytetrack.yaml', persist=True)
            
            detections = []
            
            # Process results
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = round(float(box.conf[0]), 2)
                    cls = int(box.cls[0])
                    class_name = model.names[cls]
                    track_id = int(box.id[0]) if box.id is not None else None

                    # Only consider 'car', 'truck', 'bus', 'motorcycle' for counting
                    if class_name in ['car', 'truck', 'bus', 'motorcycle']:
                        # Calculate centroid of the bounding box
                        centroid_x = (x1 + x2) // 2
                        centroid_y = (y1 + y2) // 2

                        # Counting logic
                        if track_id is not None:
                            if track_id not in state.tracked_objects:
                                # New object, store its current position
                                state.tracked_objects[track_id] = centroid_y
                            else:
                                last_centroid_y = state.tracked_objects[track_id]
                                # Check if the object crossed the line (top to bottom)
                                if last_centroid_y < state.line_y and centroid_y >= state.line_y and track_id not in state.counted_ids:
                                    state.total_vehicles_counted += 1
                                    state.counted_ids.add(track_id)
                                    print(f"Client {client_id} - Vehicle ID {track_id} counted. Total: {state.total_vehicles_counted}")
                                # Update the last known position
                                state.tracked_objects[track_id] = centroid_y
                        
                        detections.append({
                            "box": [x1, y1, x2, y2],
                            "confidence": confidence,
                            "class_name": class_name,
                            "track_id": track_id
                        })
            
            # Send detection results and total count back to the frontend
            response_data = {
                "detections": detections,
                "total_vehicles": state.total_vehicles_counted,
                "line_y": state.line_y,
                "frame_width": state.frame_width,
                "frame_height": state.frame_height
            }
            await websocket.send_text(json.dumps(response_data))

    except WebSocketDisconnect:
        print(f"Client {client_id} disconnected")
    except Exception as e:
        print(f"Error in WebSocket for client {client_id}: {e}")
    finally:
        # Clean up state for the disconnected client
        if client_id in connection_states:
            del connection_states[client_id]
