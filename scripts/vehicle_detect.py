import cv2
from ultralytics import YOLO

def main():
    # Load the YOLOv8 model
    model = YOLO('yolov8n.pt')  # You can choose other models like yolov8s.pt, yolov8m.pt, etc.

    # Open the video file
    # For webcam, use `video_path = 0`
    video_path = 'vehicles.mp4' 
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video source {video_path}")
        return

    # Define the counting line (e.g., in the middle of the frame)
    line_y = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * 0.5)
    line_x1 = 0
    line_x2 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    # Initialize a dictionary to store tracked objects' last known positions
    tracked_objects = {}
    # Initialize a set to store IDs of vehicles that have been counted
    counted_ids = set()
    total_vehicles_counted = 0

    # Loop through the video frames
    while cap.isOpened():
        success, frame = cap.read()

        if not success:
            break

        # Run YOLOv8 inference on the frame with tracking
        results = model(frame, tracker='bytetrack.yaml', persist=True)

        # Draw the counting line
        cv2.line(frame, (line_x1, line_y), (line_x2, line_y), (255, 0, 0), 2)

        # Process results
        for r in results:
            # Get bounding box coordinates, confidence, and class
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

                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    # Put label near the bounding box
                    label = f"{class_name}: {confidence}"
                    if track_id is not None:
                        label += f" ID: {track_id}"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # Counting logic
                    if track_id is not None:
                        if track_id not in tracked_objects:
                            # New object, store its current position
                            tracked_objects[track_id] = centroid_y
                        else:
                            last_centroid_y = tracked_objects[track_id]
                            # Check if the object crossed the line
                            if last_centroid_y < line_y and centroid_y >= line_y and track_id not in counted_ids:
                                total_vehicles_counted += 1
                                counted_ids.add(track_id)
                                print(f"Vehicle ID {track_id} counted. Total: {total_vehicles_counted}")
                            elif last_centroid_y > line_y and centroid_y <= line_y and track_id not in counted_ids:
                                # You can add logic for counting vehicles moving in the opposite direction
                                # For now, we'll only count crossing from top to bottom
                                pass
                            
                            # Update the last known position
                            tracked_objects[track_id] = centroid_y
        
        # Display the total count on the frame
        cv2.putText(frame, f"Total Vehicles: {total_vehicles_counted}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the annotated frame
        cv2.imshow("Vehicle Detection and Counting", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
