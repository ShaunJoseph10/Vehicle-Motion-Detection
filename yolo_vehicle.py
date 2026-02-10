from ultralytics import YOLO
import cv2

# Load YOLOv8 nano (lightweight ~6MB)
model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # CAP_DSHOW helps on Windows

if not cap.isOpened():
    print("Camera not opening")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.4, verbose=False)

    vehicle_count = 0
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            name = model.names[cls]
            if name in ["car", "truck", "bus", "motorcycle"]:
                vehicle_count += 1

    annotated = results[0].plot()

    cv2.putText(
        annotated,
        f"Vehicles: {vehicle_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("YOLOv8 Vehicle Detection", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
