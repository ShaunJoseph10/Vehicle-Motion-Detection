from flask import Flask, jsonify, Response
import cv2
import os

app = Flask(__name__)

# Load Haar Cascade
cascade_path = os.path.join(os.getcwd(), "cars.xml")
car_cascade = cv2.CascadeClassifier(cascade_path)

if car_cascade.empty():
    print("❌ ERROR: cars.xml not loaded")
else:
    print("✅ cars.xml loaded successfully")

# Open Webcam
cap = cv2.VideoCapture(0)

vehicle_count = 0


def generate_frames():
    global vehicle_count

    while True:
        success, frame = cap.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        cars = car_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=2,
            minSize=(60, 60)
        )

        vehicle_count = len(cars)

        for (x, y, w, h) in cars:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/count')
def count():
    return jsonify({"vehicle_count": vehicle_count})


if __name__ == "__main__":
    app.run(debug=True)
