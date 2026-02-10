import React, { useRef, useEffect, useState } from 'react';
import './App.css';

const WEBSOCKET_URL = "ws://localhost:8000/ws"; // Replace with your backend WebSocket URL

function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [mediaStream, setMediaStream] = useState(null);
  const [totalVehicles, setTotalVehicles] = useState(0);
  const ws = useRef(null); // WebSocket reference

  const drawDetections = (context, detections, lineY, frameWidth, frameHeight) => {
    context.clearRect(0, 0, frameWidth, frameHeight); // Clear previous detections
    
    // Draw counting line
    context.beginPath();
    context.moveTo(0, lineY);
    context.lineTo(frameWidth, lineY);
    context.strokeStyle = 'red';
    context.lineWidth = 2;
    context.stroke();

    detections.forEach(detection => {
      const [x1, y1, x2, y2] = detection.box;
      const class_name = detection.class_name;
      const confidence = detection.confidence;
      const track_id = detection.track_id;

      // Draw bounding box
      context.beginPath();
      context.rect(x1, y1, x2 - x1, y2 - y1);
      context.lineWidth = 2;
      context.strokeStyle = 'lime';
      context.stroke();

      // Draw label
      context.fillStyle = 'lime';
      context.font = '12px Arial';
      let label = `${class_name} ${Math.round(confidence * 100)}%`;
      if (track_id !== null) {
        label += ` ID: ${track_id}`;
      }
      context.fillText(label, x1, y1 > 10 ? y1 - 5 : 10);
    });
  };

  const startStream = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
      setMediaStream(stream);
      setIsStreaming(true);

      // Establish WebSocket connection
      ws.current = new WebSocket(`${WEBSOCKET_URL}?client_id=${Date.now()}`); // Unique client_id

      ws.current.onopen = () => {
        console.log("WebSocket connection established");
        const videoElement = videoRef.current;
        const canvasElement = canvasRef.current;
        const context = canvasElement.getContext('2d');

        const sendFrame = () => {
          if (videoElement && videoElement.readyState === videoElement.HAVE_ENOUGH_DATA && ws.current && ws.current.readyState === WebSocket.OPEN) {
            context.drawImage(videoElement, 0, 0, canvasElement.width, canvasElement.height);
            const imageData = canvasElement.toDataURL('image/jpeg', 0.5); // 0.5 quality for faster transmission
            ws.current.send(imageData);
          }
          if (isStreaming) {
            requestAnimationFrame(sendFrame); // Use requestAnimationFrame for smoother animation
          }
        };
        requestAnimationFrame(sendFrame);
      };

      ws.current.onmessage = (event) => {
        const data = JSON.parse(event.data);
        const { detections, total_vehicles, line_y, frame_width, frame_height } = data;
        setTotalVehicles(total_vehicles);

        const canvasElement = canvasRef.current;
        const context = canvasElement.getContext('2d');
        // Ensure canvas dimensions match frame
        if (canvasElement.width !== frameWidth || canvasElement.height !== frameHeight) {
          canvasElement.width = frameWidth;
          canvasElement.height = frameHeight;
          videoRef.current.width = frameWidth;
          videoRef.current.height = frameHeight;
        }
        drawDetections(context, detections, line_y, frameWidth, frameHeight);
      };

      ws.current.onclose = () => {
        console.log("WebSocket connection closed");
        setIsStreaming(false); // Stop streaming when WS closes
      };

      ws.current.onerror = (error) => {
        console.error("WebSocket error: ", error);
        setIsStreaming(false); // Stop streaming on error
      };

    } catch (err) {
      console.error("Error accessing webcam: ", err);
      setIsStreaming(false);
    }
  };

  const stopStream = () => {
    if (mediaStream) {
      mediaStream.getTracks().forEach(track => track.stop());
      if (videoRef.current) {
        videoRef.current.srcObject = null;
      }
      setMediaStream(null);
    }
    if (ws.current) {
      ws.current.close();
    }
    setIsStreaming(false);
    setTotalVehicles(0);
    // Clear canvas
    const canvasElement = canvasRef.current;
    if (canvasElement) {
      const context = canvasElement.getContext('2d');
      context.clearRect(0, 0, canvasElement.width, canvasElement.height);
    }
  };

  useEffect(() => {
    // Clean up stream when component unmounts
    return () => {
      stopStream();
    };
  }, []);

  return (
    <div className="App">
      <header className="App-header">
        <h1>Vehicle Detection and Counting</h1>
      </header>
      <main className="App-main">
        <div className="video-container">
          <video id="videoFeed" ref={videoRef} autoPlay muted playsInline></video>
          <canvas id="detectionCanvas" ref={canvasRef}></canvas>
        </div>
        <div className="controls-container">
          {!isStreaming ? (
            <button onClick={startStream}>Start Webcam</button>
          ) : (
            <button onClick={stopStream}>Stop Webcam</button>
          )}
          <p>Status: {isStreaming ? "Streaming..." : "Stopped"}</p>
          <p>Total Vehicles: {totalVehicles}</p>
          <p className="backend-note">Ensure your FastAPI backend is running on ws://localhost:8000/ws</p>
        </div>
      </main>
    </div>
  );
}

export default App;
