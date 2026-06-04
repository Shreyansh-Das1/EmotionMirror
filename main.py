import cv2
import numpy as np
import base64
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fer.fer import FER

app = FastAPI()

# Initialize the detector. mtcnn=False uses Haar Cascades (faster for real-time)
detector = FER(mtcnn=True) 

@app.websocket("/ws/emotions")
async def emotion_websocket(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # 1. Receive the frame from the frontend
            data = await websocket.receive_text()
            
            # 2. Decode the base64 jpeg into an OpenCV format
            encoded_data = data.split(',')[1]
            nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                continue

            # 3. Detect faces and emotions
            # Returns a list: [{'box': [x, y, w, h], 'emotions': {'happy': 0.9, ...}}]
            results = detector.detect_emotions(frame)
            
            if not results:
                # Handle the "No Face" scenario gracefully
                await websocket.send_json({"status": "no_face", "faces": []})
                continue
                
            faces_data = []
            for face in results:
                # Extract the highest scoring emotion
                top_emotion = max(face["emotions"], key=face["emotions"].get)
                
                # FIX: Force bounding box values to standard Python ints
                box = [int(x) for x in face["box"]]
                
                # FIX: Force confidence score to a standard Python float
                confidence = float(face["emotions"][top_emotion])
                
                faces_data.append({
                    "box": box,
                    "emotion": top_emotion,
                    "confidence": confidence
                })
                
            # 4. Send the cleaned data back to the client
            await websocket.send_json({"status": "success", "faces": faces_data})
            
    except WebSocketDisconnect:
        print("Client disconnected")