import cv2
import numpy as np
import base64
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fer.fer import FER

app = FastAPI()

# mtcnn=True ensures high-accuracy face tracking inside the container
detector = FER(mtcnn=True) 

@app.get("/")
async def get_frontend():
    return FileResponse("index.html")

@app.websocket("/ws/emotions")
async def emotion_websocket(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            
            encoded_data = data.split(',')[1]
            nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                continue

            results = detector.detect_emotions(frame)
            
            if not results:
                await websocket.send_json({"status": "no_face", "faces": []})
                continue
                
            faces_data = []
            for face in results:
                top_emotion = max(face["emotions"], key=face["emotions"].get)
                box = [int(x) for x in face["box"]]
                confidence = float(face["emotions"][top_emotion])
                
                faces_data.append({
                    "box": box,
                    "emotion": top_emotion,
                    "confidence": confidence
                })
            await websocket.send_json({"status": "success", "faces": faces_data})
            
    except WebSocketDisconnect:
        print("Client disconnected")