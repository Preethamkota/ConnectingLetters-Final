from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import json
import torch
import torch.nn as nn

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from f_model.extract_landmarks_runtime import get_landmarks
from gaze.gaze import get_gaze
from cnn_final.model import build_model   # make sure path is correct

# ------------------ MODEL LOAD ------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = Path(__file__).resolve().parent

model = build_model(device)
model_path = BASE_DIR / "cnn_final" / "best_phase2.pt"
checkpoint = torch.load(model_path, map_location=device)
state_dict = checkpoint.get("model_state_dict", checkpoint)
model.load_state_dict(state_dict)
model.eval()


# ------------------ PREPROCESS ------------------
def preprocess_image(frame):
    image = cv2.resize(frame, (224, 224))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = image / 255.0

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    image = (image - mean) / std

    image = np.transpose(image, (2, 0, 1))  # HWC → CHW
    image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)

    return image


# ------------------ API ------------------
@app.post("/analyze")
async def analyze(image: UploadFile = File(...), metrics: str = Form("{}")):
    try:
        metrics_data = json.loads(metrics)

        contents = await image.read()
        np_arr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return {"error": "invalid image"}

        # -------- LANDMARKS --------
        landmarks = get_landmarks(frame)

        if landmarks is None:
            return {
                "focused": 0,
                "emotion": None,
                "gaze_ratio": None,
                "yaw": None,
                "pitch": None,
                "metrics": metrics_data,
                "error": "face not detected"
            }

        # -------- GAZE --------
        gaze_data = get_gaze(landmarks, frame.shape)

        # -------- EMOTION (CNN) --------
        input_tensor = preprocess_image(frame).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            emotion = int(torch.argmax(outputs, dim=1).item())

        # The trained model has 4 output classes. Map them to the report labels used by the game.
        emotion_map = ["happy", "confused", "frustrated", "neutral"]
        emotion_label = emotion_map[emotion]

        return {
            "focused": gaze_data["focused"],
            "gaze_ratio": gaze_data["gaze_ratio"],
            "yaw": gaze_data["yaw"],
            "pitch": gaze_data["pitch"],
            "emotion": emotion,
            "emotion_label": emotion_label,
            "metrics": metrics_data
        }

    except Exception as e:
        return {
            "error": str(e)
        }


@app.get("/health")
def health():
    return {"status": "ok", "device": str(device)}
