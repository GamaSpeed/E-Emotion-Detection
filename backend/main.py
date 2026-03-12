"""
main.py — API FastAPI pour EduSense
Endpoints :
  GET  /health                  → statut du serveur
  POST /predict/image           → prédiction sur une image base64
  WS   /ws/student/{client_id}  → flux WebSocket webcam locale
  WS   /ws/camera               → flux WebSocket webcam serveur (OpenCV)
  GET  /stream/camera           → MJPEG stream webcam serveur
"""

import asyncio
import base64
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import yaml
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# ── Ajouter le répertoire parent au path pour importer src/ ──────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.model import EmotionModel
from backend.inference import EmotionPredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
CONFIG_PATH     = os.getenv("CONFIG_PATH",     "configs/config.yaml")
CHECKPOINT_PATH = os.getenv("CHECKPOINT_PATH", "outputs/checkpoints/best_model.pt")
DEVICE          = os.getenv("DEVICE",          "auto")


def load_config() -> dict:
    """
    Charge le config YAML.
    Si le checkpoint contient un config embarqué, il a priorité
    (évite les désaccords backbone entre config.yaml et checkpoint).
    """
    # 1. Essayer de lire le config depuis le checkpoint
    try:
        ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
        if isinstance(ckpt, dict) and "config" in ckpt:
            cfg = ckpt["config"]
            logger.info(f"✅ Config lu depuis checkpoint — backbone : {cfg['model']['backbone']}")
            return cfg
    except Exception as e:
        logger.warning(f"⚠️  Impossible de lire le checkpoint pour le config : {e}")

    # 2. Fallback sur config.yaml
    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        logger.info(f"✅ Config lu depuis fichier — backbone : {cfg['model']['backbone']}")
        return cfg
    except Exception as e:
        logger.error(f"❌ Impossible de charger config.yaml : {e}")
        raise


config = load_config()

# ── Initialisation modèle ─────────────────────────────────────────────────────
predictor: Optional[EmotionPredictor] = None


def load_predictor():
    global predictor
    try:
        predictor = EmotionPredictor(
            model_path  = CHECKPOINT_PATH,
            model_class = EmotionModel,
            config      = config,
            device      = DEVICE,
        )
        logger.info("✅ Modèle chargé avec succès")
    except Exception as e:
        logger.error(f"❌ Erreur chargement modèle : {e}")
        import traceback
        traceback.print_exc()
        predictor = None


# ── App FastAPI ───────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "EduSense API",
    description = "API de détection d'émotions pour e-learning",
    version     = "1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


@app.on_event("startup")
async def startup():
    load_predictor()


# ── Schemas ───────────────────────────────────────────────────────────────────
class ImageRequest(BaseModel):
    image: str
    client_id: Optional[str] = None


# ── Endpoints REST ────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status":   "ok",
        "model":    "loaded" if predictor else "not_loaded",
        "backbone": config["model"]["backbone"],
        "device":   str(predictor.device) if predictor else "N/A",
        "version":  "1.0.0",
    }


@app.post("/predict/image")
async def predict_image(req: ImageRequest):
    if predictor is None:
        raise HTTPException(503, "Modèle non disponible")
    try:
        b64         = req.image.split(",")[-1]
        image_bytes = base64.b64decode(b64)
        result      = predictor.predict_from_bytes(image_bytes)
        if result is None:
            raise HTTPException(422, "Impossible de traiter l'image")
        return {
            "status":           "ok",
            "predictions":      result,
            "engagement_score": predictor.get_engagement_score(),
            "timestamp":        time.time(),
        }
    except Exception as e:
        logger.error(f"Erreur /predict/image : {e}")
        raise HTTPException(500, str(e))


# ── WebSocket — webcam locale ─────────────────────────────────────────────────
class ConnectionManager:
    def __init__(self):
        self.active: dict[str, WebSocket] = {}

    async def connect(self, ws: WebSocket, client_id: str):
        await ws.accept()
        self.active[client_id] = ws
        logger.info(f"[WS] Client connecté : {client_id}")

    def disconnect(self, client_id: str):
        self.active.pop(client_id, None)
        logger.info(f"[WS] Client déconnecté : {client_id}")

    async def send(self, client_id: str, data: dict):
        ws = self.active.get(client_id)
        if ws:
            try:
                await ws.send_text(json.dumps(data))
            except Exception:
                self.disconnect(client_id)


manager = ConnectionManager()


@app.websocket("/ws/student/{client_id}")
async def ws_student(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)

    try:
        client_predictor = EmotionPredictor(
            model_path  = CHECKPOINT_PATH,
            model_class = EmotionModel,
            config      = config,
            device      = DEVICE,
        )
    except Exception as e:
        await websocket.send_text(json.dumps({"error": str(e)}))
        await websocket.close()
        return

    try:
        while True:
            data = await websocket.receive_text()
            msg  = json.loads(data)
            if "frame" not in msg:
                continue

            b64         = msg["frame"].split(",")[-1]
            frame_bytes = base64.b64decode(b64)
            result      = client_predictor.predict_from_bytes(frame_bytes)

            if result is None:
                await manager.send(client_id, {"status": "error"})
            elif result.get("status") == "buffering":
                await manager.send(client_id, result)
            else:
                await manager.send(client_id, {
                    "status":           "ok",
                    "predictions":      result,
                    "engagement_score": client_predictor.get_engagement_score(),
                    "timestamp":        time.time(),
                })

    except WebSocketDisconnect:
        manager.disconnect(client_id)


# ── WebSocket — webcam serveur ────────────────────────────────────────────────
@app.websocket("/ws/camera")
async def ws_camera(websocket: WebSocket):
    await websocket.accept()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        await websocket.send_text(json.dumps({"error": "Caméra non disponible"}))
        await websocket.close()
        return

    try:
        client_predictor = EmotionPredictor(
            model_path  = CHECKPOINT_PATH,
            model_class = EmotionModel,
            config      = config,
            device      = DEVICE,
        )

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            ready = client_predictor.add_frame(frame)
            _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            frame_b64 = base64.b64encode(jpeg.tobytes()).decode()

            payload = {"frame": frame_b64, "timestamp": time.time()}

            if ready:
                result = client_predictor.predict()
                if result:
                    payload.update({
                        "status":           "ok",
                        "predictions":      result,
                        "engagement_score": client_predictor.get_engagement_score(),
                    })
            else:
                payload["status"] = "buffering"
                payload["frames"] = len(client_predictor.frame_buffer)
                payload["needed"] = client_predictor.n_frames

            await websocket.send_text(json.dumps(payload))
            await asyncio.sleep(0.1)

    except WebSocketDisconnect:
        logger.info("[WS Camera] Client déconnecté")
    finally:
        cap.release()


# ── MJPEG Stream ──────────────────────────────────────────────────────────────
def generate_mjpeg():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        _, jpeg = cv2.imencode(".jpg", frame)
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" +
               jpeg.tobytes() + b"\r\n")
    cap.release()


@app.get("/stream/camera")
async def stream_camera():
    return StreamingResponse(
        generate_mjpeg(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=False)