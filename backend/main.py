"""
main.py — API FastAPI EduSense v2
Nouveautés v2 :
  - Base de données SQLAlchemy (SQLite local → PostgreSQL GCP)
  - Sessions et prédictions persistées après chaque inférence
  - Alertes automatiques (confusion/frustration High persistant)
  - WebSocket /ws/teacher → tableau de bord prof en temps réel
  - Authentification JWT (GET /auth/login, GET /users/me)
  - Tous les anciens endpoints conservés et compatibles

Endpoints :
  GET  /health
  POST /predict/image
  POST /auth/login / /auth/register
  GET  /sessions, /sessions/{id}, /sessions/{id}/export
  GET  /alerts/pending, POST /alerts/{id}/ack
  GET  /logs
  WS   /ws/student/{client_id}
  WS   /ws/teacher
  WS   /ws/camera
  GET  /stream/camera
"""

import asyncio
import base64
import json
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import cv2
import torch
import yaml
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session as DBSession

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.model import EmotionModel
from backend.inference import EmotionPredictor

# ── DB imports ────────────────────────────────────────────────────────────────
from backend.db.database import get_db, init_db, SessionLocal
from backend.db import crud, schemas
from backend.db.routes_db import router as db_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
CONFIG_PATH     = os.getenv("CONFIG_PATH",     "configs/config.yaml")
CHECKPOINT_PATH = os.getenv("CHECKPOINT_PATH", "notebooks/outputs/checkpoints/best_model.pt")
DEVICE          = os.getenv("DEVICE",          "auto")

# Nombre de prédictions consécutives High avant de déclencher une alerte
ALERT_THRESHOLD = int(os.getenv("ALERT_THRESHOLD", "3"))


def load_config() -> dict:
    try:
        ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
        if isinstance(ckpt, dict) and "config" in ckpt:
            cfg = ckpt["config"]
            logger.info(f"✅ Config depuis checkpoint — backbone: {cfg['model']['backbone']}")
            return cfg
    except Exception as e:
        logger.warning(f"⚠️  Checkpoint config non lisible : {e}")
    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        logger.info(f"✅ Config depuis fichier — backbone: {cfg['model']['backbone']}")
        return cfg
    except Exception as e:
        logger.error(f"❌ Impossible de charger config.yaml : {e}")
        raise


config = load_config()

# ── Modèle ────────────────────────────────────────────────────────────────────
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
        db = SessionLocal()
        try:
            crud.write_log(db, "model_load", details={"backbone": config["model"]["backbone"]})
        finally:
            db.close()
        logger.info("✅ Modèle chargé")
    except Exception as e:
        logger.error(f"❌ Erreur chargement modèle : {e}")
        predictor = None


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "EduSense API",
    description = "Détection d'émotions e-learning — v2 avec DB et WebSocket prof",
    version     = "2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# Routes DB (auth, sessions, alerts, logs, export CSV)
app.include_router(db_router)


@app.on_event("startup")
async def startup():
    init_db()
    load_predictor()
    logger.info("✅ EduSense v2 démarré — DB initialisée")


# ── Schemas REST ──────────────────────────────────────────────────────────────
class ImageRequest(BaseModel):
    image:     str
    client_id: Optional[str] = None


# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status":   "ok",
        "model":    "loaded" if predictor else "not_loaded",
        "backbone": config["model"]["backbone"],
        "device":   str(predictor.device) if predictor else "N/A",
        "version":  "2.0.0",
        "db":       "connected",
    }


# ── Predict image (REST) ──────────────────────────────────────────────────────
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


# ── ConnectionManager étendu ──────────────────────────────────────────────────
class ConnectionManager:
    """
    Gère les connexions WebSocket étudiants ET professeurs.
    Chaque prédiction étudiant est broadcastée à tous les profs connectés.
    """
    def __init__(self):
        self.students: dict[str, WebSocket] = {}   # client_id → ws
        self.teachers: dict[str, WebSocket] = {}   # teacher_id → ws

    async def connect_student(self, ws: WebSocket, client_id: str):
        await ws.accept()
        self.students[client_id] = ws
        logger.info(f"[WS] Étudiant connecté : {client_id}")

    async def connect_teacher(self, ws: WebSocket, teacher_id: str):
        await ws.accept()
        self.teachers[teacher_id] = ws
        logger.info(f"[WS] Professeur connecté : {teacher_id}")

    def disconnect_student(self, client_id: str):
        self.students.pop(client_id, None)
        logger.info(f"[WS] Étudiant déconnecté : {client_id}")

    def disconnect_teacher(self, teacher_id: str):
        self.teachers.pop(teacher_id, None)
        logger.info(f"[WS] Professeur déconnecté : {teacher_id}")

    async def send_to_student(self, client_id: str, data: dict):
        ws = self.students.get(client_id)
        if ws:
            try:
                await ws.send_text(json.dumps(data, default=str))
            except Exception:
                self.disconnect_student(client_id)

    async def broadcast_to_teachers(self, data: dict):
        """Envoie une mise à jour à tous les profs connectés."""
        dead = []
        for tid, ws in self.teachers.items():
            try:
                await ws.send_text(json.dumps(data, default=str))
            except Exception:
                dead.append(tid)
        for tid in dead:
            self.disconnect_teacher(tid)

    @property
    def online_student_ids(self) -> list:
        return list(self.students.keys())


manager = ConnectionManager()

# Compteurs d'états critiques par client_id pour les alertes
# {client_id: {state: count_consecutive_high}}
_alert_counters: dict = defaultdict(lambda: defaultdict(int))


def _check_and_trigger_alerts(
    db: DBSession,
    client_id: str,
    session_id: str,
    predictions: dict,
) -> list[dict]:
    """
    Vérifie si un état critique persiste.
    Déclenche une alerte après ALERT_THRESHOLD prédictions consécutives High.
    Retourne la liste des nouvelles alertes créées.
    """
    CRITICAL_STATES = ["confusion", "frustration"]
    new_alerts = []

    for state in CRITICAL_STATES:
        key = state.capitalize()
        level = predictions.get(key, {}).get("level", 0)

        if level == 1:  # High
            _alert_counters[client_id][state] += 1
            if _alert_counters[client_id][state] == ALERT_THRESHOLD:
                # Déclencher l'alerte
                duration_s = ALERT_THRESHOLD * 0.5  # ~500ms par prédiction
                alert = crud.create_alert(db, session_id, state, duration_s)
                new_alerts.append({
                    "id":       alert.id,
                    "state":    state,
                    "duration": duration_s,
                })
                logger.info(f"[Alert] {state} persistant → étudiant {client_id}")
        else:
            # Réinitialiser le compteur si l'état repasse à Low
            _alert_counters[client_id][state] = 0

    return new_alerts


# ── WebSocket /ws/student/{client_id} ────────────────────────────────────────
@app.websocket("/ws/student/{client_id}")
async def ws_student(websocket: WebSocket, client_id: str):
    await manager.connect_student(websocket, client_id)

    # Charger un predictor dédié à cet étudiant
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

    # Ouvrir une session en DB
    db = SessionLocal()
    session_obj = None
    scores_history = []

    try:
        # Résoudre l'étudiant depuis client_id (ou créer une session anonyme)
        crud.write_log(db, "ws_connect", details={"client_id": client_id})

        # Résoudre le nom de l'étudiant depuis la DB
        # Le client_id peut être soit l'UUID utilisateur, soit un timestamp student_XXX
        student_name = None
        student_user = None
        try:
            # Essayer de trouver l'utilisateur par son ID (si client_id = UUID)
            student_user = crud.get_user_by_id(db, client_id)
            if student_user:
                student_name = student_user.name
                session_obj = crud.create_session(db, student_id=student_user.id, client_id=client_id)
            else:
                session_obj = crud.create_session(db, student_id=None, client_id=client_id)
        except Exception:
            session_obj = crud.create_session(db, student_id=None, client_id=client_id)

        # Notifier les profs qu'un étudiant vient de se connecter
        await manager.broadcast_to_teachers({
            "type":         "student_connected",
            "student_name": student_name,
            "client_id":  client_id,
            "session_id": session_obj.id,
            "timestamp":  time.time(),
        })

        while True:
            data = await websocket.receive_text()
            msg  = json.loads(data)
            if msg.get("type") == "ping":
                continue  # ping client — maintient la connexion active
            if "frame" not in msg:
                continue

            b64         = msg["frame"].split(",")[-1]
            frame_bytes = base64.b64decode(b64)
            result      = client_predictor.predict_from_bytes(frame_bytes)

            if result is None:
                await manager.send_to_student(client_id, {"status": "error"})

            elif result.get("status") == "buffering":
                await manager.send_to_student(client_id, result)

            else:
                eng_score = client_predictor.get_engagement_score()
                scores_history.append(eng_score)

                # ── Persister la prédiction en DB ─────────────────────────
                try:
                    pred_data = schemas.PredictionCreate(
                        session_id       = session_obj.id,
                        engagement       = result["Engagement"]["level"],
                        boredom          = result["Boredom"]["level"],
                        confusion        = result["Confusion"]["level"],
                        frustration      = result["Frustration"]["level"],
                        engagement_conf  = result["Engagement"]["confidence"],
                        boredom_conf     = result["Boredom"]["confidence"],
                        confusion_conf   = result["Confusion"]["confidence"],
                        frustration_conf = result["Frustration"]["confidence"],
                        engagement_score = eng_score,
                    )
                    crud.save_prediction(db, pred_data)

                    # Vérifier les alertes
                    new_alerts = _check_and_trigger_alerts(
                        db, client_id, session_obj.id, result
                    )
                except Exception as db_err:
                    logger.warning(f"[DB] Erreur persistance : {db_err}")
                    new_alerts = []

                # ── Payload vers l'étudiant ────────────────────────────────
                payload = {
                    "status":           "ok",
                    "predictions":      result,
                    "engagement_score": eng_score,
                    "timestamp":        time.time(),
                }
                await manager.send_to_student(client_id, payload)

                # ── Broadcast vers les profs ───────────────────────────────
                await manager.broadcast_to_teachers({
                    "type":             "prediction",
                    "client_id":        client_id,
                    "session_id":       session_obj.id,
                    "predictions":      result,
                    "engagement_score": eng_score,
                    "timestamp":        time.time(),
                    "new_alerts":       new_alerts,
                })

    except WebSocketDisconnect:
        pass
    finally:
        manager.disconnect_student(client_id)
        _alert_counters.pop(client_id, None)

        # Fermer la session en DB
        if session_obj:
            avg_score = (sum(scores_history) / len(scores_history)
                         if scores_history else None)
            crud.close_session(db, session_obj.id, avg_score)
            crud.write_log(db, "ws_disconnect", details={
                "client_id": client_id,
                "n_predictions": len(scores_history),
                "avg_score": round(avg_score, 1) if avg_score else None,
            })

        # Notifier les profs de la déconnexion
        await manager.broadcast_to_teachers({
            "type":      "student_disconnected",
            "client_id": client_id,
            "timestamp": time.time(),
        })
        db.close()


# ── WebSocket /ws/teacher ─────────────────────────────────────────────────────
@app.websocket("/ws/teacher")
async def ws_teacher(websocket: WebSocket):
    """
    Connexion WebSocket du professeur.
    - Reçoit en temps réel toutes les prédictions étudiants
    - Reçoit les connexions / déconnexions étudiants
    - Reçoit les nouvelles alertes
    - Envoie l'état courant de la classe à la connexion
    """
    teacher_id = f"teacher_{int(time.time())}"
    await manager.connect_teacher(websocket, teacher_id)

    db = SessionLocal()
    try:
        crud.write_log(db, "ws_connect", details={"role": "teacher", "teacher_id": teacher_id})

        # Envoyer l'état initial : étudiants actuellement connectés
        await websocket.send_text(json.dumps({
            "type":       "class_state",
            "online":     manager.online_student_ids,
            "timestamp":  time.time(),
        }))

        # Rester connecté — les données arrivent via broadcast_to_teachers
        # Le prof peut envoyer des messages (ex: acquitter une alerte)
        while True:
            try:
                raw = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                msg = json.loads(raw)

                # Acquittement d'alerte depuis le frontend prof
                if msg.get("type") == "ack_alert":
                    alert_id = msg.get("alert_id")
                    if alert_id:
                        crud.acknowledge_alert(db, alert_id)
                        await websocket.send_text(json.dumps({
                            "type":     "alert_acked",
                            "alert_id": alert_id,
                        }))

            except asyncio.TimeoutError:
                # Ping pour maintenir la connexion vivante
                await websocket.send_text(json.dumps({"type": "ping"}))

    except WebSocketDisconnect:
        pass
    finally:
        manager.disconnect_teacher(teacher_id)
        crud.write_log(db, "ws_disconnect", details={"role": "teacher"})
        db.close()


# ── WebSocket /ws/camera (webcam serveur — inchangé) ─────────────────────────
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
        logger.info("[WS Camera] Déconnecté")
    finally:
        cap.release()


# ── MJPEG Stream (inchangé) ───────────────────────────────────────────────────
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