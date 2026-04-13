"""
inference.py — Chargement du modèle DAiSEE et prédiction sur frames vidéo.
Run 6/7 : mode binaire — 2 classes (Low=0, High=1)
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from collections import deque
import logging

logger = logging.getLogger(__name__)

STATES  = ["Boredom", "Engagement", "Confusion", "Frustration"]
LEVELS  = ["Faible", "Élevé"]      # binaire : 0=Low, 1=High
EMOJIS  = {"Boredom": "😴", "Engagement": "🎯", "Confusion": "🤔", "Frustration": "😤"}
COLORS  = {"Boredom": "#6B7FD4", "Engagement": "#22D3A5",
           "Confusion": "#F59E3F", "Frustration": "#EF4B6C"}


def get_inference_transforms(image_size: int = 224):
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


class EmotionPredictor:
    """
    Prédit les 4 états affectifs (mode binaire Low/High) à partir d'un flux de frames.
    Accumule n_frames avant de faire une prédiction.
    """

    def __init__(self, model_path, model_class, config, device="auto"):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info(f"[Inference] Device : {self.device}")

        self.n_frames   = config["dataset"]["n_frames"]
        self.image_size = config["dataset"]["image_size"]
        self.n_classes  = config["model"].get("n_classes", 2)  # 2 en mode binaire
        self.transform  = get_inference_transforms(self.image_size)
        self.frame_buffer = deque(maxlen=self.n_frames)

        # Charger le modèle
        cfg_m = config["model"]
        self.model = model_class(
            n_classes   = self.n_classes,
            hidden_size = cfg_m.get("hidden_size", 256),
            gru_layers  = cfg_m.get("gru_layers", 2),
            gru_dropout = cfg_m.get("gru_dropout", 0.3),
            n_heads     = cfg_m.get("n_attention_heads", 8),
            dropout_clf = cfg_m.get("dropout_classifier", 0.4),
        ).to(self.device)

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        state_dict = checkpoint.get("model_state", checkpoint)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        logger.info(f"[Inference] Modèle chargé — {self.n_classes} classes")

        # Détecteur de visage (optionnel)
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        self.prediction_history = deque(maxlen=5)
        self.last_prediction = None

    def preprocess_frame(self, frame: np.ndarray):
        if frame is None or frame.size == 0:
            return None
        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(60, 60))
        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            pad = int(0.2 * max(w, h))
            x1, y1 = max(0, x - pad), max(0, y - pad)
            x2, y2 = min(rgb.shape[1], x + w + pad), min(rgb.shape[0], y + h + pad)
            rgb = rgb[y1:y2, x1:x2]
        try:
            return self.transform(rgb)
        except Exception as e:
            logger.warning(f"[Inference] Erreur prétraitement : {e}")
            return None

    def add_frame(self, frame: np.ndarray) -> bool:
        t = self.preprocess_frame(frame)
        if t is not None:
            self.frame_buffer.append(t)
        return len(self.frame_buffer) == self.n_frames

    def predict(self) -> dict | None:
        if len(self.frame_buffer) < self.n_frames:
            return None

        frames = torch.stack(list(self.frame_buffer)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(frames)

        result = {}
        for state in STATES:
            logits = outputs[state][0]
            probs  = F.softmax(logits, dim=0).cpu().numpy()
            level  = int(probs.argmax())   # 0=Low, 1=High
            result[state] = {
                "level":       level,
                "level_label": LEVELS[level],
                "confidence":  float(probs.max()),
                "probs":       probs.tolist(),
                "emoji":       EMOJIS[state],
                "color":       COLORS[state],
            }

        self.prediction_history.append(result)
        smoothed = self._smooth_predictions()
        self.last_prediction = smoothed
        return smoothed

    def _smooth_predictions(self) -> dict:
        if len(self.prediction_history) == 1:
            return self.prediction_history[-1]
        smoothed = {}
        for state in STATES:
            avg_probs = np.mean([p[state]["probs"] for p in self.prediction_history], axis=0)
            level = int(avg_probs.argmax())
            smoothed[state] = {
                "level":       level,
                "level_label": LEVELS[level],
                "confidence":  float(avg_probs.max()),
                "probs":       avg_probs.tolist(),
                "emoji":       EMOJIS[state],
                "color":       COLORS[state],
            }
        return smoothed

    def predict_from_bytes(self, image_bytes: bytes) -> dict | None:
        """Entrée : bytes JPEG/PNG depuis le navigateur (WebSocket)."""
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return None
        ready = self.add_frame(frame)
        if ready:
            return self.predict()
        return {
            "status": "buffering",
            "frames": len(self.frame_buffer),
            "needed": self.n_frames,
        }

    def get_engagement_score(self) -> int:
        """
        Score d'engagement 0-100.
        Mode binaire : Low=20, High=80 + pondération confiance.
        """
        if self.last_prediction is None:
            return 0
        pred  = self.last_prediction["Engagement"]
        level = pred["level"]       # 0 ou 1
        conf  = pred["confidence"]  # 0.5 → 1.0
        if level == 1:
            return int(50 + conf * 50)   # 50-100
        else:
            return int((1 - conf) * 50)  # 0-50
