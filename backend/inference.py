"""
inference.py — Chargement du modèle DAiSEE et prédiction sur frames vidéo.
Compatible avec le modèle EfficientNet-B2 + GRU + Attention entraîné.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from collections import deque
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

STATES  = ["Boredom", "Engagement", "Confusion", "Frustration"]
LEVELS  = ["Très faible", "Faible", "Élevé", "Très élevé"]
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
    Prédit les 4 états affectifs à partir d'un flux de frames vidéo.
    Accumule n_frames frames avant de faire une prédiction.
    """

    def __init__(
        self,
        model_path: str,
        model_class,           # classe EmotionModel importée depuis src/model.py
        config: dict,
        device: str = "auto",
    ):
        # Device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info(f"[Inference] Device : {self.device}")

        # Config
        self.n_frames   = config["dataset"]["n_frames"]
        self.image_size = config["dataset"]["image_size"]
        self.transform  = get_inference_transforms(self.image_size)

        # Buffer de frames (FIFO)
        self.frame_buffer = deque(maxlen=self.n_frames)

        # Charger le modèle
        cfg_m = config["model"]
        self.model = model_class(
            n_classes    = cfg_m.get("n_classes", 4),
            hidden_size  = cfg_m.get("hidden_size", 256),
            gru_layers   = cfg_m.get("gru_layers", 2),
            gru_dropout  = cfg_m.get("gru_dropout", 0.3),
            n_heads      = cfg_m.get("n_attention_heads", 8),
            dropout_clf  = cfg_m.get("dropout_classifier", 0.4),
        ).to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)

        # Compatibilité : checkpoint peut contenir 'model_state' ou être le state_dict direct
        state_dict = checkpoint.get("model_state", checkpoint)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        logger.info(f"[Inference] Modèle chargé depuis {model_path}")

        # Détecteur de visage (optionnel)
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        # Historique des prédictions (pour moyennage temporel)
        self.prediction_history = deque(maxlen=5)
        self.last_prediction = None

    def preprocess_frame(self, frame: np.ndarray) -> torch.Tensor | None:
        """
        Prétraite une frame BGR (OpenCV) → tensor normalisé.
        Tente de détecter et recadrer le visage si possible.
        """
        if frame is None or frame.size == 0:
            return None

        # Conversion BGR → RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Détection visage (optionnelle)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(60, 60))
        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])  # plus grand visage
            pad = int(0.2 * max(w, h))
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(rgb.shape[1], x + w + pad)
            y2 = min(rgb.shape[0], y + h + pad)
            rgb = rgb[y1:y2, x1:x2]

        try:
            return self.transform(rgb)
        except Exception as e:
            logger.warning(f"[Inference] Erreur prétraitement : {e}")
            return None

    def add_frame(self, frame: np.ndarray) -> bool:
        """
        Ajoute une frame au buffer.
        Retourne True si le buffer est plein et prêt pour une prédiction.
        """
        tensor = self.preprocess_frame(frame)
        if tensor is not None:
            self.frame_buffer.append(tensor)
        return len(self.frame_buffer) == self.n_frames

    def predict(self) -> dict | None:
        """
        Lance une inférence sur les n_frames du buffer.
        Retourne un dict avec les prédictions par état.
        """
        if len(self.frame_buffer) < self.n_frames:
            return None

        # Stack frames : (n_frames, C, H, W) → (1, n_frames, C, H, W)
        frames = torch.stack(list(self.frame_buffer)).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(frames)  # dict {state: logits (1,4)}

        result = {}
        for state in STATES:
            logits = outputs[state][0]           # (4,)
            probs  = F.softmax(logits, dim=0).cpu().numpy()
            level  = int(probs.argmax())
            result[state] = {
                "level":       level,
                "level_label": LEVELS[level],
                "confidence":  float(probs.max()),
                "probs":       probs.tolist(),
                "emoji":       EMOJIS[state],
                "color":       COLORS[state],
            }

        # Moyennage temporel sur les 5 dernières prédictions
        self.prediction_history.append(result)
        smoothed = self._smooth_predictions()

        self.last_prediction = smoothed
        return smoothed

    def _smooth_predictions(self) -> dict:
        """Moyenne les probabilités sur l'historique pour réduire le bruit."""
        if len(self.prediction_history) == 1:
            return self.prediction_history[-1]

        smoothed = {}
        for state in STATES:
            avg_probs = np.mean([
                p[state]["probs"] for p in self.prediction_history
            ], axis=0)
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
        """
        Entrée : image encodée en bytes (JPEG/PNG depuis le navigateur).
        Utilisé pour le mode webcam locale (WebSocket).
        """
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return None
        ready = self.add_frame(frame)
        if ready:
            return self.predict()
        return {"status": "buffering", "frames": len(self.frame_buffer),
                "needed": self.n_frames}

    def get_engagement_score(self) -> int:
        """Retourne le score d'engagement de 0 à 100."""
        if self.last_prediction is None:
            return 0
        level = self.last_prediction["Engagement"]["level"]
        return int((level / 3) * 100)