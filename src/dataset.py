"""
dataset.py — DAiSEEDataset : chargement et prétraitement des vidéos
"""

import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms
from PIL import Image


# Transformations

def get_transforms(split: str, image_size: int = 224):
    """Retourne les transformations selon le split (train/val/test)."""
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    if split == "train":
        return transforms.Compose([
            transforms.Resize((image_size + 20, image_size + 20)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])


# Dataset Class

class DAiSEEDataset(Dataset):
    """
    Dataset PyTorch pour DAiSEE.

    Args:
        csv_path    : Chemin vers le fichier CSV des labels.
        video_dir   : Dossier contenant les vidéos.
        n_frames    : Nombre de frames à extraire uniformément par clip.
        split       : 'train', 'val', ou 'test'.
        image_size  : Taille de resize des frames.
        use_face    : Si True, tente de détecter et recadrer le visage (MTCNN).
    """

    LABEL_COLS = ["Boredom", "Engagement", "Confusion", "Frustration"]

    def __init__(self, csv_path, video_dir, n_frames=16, split="train", image_size=224, use_face=True):
        self.df = pd.read_csv(csv_path)          
        self.video_dir = video_dir
        self.n_frames = n_frames
        self.split = split
        self.transform = get_transforms(split, image_size)
        self.use_face = use_face
        self._face_detector = None

        # Nettoyer les labels manquants
        self.df = self.df.dropna(subset=self.LABEL_COLS).reset_index(drop=True)

        # Convertir labels en int (0-3)
        for col in self.LABEL_COLS:
            self.df[col] = self.df[col].astype(int)
            
        # Pré-construire l'index des chemins valides et filtrer les manquants
        print("Vérification des fichiers vidéo...", flush=True)
        valid_indices = []
        for i, row in self.df.iterrows():
            clip_name = os.path.splitext(str(row["ClipID"]).strip())[0]
            user_id = clip_name[:6]
            path_avi = os.path.join(self.video_dir, user_id, clip_name, clip_name + ".avi")
            path_mp4 = os.path.join(self.video_dir, user_id, clip_name, clip_name + ".mp4")
            if os.path.exists(path_avi) or os.path.exists(path_mp4):
                valid_indices.append(i)

        n_removed = len(self.df) - len(valid_indices)
        if n_removed > 0:
            print(f"{n_removed} clips ignorés (fichiers manquants)")
        self.df = self.df.loc[valid_indices].reset_index(drop=True)
        print(f"Dataset prêt : {len(self.df)} clips valides", flush=True)

    # Détecteur de visage (lazy init)
    @property
    def face_detector(self):
        if self._face_detector is None and self.use_face:
            try:
                from facenet_pytorch import MTCNN
                self._face_detector = MTCNN(
                    keep_all=False, post_process=False,
                    device="cuda" if torch.cuda.is_available() else "cpu"
                )
            except ImportError:
                print("[Warning] facenet-pytorch non installé. Désactivation de la détection de visage.")
                self.use_face = False
        return self._face_detector

    # Extraction des frames
    def _extract_frames(self, video_path: str):
        """Extrait n_frames frames uniformément espacées depuis une vidéo."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Impossible d'ouvrir : {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_frames = max(total_frames, 1)
        indices = np.linspace(0, total_frames - 1, self.n_frames, dtype=int)

        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame))
            elif frames:
                frames.append(frames[-1])  # duplique la dernière frame si échec
        cap.release()

        # Compléter si nécessaire
        while len(frames) < self.n_frames:
            frames.append(frames[-1] if frames else Image.new("RGB", (224, 224)))

        return frames[:self.n_frames]

    # Détection et crop du visage
    def _crop_face(self, pil_img: Image.Image) -> Image.Image:
        """Tente de détecter et recadrer le visage. Fallback sur image entière."""
        if not self.use_face or self.face_detector is None:
            return pil_img
        try:
            boxes, _ = self.face_detector.detect(pil_img)
            if boxes is not None and len(boxes) > 0:
                x1, y1, x2, y2 = [int(v) for v in boxes[0]]
                # Ajouter une marge de 20%
                w, h = pil_img.size
                margin_x = int((x2 - x1) * 0.2)
                margin_y = int((y2 - y1) * 0.2)
                x1 = max(0, x1 - margin_x)
                y1 = max(0, y1 - margin_y)
                x2 = min(w, x2 + margin_x)
                y2 = min(h, y2 + margin_y)
                return pil_img.crop((x1, y1, x2, y2))
        except Exception:
            pass
        return pil_img  # fallback

    # __len__ & __getitem__
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # ClipID contient déjà l'extension (ex: "1100011002.avi")
        clip_id = str(row["ClipID"]).strip()

        # Chercher le fichier vidéo en testant plusieurs cas
        clip_id = str(row["ClipID"]).strip()  
        clip_name = os.path.splitext(clip_id)[0] 
        user_id = clip_name[:6]                  

        video_path = os.path.join(self.video_dir, user_id, clip_name, clip_id)

        if not os.path.exists(video_path):
            mp4_id = clip_name + ".mp4"
            video_path = os.path.join(self.video_dir, user_id, clip_name, mp4_id)

        if not os.path.exists(video_path):
            raise FileNotFoundError(
                f"Vidéo introuvable pour ClipID='{clip_id}'.\n"
                f"Chemin tenté : {os.path.join(self.video_dir, user_id, clip_name, clip_id)}"
            )

        # Extraire les frames
        frames = self._extract_frames(video_path)

        # Détection de visage + transformation
        tensor_frames = []
        for frame in frames:
            frame = self._crop_face(frame)
            tensor_frames.append(self.transform(frame))

        # Stack → (T, C, H, W)
        video_tensor = torch.stack(tensor_frames)

        # Labels → (4,) avec valeurs 0-3
        labels = torch.tensor(
            [row[col] for col in self.LABEL_COLS],
            dtype=torch.long
        )

        return video_tensor, labels


# WeightedRandomSampler (déséquilibre de classes)

def get_weighted_sampler(dataset: DAiSEEDataset, target_col: str = "Engagement"):
    """
    Crée un WeightedRandomSampler basé sur une colonne cible.
    Utile pour équilibrer les classes déséquilibrées (Frustration, Boredom).
    """
    col_idx = DAiSEEDataset.LABEL_COLS.index(target_col)
    labels = dataset.df[target_col].values

    class_counts = np.bincount(labels, minlength=4)
    class_weights = 1.0 / (class_counts + 1e-6)
    sample_weights = class_weights[labels]

    return WeightedRandomSampler(
        weights=torch.FloatTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True
    )
