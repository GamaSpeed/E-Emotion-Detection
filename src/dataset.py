"""
dataset.py — DAiSEE Dataset avec Oversampling des classes rares
Stratégie : weighted oversampling au niveau clip, ratio max configurable.

Run 10 — Augmentations couleur/luminosité ajoutées :
  - Shift Hue (HSV) : simule la variabilité des tons de peau
  - Correction gamma : simule éclairages sombres ou surexposés
  - Augmentation appliquée sur TOUS les clips (pas seulement les oversampled)
  Objectif : réduire le biais lié à la couleur de peau et aux conditions d'éclairage.
"""

import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms
import cv2
from collections import Counter


STATES = ["Boredom", "Engagement", "Confusion", "Frustration"]


# ── Augmentations ──────────────────────────────────────────────────────────────

def _hue_shift(frame, delta_hue):
    """Shift le canal Hue en espace HSV — simule variation de ton de peau."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + delta_hue) % 180
    return cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB)


def _saturation_shift(frame, factor):
    """Modifie la saturation — simule désaturation sous éclairage artificiel."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def _gamma_correction(frame, gamma):
    """Correction gamma — simule éclairages sombres (gamma>1) ou surexposés (gamma<1)."""
    table = (np.arange(256) / 255.0) ** gamma * 255
    return cv2.LUT(frame, table.astype(np.uint8))


def _brightness_contrast(frame, alpha, beta=0):
    """Ajustement luminosité/contraste linéaire."""
    return np.clip(alpha * frame.astype(np.float32) + beta, 0, 255).astype(np.uint8)


def augment_frames(frames, is_oversampled=False):
    """
    Augmentation temporelle appliquée de façon cohérente sur tous les frames du clip.

    Augmentations standard (tous clips) :
      - Flip horizontal (p=0.5)
      - Hue shift (p=0.4) : ±15° → variabilité tons de peau
      - Gamma correction (p=0.4) : 0.5–1.8 → variabilité éclairage
      - Saturation shift (p=0.3) : 0.7–1.3 → éclairage artificiel

    Augmentations renforcées (clips oversampled uniquement) :
      - Brightness/contrast (p=0.5)
      - Contraste renforcé (p=0.3)

    Note : toutes les transformations couleur sont appliquées de façon IDENTIQUE
    sur tous les frames du clip pour préserver la cohérence temporelle.
    """

    # ── Flip horizontal ────────────────────────────────────────────────────────
    if random.random() < 0.5:
        frames = [cv2.flip(f, 1) for f in frames]

    # ── Hue shift — variabilité tons de peau ──────────────────────────────────
    # Appliqué sur tous les clips, pas seulement les oversampled
    if random.random() < 0.4:
        delta_hue = random.uniform(-15, 15)
        frames = [_hue_shift(f, delta_hue) for f in frames]

    # ── Gamma correction — variabilité éclairage ──────────────────────────────
    if random.random() < 0.4:
        gamma = random.uniform(0.5, 1.8)
        frames = [_gamma_correction(f, gamma) for f in frames]

    # ── Saturation — éclairage artificiel ─────────────────────────────────────
    if random.random() < 0.3:
        sat_factor = random.uniform(0.7, 1.3)
        frames = [_saturation_shift(f, sat_factor) for f in frames]

    # ── Augmentations renforcées (clips oversampled uniquement) ───────────────
    if is_oversampled:

        if random.random() < 0.5:
            alpha = random.uniform(0.8, 1.2)
            frames = [_brightness_contrast(f, alpha) for f in frames]

        if random.random() < 0.3:
            alpha = random.uniform(0.85, 1.15)
            frames = [
                np.clip(128 + alpha * (f.astype(float) - 128), 0, 255).astype(np.uint8)
                for f in frames
            ]

    return frames


# ── Dataset ────────────────────────────────────────────────────────────────────

class DAiSEEDataset(Dataset):
    def __init__(
        self,
        csv_path,
        video_dir,
        n_frames    = 8,
        image_size  = 224,
        transform   = None,
        oversample  = True,
        max_ratio   = 4.0,
        augment     = True,
        face_detect = False,
        binary      = False,
    ):
        self.video_dir  = video_dir
        self.n_frames   = n_frames
        self.image_size = image_size
        self.augment    = augment
        self.face_detect= face_detect
        self.binary     = binary

        # Charger CSV
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
        for s in STATES:
            df[s] = df[s].astype(int)

        # Mode binaire : L0+L1 → 0 (Low), L2+L3 → 1 (High)
        if self.binary:
            for s in STATES:
                df[s] = (df[s] >= 2).astype(int)

        # Filtrer clips existants
        valid_rows, valid_paths = [], []
        for _, row in df.iterrows():
            clip_id   = str(row["ClipID"]).strip()
            clip_name = os.path.splitext(clip_id)[0]
            user_id   = clip_name[:6]
            for ext in [".avi", ".mp4"]:
                path = os.path.join(video_dir, user_id, clip_name, clip_name + ext)
                if os.path.exists(path):
                    valid_rows.append(row)
                    valid_paths.append(path)
                    break

        print(f"[Dataset] {len(valid_rows)}/{len(df)} clips valides")
        self.df    = pd.DataFrame(valid_rows).reset_index(drop=True)
        self.paths = valid_paths
        self.oversampled_flags = [False] * len(self.df)

        if oversample:
            self._apply_oversampling(max_ratio)

        # Transform
        self.transform = transform or transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        if self.face_detect:
            self.face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

        self._print_distribution()

    def _apply_oversampling(self, max_ratio):
        df_orig    = self.df.copy()
        paths_orig = self.paths.copy()
        extra_rows, extra_paths, extra_flags = [], [], []

        for state in STATES:
            counts    = Counter(df_orig[state].values)
            max_count = max(counts.values())

            for level, count in counts.items():
                ratio = max_count / max(count, 1)
                if ratio <= max_ratio:
                    continue

                target  = int(max_count / max_ratio)
                n_add   = max(0, target - count)
                rare_idx= df_orig.index[df_orig[state] == level].tolist()

                print(f"[Oversample] {state} L{level}: "
                      f"{count} → {count+n_add}  (ratio {ratio:.0f}x → {max_count/(count+n_add):.1f}x)")

                for _ in range(n_add):
                    i = random.choice(rare_idx)
                    extra_rows.append(df_orig.iloc[i])
                    extra_paths.append(paths_orig[i])
                    extra_flags.append(True)

        if extra_rows:
            self.df    = pd.concat([self.df, pd.DataFrame(extra_rows)], ignore_index=True)
            self.paths = self.paths + extra_paths
            self.oversampled_flags += extra_flags
            print(f"[Oversample] Total : {len(self.df)} clips (+{len(extra_rows)} dupliqués)\n")

    def _print_distribution(self):
        mode = "binaire (Low/High)" if self.binary else "4 niveaux"
        print(f"\n[Distribution après oversampling — mode {mode}]")
        for state in STATES:
            counts = Counter(self.df[state].values)
            total  = sum(counts.values())
            if self.binary:
                dist = (f"Low:{counts.get(0,0)}({counts.get(0,0)/total*100:.0f}%) | "
                        f"High:{counts.get(1,0)}({counts.get(1,0)/total*100:.0f}%)")
            else:
                dist = " | ".join([f"L{k}:{counts.get(k,0)}({counts.get(k,0)/total*100:.0f}%)"
                                   for k in range(4)])
            print(f"  {state:12s}: {dist}")
        print()

    def _load_frames(self, video_path):
        cap    = cv2.VideoCapture(video_path)
        total  = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
        indices= np.linspace(0, total - 1, self.n_frames, dtype=int)
        frames = []

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret and frame is not None:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            elif frames:
                frames.append(frames[-1].copy())
            else:
                frames.append(np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8))

        cap.release()
        while len(frames) < self.n_frames:
            frames.append(frames[-1].copy() if frames else
                          np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8))
        return frames[:self.n_frames]

    def _crop_face(self, frame):
        cascade = cv2.CascadeClassifier(self.face_cascade_path)
        gray    = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        faces   = cascade.detectMultiScale(gray, 1.1, 4, minSize=(60, 60))
        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda f: f[2]*f[3])
            pad = int(0.2 * max(w, h))
            x1, y1 = max(0, x-pad), max(0, y-pad)
            x2, y2 = min(frame.shape[1], x+w+pad), min(frame.shape[0], y+h+pad)
            return frame[y1:y2, x1:x2]
        return frame

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row    = self.df.iloc[idx]
        is_over= self.oversampled_flags[idx]
        frames = self._load_frames(self.paths[idx])

        if self.face_detect:
            frames = [self._crop_face(f) for f in frames]
        if self.augment:
            frames = augment_frames(frames, is_oversampled=is_over)

        tensors = []
        for f in frames:
            try:
                tensors.append(self.transform(f))
            except Exception:
                tensors.append(torch.zeros(3, self.image_size, self.image_size))

        video_tensor = torch.stack(tensors)  # (T, C, H, W)
        labels = {s: torch.tensor(int(row[s]), dtype=torch.long) for s in STATES}
        return video_tensor, labels


# ── WeightedRandomSampler helper ───────────────────────────────────────────────

def make_weighted_sampler(dataset):
    """Sampler pondéré par rareté combinée des 4 états (optionnel)."""
    df = dataset.df
    state_weights = {}
    for state in STATES:
        counts = Counter(df[state].values)
        total  = sum(counts.values())
        state_weights[state] = {lvl: total / max(cnt, 1) for lvl, cnt in counts.items()}

    sample_weights = [
        np.mean([state_weights[s][int(df.iloc[i][s])] for s in STATES])
        for i in range(len(df))
    ]
    return WeightedRandomSampler(
        weights     = torch.tensor(sample_weights, dtype=torch.float),
        num_samples = len(sample_weights),
        replacement = True,
    )