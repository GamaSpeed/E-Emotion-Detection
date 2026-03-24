"""
dataset.py — DAiSEE Dataset
Run 6 : mode binaire (binary=True)
  - L0+L1 -> classe 0 (Low)
  - L2+L3 -> classe 1 (High)
  - n_classes passe de 4 a 2 dans config.yaml et model
  - oversample=False (class_weights gerent le desequilibre)
  - Augmentation renforcee sur tous les clips
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


def augment_frames(frames):
    if random.random() < 0.5:
        frames = [cv2.flip(f, 1) for f in frames]
    if random.random() < 0.5:
        brightness = random.uniform(0.7, 1.3)
        contrast   = random.uniform(0.8, 1.2)
        frames = [
            np.clip(128 + contrast * (f.astype(float) * brightness - 128), 0, 255).astype(np.uint8)
            for f in frames
        ]
    if random.random() < 0.1:
        frames = [
            cv2.cvtColor(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
            for f in frames
        ]
    return frames


def random_erasing_frame(tensor):
    if random.random() > 0.3:
        return tensor
    _, H, W = tensor.shape
    erase_area = random.uniform(0.02, 0.2) * H * W
    aspect     = random.uniform(0.3, 3.3)
    h = int(round((erase_area * aspect) ** 0.5))
    w = int(round((erase_area / aspect) ** 0.5))
    h, w = min(h, H - 1), min(w, W - 1)
    if h <= 0 or w <= 0:
        return tensor
    y = random.randint(0, H - h)
    x = random.randint(0, W - w)
    tensor[:, y:y+h, x:x+w] = torch.randn(tensor.shape[0], h, w) * 0.1
    return tensor


class DAiSEEDataset(Dataset):
    def __init__(
        self,
        csv_path,
        video_dir,
        n_frames    = 16,
        image_size  = 224,
        transform   = None,
        oversample  = False,
        augment     = True,
        face_detect = False,
        binary      = True,
    ):
        self.video_dir   = video_dir
        self.n_frames    = n_frames
        self.image_size  = image_size
        self.augment     = augment
        self.face_detect = face_detect
        self.binary      = binary

        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
        for s in STATES:
            df[s] = df[s].astype(int)

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

        print(f"[Dataset] {len(valid_rows)}/{len(df)} clips valides | binary={binary}")
        self.df    = pd.DataFrame(valid_rows).reset_index(drop=True)
        self.paths = valid_paths

        self.transform = transform or transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        if self.face_detect:
            self.face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

        self._print_distribution()

    def _print_distribution(self):
        mode = 'binaire' if self.binary else '4-classes'
        print(f"\n[Distribution {mode}]")
        for state in STATES:
            if self.binary:
                vals   = (self.df[state].values >= 2).astype(int)
                counts = np.bincount(vals, minlength=2)
                total  = counts.sum()
                print(f"  {state:12s}: Low={counts[0]}({counts[0]/total*100:.0f}%)  High={counts[1]}({counts[1]/total*100:.0f}%)")
            else:
                counts = Counter(self.df[state].values)
                total  = sum(counts.values())
                dist   = " | ".join([f"L{k}:{counts.get(k,0)}({counts.get(k,0)/total*100:.0f}%)" for k in range(4)])
                print(f"  {state:12s}: {dist}")
        print()

    def _load_frames(self, video_path):
        cap     = cv2.VideoCapture(video_path)
        total   = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
        indices = np.linspace(0, total - 1, self.n_frames, dtype=int)
        frames  = []
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
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            pad = int(0.2 * max(w, h))
            x1, y1 = max(0, x - pad), max(0, y - pad)
            x2, y2 = min(frame.shape[1], x + w + pad), min(frame.shape[0], y + h + pad)
            return frame[y1:y2, x1:x2]
        return frame

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row    = self.df.iloc[idx]
        frames = self._load_frames(self.paths[idx])

        if self.face_detect:
            frames = [self._crop_face(f) for f in frames]
        if self.augment:
            frames = augment_frames(frames)

        tensors = []
        for f in frames:
            try:
                t = self.transform(f)
                if self.augment:
                    t = random_erasing_frame(t)
                tensors.append(t)
            except Exception:
                tensors.append(torch.zeros(3, self.image_size, self.image_size))

        video_tensor = torch.stack(tensors)

        if self.binary:
            labels = {
                s: torch.tensor(0 if int(row[s]) <= 1 else 1, dtype=torch.long)
                for s in STATES
            }
        else:
            labels = {s: torch.tensor(int(row[s]), dtype=torch.long) for s in STATES}

        return video_tensor, labels


def compute_binary_class_weights(train_df, device, cap=4.0):
    """
    Class weights pour mode binaire depuis le train set.
    cap=4.0 — ratio max pour eviter le collapse (Run 5 avait cap=10 = desastre).

    Distribution binaire train (calculee depuis TrainLabels.csv) :
      Boredom     : Low=77%  High=23%  -> weights [0.298, 1.000]
      Engagement  : Low=5%   High=95%  -> weights [1.000, 0.250]
      Confusion   : Low=91%  High=9%   -> weights [0.250, 1.000]
      Frustration : Low=96%  High=4%   -> weights [0.250, 1.000]
    """
    class_weights = {}
    print(f"[Class Weights binaires (cap={cap}x)]")
    for state in STATES:
        binary  = (train_df[state].values >= 2).astype(int)
        counts  = np.bincount(binary, minlength=2).astype(float)
        weights = 1.0 / (counts + 1e-6)
        weights = weights / weights.max()
        weights = np.clip(weights, 1.0 / cap, 1.0)
        class_weights[state] = torch.FloatTensor(weights).to(device)
        print(f"  {state:12s}: Low={int(counts[0])}({counts[0]/counts.sum()*100:.0f}%)  "
              f"High={int(counts[1])}({counts[1]/counts.sum()*100:.0f}%)  "
              f"weights=[{weights[0]:.3f}, {weights[1]:.3f}]")
    return class_weights