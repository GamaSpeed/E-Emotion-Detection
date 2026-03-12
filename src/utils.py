"""
utils.py — Fonctions utilitaires : seed, class weights, early stopping, logging
"""

import os
import random
import numpy as np
import torch
import yaml
from pathlib import Path


# Reproductibilité

def set_seed(seed: int = 42):
    """Fixe tous les seeds pour la reproductibilité."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[Seed] Seed fixé à {seed}")


# Chargement de la config

def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Charge le fichier de configuration YAML."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    print(f"[Config] Chargée depuis : {config_path}")
    return config


# Calcul des class weights (déséquilibre)

# def compute_class_weights(df, label_cols, device="cpu"):
#     """
#     Calcule les poids de classe pour chaque état affectif.
#     Formule : weight_i = N_total / (n_classes * count_i)

#     Returns:
#         dict {state: Tensor(4)} à passer à MultiTaskLoss
#     """
#     class_weights = {}
#     n_classes = 4

#     for col in label_cols:
#         counts = np.bincount(df[col].astype(int).values, minlength=n_classes).astype(float)
#         n_total = counts.sum()
#         weights = n_total / (n_classes * (counts + 1e-6))
#         weights = weights / weights.sum() * n_classes  # normaliser
#         class_weights[col] = torch.FloatTensor(weights).to(device)
#         print(f"[Weights] {col}: {[f'{w:.2f}' for w in weights]}")

#     return class_weights


# Early Stopping

class EarlyStopping:
    """
    Arrête l'entraînement si la validation ne s'améliore pas pendant `patience` epochs.
    Sauvegarde automatiquement le meilleur modèle.
    """

    def __init__(self, patience: int = 8, min_delta: float = 1e-4,
                 checkpoint_path: str = "outputs/checkpoints/best_model.pt"):
        self.patience = patience
        self.min_delta = min_delta
        self.checkpoint_path = checkpoint_path
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    def __call__(self, val_loss: float, model: torch.nn.Module):
        score = -val_loss  # on minimise la loss

        if self.best_score is None:
            self.best_score = score
            self._save(model)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            print(f"[EarlyStopping] Pas d'amélioration ({self.counter}/{self.patience})")
            if self.counter >= self.patience:
                self.early_stop = True
                print("[EarlyStopping] Arrêt anticipé déclenché.")
        else:
            self.best_score = score
            self._save(model)
            self.counter = 0

    def _save(self, model):
        torch.save(model.state_dict(), self.checkpoint_path)
        print(f"[EarlyStopping] Meilleur modèle sauvegardé → {self.checkpoint_path}")


# AverageMeter (suivi des métriques)

class AverageMeter:
    """Calcule et stocke la moyenne et valeur courante d'une métrique."""

    def __init__(self, name: str):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self):
        return f"{self.name}: {self.avg:.4f}"


# Sauvegarde et chargement de checkpoints

def save_checkpoint(state: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
    print(f"[Checkpoint] Sauvegardé → {path}")


def load_checkpoint(model, optimizer, path: str, device="cpu"):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    if optimizer and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    epoch = checkpoint.get("epoch", 0)
    print(f"[Checkpoint] Chargé depuis {path} (epoch {epoch})")
    return model, optimizer, epoch
