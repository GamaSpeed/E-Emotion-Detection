"""
evaluate.py — Métriques d'évaluation : Accuracy, F1, Confusion Matrix, Kappa
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, cohen_kappa_score
)


STATES  = ["Boredom", "Engagement", "Confusion", "Frustration"]
LEVELS  = ["Very Low", "Low", "High", "Very High"]


# Évaluation sur un DataLoader

@torch.no_grad()
def evaluate(model, dataloader, device, criterion=None):
    """
    Évalue le modèle sur un dataloader complet.

    Returns:
        metrics : dict contenant accuracy, F1, Kappa par état
        all_preds : dict {state: np.array}
        all_labels : dict {state: np.array}
        avg_loss : float (si criterion fourni)
    """
    model.eval()
    all_preds  = {s: [] for s in STATES}
    all_labels = {s: [] for s in STATES}
    total_loss = 0.0
    n_batches  = 0

    for frames, labels in dataloader:
        frames = frames.to(device)
        labels = labels.to(device)

        outputs = model(frames)

        if criterion:
            loss, _ = criterion(outputs, labels)
            total_loss += loss.item()
            n_batches  += 1

        for i, state in enumerate(STATES):
            preds = outputs[state].argmax(dim=1).cpu().numpy()
            all_preds[state].extend(preds)
            all_labels[state].extend(labels[:, i].cpu().numpy())

    # Convertir en arrays
    for s in STATES:
        all_preds[s]  = np.array(all_preds[s])
        all_labels[s] = np.array(all_labels[s])

    metrics = compute_metrics(all_preds, all_labels)
    avg_loss = total_loss / max(n_batches, 1)

    return metrics, all_preds, all_labels, avg_loss


# Calcul des métriques

def compute_metrics(all_preds: dict, all_labels: dict) -> dict:
    """Calcule Accuracy, F1 macro, Cohen's Kappa pondéré pour chaque état."""
    metrics = {}

    for state in STATES:
        preds  = all_preds[state]
        labels = all_labels[state]

        acc   = accuracy_score(labels, preds) * 100
        f1    = f1_score(labels, preds, average="macro", zero_division=0) * 100
        kappa = cohen_kappa_score(labels, preds, weights="quadratic")

        metrics[state] = {"accuracy": acc, "f1_macro": f1, "kappa": kappa}

    # Moyennes globales
    metrics["mean"] = {
        "accuracy": np.mean([metrics[s]["accuracy"] for s in STATES]),
        "f1_macro": np.mean([metrics[s]["f1_macro"] for s in STATES]),
        "kappa":    np.mean([metrics[s]["kappa"]    for s in STATES]),
    }

    return metrics


def print_metrics(metrics: dict):
    """Affiche les métriques sous forme de tableau."""
    print("\n" + "="*65)
    print(f"{'État':<14} {'Accuracy':>10} {'F1 Macro':>10} {'Kappa':>10}")
    print("="*65)
    for state in STATES:
        m = metrics[state]
        print(f"{state:<14} {m['accuracy']:>9.2f}% {m['f1_macro']:>9.2f}% {m['kappa']:>10.4f}")
    print("-"*65)
    m = metrics["mean"]
    print(f"{'MOYENNE':<14} {m['accuracy']:>9.2f}% {m['f1_macro']:>9.2f}% {m['kappa']:>10.4f}")
    print("="*65 + "\n")


# Visualisations

def plot_confusion_matrices(all_preds: dict, all_labels: dict, save_path: str = None):
    """Affiche les 4 matrices de confusion côte à côte."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    axes = axes.flatten()

    for i, state in enumerate(STATES):
        cm = confusion_matrix(all_labels[state], all_preds[state], labels=[0,1,2,3])
        cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

        sns.heatmap(
            cm_pct, annot=True, fmt=".1f", cmap="Blues",
            xticklabels=LEVELS, yticklabels=LEVELS,
            ax=axes[i], cbar=True, linewidths=0.5,
        )
        axes[i].set_title(f"{state}", fontsize=14, fontweight="bold", pad=10)
        axes[i].set_xlabel("Prédiction", fontsize=11)
        axes[i].set_ylabel("Vérité terrain", fontsize=11)
        axes[i].tick_params(axis="x", rotation=30)
        axes[i].tick_params(axis="y", rotation=0)

    plt.suptitle("Matrices de Confusion (% par ligne)", fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Plot] Matrices de confusion → {save_path}")
    plt.show()


def plot_training_history(history: dict, save_path: str = None):
    """
    Trace les courbes de loss et accuracy pendant l'entraînement.

    Args:
        history : dict avec clés 'train_loss', 'val_loss',
                  et 'val_acc_{state}' pour chaque état.
    """
    n_states = len(STATES)
    fig, axes = plt.subplots(1, 2 + n_states // 2, figsize=(20, 5))

    # Loss 
    axes[0].plot(history["train_loss"], label="Train", color="#2196F3", linewidth=2)
    axes[0].plot(history["val_loss"],   label="Val",   color="#F44336", linewidth=2)
    axes[0].set_title("Loss d'entraînement", fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # ccuracy par état 
    colors = ["#4CAF50", "#2196F3", "#FF9800", "#9C27B0"]
    for i, state in enumerate(STATES):
        key = f"val_acc_{state}"
        if key in history:
            ax_idx = 1 + i // 2 if i < 2 else 2 + i // 2
            ax_idx = min(ax_idx, len(axes) - 1)
            axes[1].plot(history[key], label=state, color=colors[i], linewidth=2)

    axes[1].set_title("Accuracy Validation par État", fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # LR
    if "lr" in history and len(axes) > 2:
        axes[2].plot(history["lr"], color="#607D8B", linewidth=2)
        axes[2].set_title("Learning Rate", fontweight="bold")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("LR")
        axes[2].set_yscale("log")
        axes[2].grid(True, alpha=0.3)

    plt.suptitle("Historique d'Entraînement", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Plot] Courbes d'entraînement → {save_path}")
    plt.show()


def plot_metrics_comparison(metrics: dict, baseline_metrics: dict = None, save_path: str = None):
    """
    Compare les métriques du modèle avec les baselines du papier DAiSEE.
    """
    # Baselines du papier (LRCN)
    if baseline_metrics is None:
        baseline_metrics = {
            "Boredom":     {"accuracy": 53.7},
            "Engagement":  {"accuracy": 57.9},
            "Confusion":   {"accuracy": 72.3},
            "Frustration": {"accuracy": 73.5},
        }

    x = np.arange(len(STATES))
    width = 0.35

    our_accs      = [metrics[s]["accuracy"]          for s in STATES]
    baseline_accs = [baseline_metrics[s]["accuracy"] for s in STATES]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, baseline_accs, width, label="LRCN (Papier)", color="#90CAF9", edgecolor="white")
    bars2 = ax.bar(x + width/2, our_accs,      width, label="Notre Modèle",  color="#1565C0", edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(STATES, fontsize=12)
    ax.set_ylabel("Top-1 Accuracy (%)", fontsize=12)
    ax.set_title("Comparaison avec les Baselines du Papier DAiSEE", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_ylim(0, 100)
    ax.grid(True, axis="y", alpha=0.3)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Plot] Comparaison métriques → {save_path}")
    plt.show()
