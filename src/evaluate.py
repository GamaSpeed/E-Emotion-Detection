"""
evaluate.py — Métriques d'évaluation : Accuracy, F1, Confusion Matrix, Kappa
Run 6 : support mode binaire (2 classes) et 4-classes
  - plot_confusion_matrices détecte automatiquement le nombre de classes
  - compute_metrics adapté (kappa linéaire en binaire, quadratique en 4-classes)
  - print_metrics inchangé
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, cohen_kappa_score
)


STATES       = ["Boredom", "Engagement", "Confusion", "Frustration"]
LEVELS_4     = ["Very Low", "Low", "High", "Very High"]
LEVELS_2     = ["Low", "High"]


# ── Évaluation sur un DataLoader ───────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, dataloader, device, criterion=None):
    model.eval()
    all_preds  = {s: [] for s in STATES}
    all_labels = {s: [] for s in STATES}
    total_loss = 0.0
    n_batches  = 0

    for frames, labels in dataloader:
        frames = frames.to(device)
        labels = {k: v.to(device) for k, v in labels.items()}
        labels_tensor = torch.stack([labels[s] for s in STATES], dim=1)

        outputs = model(frames)

        if criterion:
            loss, _ = criterion(outputs, labels_tensor)
            total_loss += loss.item()
            n_batches  += 1

        for i, state in enumerate(STATES):
            preds = outputs[state].argmax(dim=1).cpu().numpy()
            all_preds[state].extend(preds)
            all_labels[state].extend(labels_tensor[:, i].cpu().numpy())

    for s in STATES:
        all_preds[s]  = np.array(all_preds[s])
        all_labels[s] = np.array(all_labels[s])

    metrics  = compute_metrics(all_preds, all_labels)
    avg_loss = total_loss / max(n_batches, 1)

    return metrics, all_preds, all_labels, avg_loss


# ── Calcul des métriques ───────────────────────────────────────────────────────

def compute_metrics(all_preds: dict, all_labels: dict) -> dict:
    """
    Calcule Accuracy, F1 macro, Cohen's Kappa pour chaque état.
    Détecte automatiquement le mode binaire (2 classes) ou 4-classes.
    Kappa : linéaire en binaire, quadratique en 4-classes.
    """
    metrics = {}

    for state in STATES:
        preds  = all_preds[state]
        labels = all_labels[state]

        n_classes = len(np.unique(np.concatenate([labels, preds])))
        binary    = n_classes <= 2

        acc   = accuracy_score(labels, preds) * 100
        f1    = f1_score(labels, preds, average="macro", zero_division=0) * 100

        # Kappa linéaire en binaire (quadratique n'a pas de sens sur 2 classes)
        try:
            kappa_weights = None if binary else "quadratic"
            kappa = cohen_kappa_score(labels, preds, weights=kappa_weights)
        except Exception:
            kappa = 0.0

        metrics[state] = {"accuracy": acc, "f1_macro": f1, "kappa": kappa}

    metrics["mean"] = {
        "accuracy": np.mean([metrics[s]["accuracy"] for s in STATES]),
        "f1_macro": np.mean([metrics[s]["f1_macro"] for s in STATES]),
        "kappa":    np.mean([metrics[s]["kappa"]    for s in STATES]),
    }

    return metrics


def print_metrics(metrics: dict):
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


# ── Matrices de confusion ──────────────────────────────────────────────────────

def plot_confusion_matrices(
    all_preds: dict,
    all_labels: dict,
    save_path: str = None,
    binary: bool = None,   # None = détection automatique
):
    """
    Matrices de confusion adaptées au mode binaire ou 4-classes.
    binary=None : détecte automatiquement depuis les données.
    """
    # Détection automatique du mode
    if binary is None:
        sample = all_labels[STATES[0]]
        binary = len(np.unique(sample)) <= 2

    if binary:
        class_labels = [0, 1]
        level_names  = LEVELS_2
        title_suffix = "(Binaire : Low / High)"
    else:
        class_labels = [0, 1, 2, 3]
        level_names  = LEVELS_4
        title_suffix = "(% par ligne)"

    fig, axes = plt.subplots(2, 2, figsize=(12 if binary else 14, 10 if binary else 11))
    axes = axes.flatten()

    for i, state in enumerate(STATES):
        labels_arr = all_labels[state].astype(int)
        preds_arr  = all_preds[state].astype(int)

        cm     = confusion_matrix(labels_arr, preds_arr, labels=class_labels)
        # Éviter division par zéro sur les lignes vides
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        cm_pct   = cm.astype(float) / row_sums * 100

        sns.heatmap(
            cm_pct,
            annot      = True,
            fmt        = ".1f",
            cmap       = "Blues",
            xticklabels = level_names,
            yticklabels = level_names,
            ax         = axes[i],
            cbar       = True,
            linewidths = 0.5,
            vmin       = 0,
            vmax       = 100,
        )

        # Accuracy sur la diagonale
        diag_acc = np.diag(cm_pct)
        title_acc = "  |  ".join([
            f"{level_names[j]}={diag_acc[j]:.0f}%"
            for j in range(len(class_labels))
            if cm[j].sum() > 0
        ])

        axes[i].set_title(f"{state}\n{title_acc}", fontsize=12, fontweight="bold", pad=8)
        axes[i].set_xlabel("Prédiction", fontsize=10)
        axes[i].set_ylabel("Vérité terrain", fontsize=10)
        axes[i].tick_params(axis="x", rotation=0 if binary else 30)
        axes[i].tick_params(axis="y", rotation=0)

    plt.suptitle(
        f"Matrices de Confusion {title_suffix}",
        fontsize=15, fontweight="bold", y=1.01
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Plot] Matrices de confusion → {save_path}")
    plt.show()


# ── Courbes d'entraînement ─────────────────────────────────────────────────────

def plot_training_history(history: dict, save_path: str = None):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss
    axes[0].plot(history["train_loss"], label="Train", color="#2196F3", linewidth=2)
    axes[0].plot(history["val_loss"],   label="Val",   color="#F44336", linewidth=2)
    axes[0].set_title("Loss", fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy val par état
    colors = ["#4CAF50", "#2196F3", "#FF9800", "#9C27B0"]
    for i, state in enumerate(STATES):
        key = f"val_acc_{state}"
        if key in history:
            axes[1].plot(history[key], label=state, color=colors[i], linewidth=2)
    axes[1].set_title("Accuracy Validation par État", fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # F1 val par état
    for i, state in enumerate(STATES):
        key = f"val_f1_{state}"
        if key in history:
            axes[2].plot(history[key], label=state, color=colors[i], linewidth=2)
    axes[2].set_title("F1 Macro Validation par État", fontweight="bold")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("F1 Macro (%)")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.suptitle("Historique d'Entraînement", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Plot] Courbes → {save_path}")
    plt.show()


# ── Comparaison baselines ──────────────────────────────────────────────────────

def plot_metrics_comparison(
    metrics: dict,
    baseline_metrics: dict = None,
    save_path: str = None,
    binary: bool = False,
):
    if baseline_metrics is None:
        if binary:
            # Seul Engagement est documenté en binaire dans le papier
            baseline_metrics = {
                "Boredom":     {"accuracy": None},
                "Engagement":  {"accuracy": 94.6},
                "Confusion":   {"accuracy": None},
                "Frustration": {"accuracy": None},
            }
        else:
            baseline_metrics = {
                "Boredom":     {"accuracy": 53.7},
                "Engagement":  {"accuracy": 57.9},
                "Confusion":   {"accuracy": 72.3},
                "Frustration": {"accuracy": 73.5},
            }

    x     = np.arange(len(STATES))
    width = 0.35

    our_accs      = [metrics[s]["accuracy"] for s in STATES]
    baseline_accs = [baseline_metrics[s]["accuracy"] or 0 for s in STATES]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, baseline_accs, width,
                   label="LRCN (Papier)", color="#90CAF9", edgecolor="white")
    bars2 = ax.bar(x + width/2, our_accs, width,
                   label="Notre Modèle",  color="#1565C0", edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(STATES, fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    mode_str = "Binaire" if binary else "4-Classes"
    ax.set_title(f"Comparaison avec Baselines DAiSEE ({mode_str})",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_ylim(0, 105)
    ax.grid(True, axis="y", alpha=0.3)

    for bar in bars1:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.5,
                    f"{h:.1f}%", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{bar.get_height():.1f}%", ha="center", va="bottom",
                fontsize=9, fontweight="bold")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Plot] Comparaison → {save_path}")
    plt.show()