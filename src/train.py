"""
train.py — Boucle d'entraînement
Run 5 changes:
  - Focal Loss gamma 1.5 (was 2.0 — trop agressif avec class_weights forts)
  - Early stopping sur mean(F1_Engagement, F1_Confusion, F1_Frustration)
    → Boredom exclu du critère d'arrêt (val distribution biaisée : 3% train vs 9% val)
  - Boredom loggé séparément pour monitoring
  - use_mixup confirmé off
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm

from src.utils import AverageMeter
from src.evaluate import compute_metrics

STATES = ["Boredom", "Engagement", "Confusion", "Frustration"]
# États utilisés pour l'early stopping — Boredom exclu car val distribution biaisée
STATES_FOR_ES = ["Engagement", "Confusion", "Frustration"]


# ── Focal Loss ─────────────────────────────────────────────────────────────────

class FocalLoss(torch.nn.Module):
    """
    FL(p) = -alpha*(1-p)^gamma * log(p)
    gamma=1.5 (Run 5) — moins agressif que 2.0 quand class_weights sont déjà forts.
    """
    def __init__(self, weight=None, gamma=1.5, label_smoothing=0.1):
        super().__init__()
        self.weight          = weight
        self.gamma           = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        ce = F.cross_entropy(
            logits, targets,
            weight=self.weight,
            label_smoothing=self.label_smoothing,
            reduction="none"
        )
        pt = torch.exp(-ce)
        return (((1 - pt) ** self.gamma) * ce).mean()


# ── MultiTaskFocalLoss ─────────────────────────────────────────────────────────

class MultiTaskFocalLoss(torch.nn.Module):
    """
    Loss multi-tâche avec Focal Loss par état affectif.
    class_weights calculés depuis AllLabels.csv via dataset.compute_class_weights().
    """
    def __init__(
        self,
        class_weights: dict = None,
        gamma: float = 1.5,            # Run 5: 1.5 (was 2.0)
        label_smoothing: float = 0.1,  # Run 5: 0.1 (was 0.05)
    ):
        super().__init__()
        self.losses = torch.nn.ModuleDict()
        for state in STATES:
            w = class_weights.get(state) if class_weights else None
            self.losses[state] = FocalLoss(
                weight=w,
                gamma=gamma,
                label_smoothing=label_smoothing,
            )

    def forward(self, outputs: dict, targets: torch.Tensor):
        total, per_state = 0.0, {}
        for i, state in enumerate(STATES):
            l = self.losses[state](outputs[state], targets[:, i])
            per_state[state] = l.item()
            total = total + l
        return total, per_state


# ── Optimiseur + Scheduler ─────────────────────────────────────────────────────

def build_optimizer_scheduler(model, config, n_steps_per_epoch):
    """
    Différentiel LR : backbone × 0.1, reste × 1.
    OneCycleLR avec warmup 15%.
    """
    lr     = config["training"]["learning_rate"]   # 1e-4 en Run 5
    wd     = config["training"]["weight_decay"]
    epochs = config["training"]["epochs"]

    backbone_params = list(model.cnn.parameters())
    other_params    = [p for p in model.parameters()
                       if not any(p is bp for bp in backbone_params)]

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": lr * 0.1},
        {"params": other_params,    "lr": lr},
    ], weight_decay=wd)

    scheduler = OneCycleLR(
        optimizer,
        max_lr           = [lr * 0.1, lr],
        total_steps      = epochs * n_steps_per_epoch,
        pct_start        = 0.15,
        anneal_strategy  = "cos",
        div_factor       = 25.0,
        final_div_factor = 1000.0,
    )
    return optimizer, scheduler


# ── Epoch d'entraînement ───────────────────────────────────────────────────────

def train_one_epoch(
    model, dataloader, criterion, optimizer, scheduler,
    device, scaler=None, accumulation_steps=4,
):
    """use_mixup retiré — confirmé contre-productif sur DAiSEE déséquilibré."""
    model.train()
    loss_meter = AverageMeter("Loss")
    all_preds  = {s: [] for s in STATES}
    all_labels = {s: [] for s in STATES}
    n_batches  = len(dataloader)
    optimizer.zero_grad()

    for step, (frames, labels) in enumerate(tqdm(dataloader, desc="  Train", leave=False)):
        frames = frames.to(device, non_blocking=True)
        labels = {k: v.to(device, non_blocking=True) for k, v in labels.items()}
        labels_tensor = torch.stack([labels[s] for s in STATES], dim=1)  # (B, 4)

        with autocast(enabled=(scaler is not None)):
            outputs = model(frames)
            loss, _ = criterion(outputs, labels_tensor)
            loss = loss / accumulation_steps

        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if ((step + 1) % accumulation_steps == 0) or (step + 1 == n_batches):
            if scaler:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()

        loss_meter.update(loss.item() * accumulation_steps, frames.size(0))

        with torch.no_grad():
            for i, state in enumerate(STATES):
                all_preds[state].extend(outputs[state].argmax(1).cpu().numpy())
                all_labels[state].extend(labels_tensor[:, i].cpu().numpy())

    for s in STATES:
        all_preds[s]  = np.array(all_preds[s])
        all_labels[s] = np.array(all_labels[s])

    return loss_meter.avg, compute_metrics(all_preds, all_labels)


# ── Boucle principale ──────────────────────────────────────────────────────────

def train(
    model, train_loader, val_loader, criterion, optimizer, scheduler,
    device, config: dict, early_stopping=None,
    mlflow_run=None, run_name: str = None,
):
    """
    Run 5 — early stopping sur mean F1(Engagement, Confusion, Frustration).
    Boredom loggé séparément (val distribution biaisée 3% train → 9% val).
    """
    from src.evaluate import evaluate
    try:
        import mlflow
        _mlflow = mlflow if mlflow_run is not None else None
    except ImportError:
        _mlflow = None

    epochs          = config["training"]["epochs"]
    mixed_precision = config["training"]["mixed_precision"]
    accum_steps     = config["training"]["accumulation_steps"]
    unfreeze_epoch  = config["training"].get("unfreeze_epoch", 10)

    scaler = GradScaler() if mixed_precision and torch.cuda.is_available() else None

    history = {
        "train_loss": [], "val_loss": [], "lr": [],
        **{f"train_acc_{s}": [] for s in STATES},
        **{f"val_acc_{s}":   [] for s in STATES},
        **{f"val_f1_{s}":    [] for s in STATES},
    }

    print(f"\n{'='*65}")
    print(f"  Run : {run_name or 'unnamed'}")
    print(f"  Epochs : {epochs} | Device : {device}")
    print(f"  AMP : {mixed_precision} | Accum : {accum_steps}")
    print(f"  Early stopping sur : {STATES_FOR_ES}")
    print(f"  Boredom : loggé séparément (distribution val biaisée)")
    print(f"{'='*65}\n")

    if _mlflow:
        _mlflow.log_params({
            "run_name":           run_name or "unnamed",
            "epochs":             epochs,
            "batch_size":         config["training"]["batch_size"],
            "accumulation_steps": accum_steps,
            "effective_batch":    config["training"]["batch_size"] * accum_steps,
            "learning_rate":      config["training"]["learning_rate"],
            "weight_decay":       config["training"]["weight_decay"],
            "mixed_precision":    mixed_precision,
            "use_mixup":          False,
            "unfreeze_epoch":     unfreeze_epoch,
            "backbone":           config["model"]["backbone"],
            "hidden_size":        config["model"]["hidden_size"],
            "gru_layers":         config["model"]["gru_layers"],
            "n_frames":           config["dataset"]["n_frames"],
            "image_size":         config["dataset"]["image_size"],
            "n_train":            len(train_loader.dataset),
            "n_val":              len(val_loader.dataset),
            "early_stopping_on":  str(STATES_FOR_ES),
        })
        if hasattr(criterion, "losses"):
            focal = list(criterion.losses.values())[0]
            _mlflow.log_param("focal_gamma",     getattr(focal, "gamma", "N/A"))
            _mlflow.log_param("label_smoothing", getattr(focal, "label_smoothing", "N/A"))
            for state, loss_fn in criterion.losses.items():
                if loss_fn.weight is not None:
                    for i, w in enumerate(loss_fn.weight.cpu().numpy()):
                        _mlflow.log_param(f"cw_{state}_{i}", round(float(w), 4))

    best_val_f1_es = 0.0   # F1 sur STATES_FOR_ES (critère d'arrêt)
    best_val_f1_all = 0.0  # F1 tous états (pour logging)

    for epoch in range(1, epochs + 1):

        # Dégel progressif du backbone
        if epoch == unfreeze_epoch and hasattr(model, "unfreeze_backbone"):
            model.unfreeze_backbone(n_blocks=2)
            print(f"[Epoch {epoch}] Backbone partiellement dégelé.")

        train_loss, train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            device, scaler, accum_steps,
        )

        val_metrics, _, _, val_loss = evaluate(model, val_loader, device, criterion)
        current_lr = optimizer.param_groups[1]["lr"]

        # Historique
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(current_lr)
        for s in STATES:
            history[f"train_acc_{s}"].append(train_metrics[s]["accuracy"])
            history[f"val_acc_{s}"].append(val_metrics[s]["accuracy"])
            history[f"val_f1_{s}"].append(val_metrics[s]["f1_macro"])

        # Métriques d'arrêt : moyenne sur STATES_FOR_ES uniquement
        mean_val_f1_es  = np.mean([val_metrics[s]["f1_macro"] for s in STATES_FOR_ES])
        mean_val_f1_all = val_metrics["mean"]["f1_macro"]
        mean_val_acc    = val_metrics["mean"]["accuracy"]

        print(
            f"Epoch [{epoch:>3}/{epochs}] | "
            f"Loss train={train_loss:.4f} val={val_loss:.4f} | "
            f"Acc={mean_val_acc:.2f}% | "
            f"F1(ES)={mean_val_f1_es:.2f}% F1(all)={mean_val_f1_all:.2f}% | "
            f"LR={current_lr:.2e}"
        )
        for s in STATES:
            marker = " ← (exclu ES)" if s == "Boredom" else ""
            print(
                f"  {s:<14}: train={train_metrics[s]['accuracy']:.1f}%  "
                f"val={val_metrics[s]['accuracy']:.1f}%  "
                f"F1={val_metrics[s]['f1_macro']:.1f}%"
                f"{marker}"
            )

        if _mlflow:
            metrics_to_log = {
                "train_loss":       train_loss,
                "val_loss":         val_loss,
                "learning_rate":    current_lr,
                "mean_val_acc":     mean_val_acc,
                "mean_val_f1_all":  mean_val_f1_all,
                "mean_val_f1_es":   mean_val_f1_es,
            }
            for s in STATES:
                metrics_to_log[f"train_acc_{s}"] = train_metrics[s]["accuracy"]
                metrics_to_log[f"val_acc_{s}"]   = val_metrics[s]["accuracy"]
                metrics_to_log[f"val_f1_{s}"]    = val_metrics[s]["f1_macro"]
                metrics_to_log[f"val_kappa_{s}"] = val_metrics[s].get("kappa", 0.0)
            _mlflow.log_metrics(metrics_to_log, step=epoch)

        # Early stopping sur F1 des 3 états stables (Boredom exclu)
        if early_stopping:
            early_stopping(-mean_val_f1_es, model)
            if mean_val_f1_es > best_val_f1_es:
                best_val_f1_es  = mean_val_f1_es
                best_val_f1_all = mean_val_f1_all
            if early_stopping.early_stop:
                print(f"\n[Train] Arrêt anticipé à l'epoch {epoch}.")
                print(f"[Train] Meilleur F1(ES={STATES_FOR_ES}) : {best_val_f1_es:.2f}%")
                print(f"[Train] F1 tous états correspondant      : {best_val_f1_all:.2f}%")
                if _mlflow:
                    _mlflow.log_param("stopped_at_epoch", epoch)
                    _mlflow.log_metric("best_val_f1_es",  best_val_f1_es)
                    _mlflow.log_metric("best_val_f1_all", best_val_f1_all)
                break

    if _mlflow:
        _mlflow.log_metric("best_val_f1_es",  best_val_f1_es)
        _mlflow.log_metric("best_val_f1_all", best_val_f1_all)
        checkpoint_path = early_stopping.checkpoint_path if early_stopping else None
        if checkpoint_path and __import__("os").path.exists(checkpoint_path):
            _mlflow.log_artifact(checkpoint_path, artifact_path="checkpoints")

    return history
