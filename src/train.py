"""
train.py — Boucle d'entraînement complète
  - MultiTaskFocalLoss : Focal Loss pour gérer le déséquilibre des classes
  - OneCycleLR par step avec warmup 15%
  - Early stopping sur val F1 macro moyen (plus robuste qu'accuracy)
  - Mixup optionnel pour régularisation
  - build_optimizer_scheduler à appeler depuis le notebook
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


# Focal Loss

class FocalLoss(torch.nn.Module):
    """FL(p) = -alpha*(1-p)^gamma * log(p) — réduit le poids des classes faciles."""
    def __init__(self, weight=None, gamma=2.0, label_smoothing=0.1):
        super().__init__()
        self.weight          = weight
        self.gamma           = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.weight,
                             label_smoothing=self.label_smoothing, reduction="none")
        pt = torch.exp(-ce)
        return (((1 - pt) ** self.gamma) * ce).mean()


# MultiTaskFocalLoss

class MultiTaskFocalLoss(torch.nn.Module):
    """Loss multi-tâche avec Focal Loss par état affectif."""
    def __init__(self, class_weights: dict = None, gamma: float = 2.0,
                 label_smoothing: float = 0.1):
        super().__init__()
        self.losses = torch.nn.ModuleDict()
        for state in STATES:
            w = class_weights.get(state) if class_weights else None
            self.losses[state] = FocalLoss(weight=w, gamma=gamma,
                                           label_smoothing=label_smoothing)

    def forward(self, outputs: dict, targets: torch.Tensor):
        total, per_state = 0.0, {}
        for i, state in enumerate(STATES):
            l = self.losses[state](outputs[state], targets[:, i])
            per_state[state] = l.item()
            total = total + l
        return total, per_state


# Mixup

def mixup_data(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam

def mixup_criterion(criterion, outputs, y_a, y_b, lam):
    la, _ = criterion(outputs, y_a)
    lb, _ = criterion(outputs, y_b)
    return lam * la + (1 - lam) * lb, {}


# Optimiseur + Scheduler

def build_optimizer_scheduler(model, config, n_steps_per_epoch):
    """
    À appeler depuis le notebook :
        optimizer, scheduler = build_optimizer_scheduler(model, config, len(train_loader))
        criterion = MultiTaskFocalLoss(class_weights=..., gamma=2.0)
    """
    lr     = config["training"]["learning_rate"]
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
        max_lr          = [lr * 0.1, lr],
        total_steps     = epochs * n_steps_per_epoch,
        pct_start       = 0.15,
        anneal_strategy = "cos",
        div_factor      = 25.0,
        final_div_factor= 1000.0,
    )
    return optimizer, scheduler


# Epoch d'entraînement

def train_one_epoch(model, dataloader, criterion, optimizer, scheduler,
                    device, scaler=None, accumulation_steps=4, use_mixup=False):
    model.train()
    loss_meter = AverageMeter("Loss")
    all_preds  = {s: [] for s in STATES}
    all_labels = {s: [] for s in STATES}
    n_batches  = len(dataloader)
    optimizer.zero_grad()

    for step, (frames, labels) in enumerate(tqdm(dataloader, desc="  Train", leave=False)):
        frames = frames.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if use_mixup and np.random.random() < 0.5:
            frames, y_a, y_b, lam = mixup_data(frames, labels, alpha=0.2)
            with autocast(enabled=(scaler is not None)):
                outputs = model(frames)
                loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)[0]
                loss = loss / accumulation_steps
        else:
            with autocast(enabled=(scaler is not None)):
                outputs = model(frames)
                loss, _ = criterion(outputs, labels)
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
                all_labels[state].extend(labels[:, i].cpu().numpy())

    for s in STATES:
        all_preds[s]  = np.array(all_preds[s])
        all_labels[s] = np.array(all_labels[s])

    return loss_meter.avg, compute_metrics(all_preds, all_labels)


# Boucle principale

def train(model, train_loader, val_loader, criterion, optimizer, scheduler,
          device, config: dict, early_stopping=None,
          mlflow_run=None, run_name: str = None):
    """
    mlflow_run  : objet mlflow.ActiveRun (optionnel), obtenu via `with mlflow.start_run() as run`
    run_name    : nom lisible du run (ex: "focal_gamma05_lr3e4")
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
    unfreeze_epoch  = config["training"].get("unfreeze_epoch", 5)
    use_mixup       = config["training"].get("use_mixup", False)

    scaler = GradScaler() if mixed_precision and torch.cuda.is_available() else None

    history = {
        "train_loss": [], "val_loss": [], "lr": [],
        **{f"train_acc_{s}": [] for s in STATES},
        **{f"val_acc_{s}":   [] for s in STATES},
        **{f"val_f1_{s}":    [] for s in STATES},
    }

    print(f"\n{'='*60}")
    print(f"  Entraînement : {epochs} epochs | Device : {device}")
    print(f"  AMP : {mixed_precision} | Accum. Steps : {accum_steps} | Mixup : {use_mixup}")
    if run_name:
        print(f"  MLflow run    : {run_name}")
    print(f"{'='*60}\n")

    # Logger les hyperparamètres MLflow une seule fois 
    if _mlflow:
        _mlflow.log_params({
            "run_name":          run_name or "unnamed",
            "epochs":            epochs,
            "batch_size":        config["training"]["batch_size"],
            "accumulation_steps":accum_steps,
            "effective_batch":   config["training"]["batch_size"] * accum_steps,
            "learning_rate":     config["training"]["learning_rate"],
            "weight_decay":      config["training"]["weight_decay"],
            "mixed_precision":   mixed_precision,
            "use_mixup":         use_mixup,
            "unfreeze_epoch":    unfreeze_epoch,
            "backbone":          config["model"]["backbone"],
            "hidden_size":       config["model"]["hidden_size"],
            "gru_layers":        config["model"]["gru_layers"],
            "n_frames":          config["dataset"]["n_frames"],
            "image_size":        config["dataset"]["image_size"],
            "n_train":           len(train_loader.dataset),
            "n_val":             len(val_loader.dataset),
        })
        # Logger les class weights si disponibles
        if hasattr(criterion, "losses"):
            for state, loss_fn in criterion.losses.items():
                if loss_fn.weight is not None:
                    for i, w in enumerate(loss_fn.weight.cpu().numpy()):
                        _mlflow.log_param(f"cw_{state}_{i}", round(float(w), 4))
            _mlflow.log_param("focal_gamma",         getattr(list(criterion.losses.values())[0], "gamma", "N/A"))
            _mlflow.log_param("label_smoothing",     getattr(list(criterion.losses.values())[0], "label_smoothing", "N/A"))

    best_val_f1 = 0.0

    for epoch in range(1, epochs + 1):

        if epoch == unfreeze_epoch and hasattr(model, "unfreeze_backbone"):
            model.unfreeze_backbone(n_blocks=2)
            print(f"[Epoch {epoch}] Backbone partiellement dégelé.")

        train_loss, train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            device, scaler, accum_steps, use_mixup
        )

        val_metrics, _, _, val_loss = evaluate(model, val_loader, device, criterion)
        current_lr = optimizer.param_groups[1]["lr"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(current_lr)
        for s in STATES:
            history[f"train_acc_{s}"].append(train_metrics[s]["accuracy"])
            history[f"val_acc_{s}"].append(val_metrics[s]["accuracy"])
            history[f"val_f1_{s}"].append(val_metrics[s]["f1_macro"])

        mean_val_acc = val_metrics["mean"]["accuracy"]
        mean_val_f1  = val_metrics["mean"]["f1_macro"]
        mean_train_acc = train_metrics["mean"]["accuracy"]

        print(f"Epoch [{epoch:>3}/{epochs}] | Loss train={train_loss:.4f} val={val_loss:.4f} "
              f"| Acc={mean_val_acc:.2f}% F1={mean_val_f1:.2f}% | LR={current_lr:.2e}")
        for s in STATES:
            print(f"  {s:<14}: train={train_metrics[s]['accuracy']:.1f}%  "
                  f"val={val_metrics[s]['accuracy']:.1f}%  "
                  f"F1={val_metrics[s]['f1_macro']:.1f}%")

        # Logger les métriques par epoch dans MLflow 
        if _mlflow:
            metrics_to_log = {
                "train_loss":     train_loss,
                "val_loss":       val_loss,
                "learning_rate":  current_lr,
                "mean_val_acc":   mean_val_acc,
                "mean_val_f1":    mean_val_f1,
                "mean_train_acc": mean_train_acc,
            }
            for s in STATES:
                metrics_to_log[f"train_acc_{s}"] = train_metrics[s]["accuracy"]
                metrics_to_log[f"val_acc_{s}"]   = val_metrics[s]["accuracy"]
                metrics_to_log[f"val_f1_{s}"]    = val_metrics[s]["f1_macro"]
                metrics_to_log[f"val_kappa_{s}"] = val_metrics[s].get("kappa", 0.0)
            _mlflow.log_metrics(metrics_to_log, step=epoch)

        # Early stopping sur F1 macro moyen
        if early_stopping:
            early_stopping(-mean_val_f1, model)
            if mean_val_f1 > best_val_f1:
                best_val_f1 = mean_val_f1
            if early_stopping.early_stop:
                print(f"\n[Train] Arrêt anticipé à l'epoch {epoch}.")
                print(f"[Train] Meilleur val F1 : {best_val_f1:.2f}%")
                if _mlflow:
                    _mlflow.log_param("stopped_at_epoch", epoch)
                    _mlflow.log_metric("best_val_f1", best_val_f1)
                break

    # Logger le meilleur checkpoint comme artefact MLflow
    if _mlflow:
        _mlflow.log_metric("best_val_f1", best_val_f1)
        checkpoint_path = early_stopping.checkpoint_path if early_stopping else None
        if checkpoint_path and __import__("os").path.exists(checkpoint_path):
            _mlflow.log_artifact(checkpoint_path, artifact_path="checkpoints")
            print(f"[MLflow] Checkpoint loggé : {checkpoint_path}")

    return history