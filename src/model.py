"""
model.py — EfficientNet-B4 + GRU Bidirectionnel + Multi-head Attention
Architecture multi-tâche : 4 têtes (Boredom, Engagement, Confusion, Frustration)
"""

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights


class EmotionModel(nn.Module):
    """
    Modèle de détection d'états affectifs multi-tâche.

    Architecture :
        1. EfficientNet-B4  → extraction de features spatiales par frame
        2. GRU Bidirectionnel → modélisation de la dynamique temporelle
        3. Multi-head Attention → pondération des frames importantes
        4. 4 têtes de classification → une par état affectif

    Args:
        n_classes       : Nombre de niveaux par état (4 : Very Low→Very High).
        hidden_size     : Dimension du GRU (256 pour 4-6 GB VRAM).
        gru_layers      : Nombre de couches GRU.
        gru_dropout     : Dropout entre les couches GRU.
        n_heads         : Nombre de têtes d'attention.
        dropout_clf     : Dropout dans les têtes de classification.
        freeze_backbone : Si True, gèle les couches du CNN au début.
    """

    AFFECTIVE_STATES = ["Boredom", "Engagement", "Confusion", "Frustration"]

    def __init__(
        self,
        n_classes: int = 4,
        hidden_size: int = 256,
        gru_layers: int = 2,
        gru_dropout: float = 0.3,
        n_heads: int = 8,
        dropout_clf: float = 0.4,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size

        # 1. Backbone CNN : EfficientNet-B4
        backbone = efficientnet_b4(weights=EfficientNet_B4_Weights.DEFAULT)
        # Supprimer le classifier final → garder seulement le feature extractor
        self.cnn = nn.Sequential(*list(backbone.children())[:-1])
        self.feat_dim = 1792  # Dimension de sortie d'EfficientNet-B4

        # Geler le backbone au début (fine-tuning progressif)
        if freeze_backbone:
            for param in self.cnn.parameters():
                param.requires_grad = False

        # Projection vers hidden_size avant le GRU
        self.input_proj = nn.Sequential(
            nn.Linear(self.feat_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # 2. GRU Bidirectionnel 
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=gru_layers,
            bidirectional=True,
            batch_first=True,
            dropout=gru_dropout if gru_layers > 1 else 0.0,
        )
        gru_output_dim = hidden_size * 2  # bidirectionnel → x2

        # 3. Multi-head Attention
        self.attn_norm = nn.LayerNorm(gru_output_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=gru_output_dim,
            num_heads=n_heads,
            dropout=0.1,
            batch_first=True,
        )

        # 4. Têtes de classification (une par état affectif)
        self.heads = nn.ModuleDict({
            state: nn.Sequential(
                nn.Linear(gru_output_dim, 128),
                nn.ReLU(),
                nn.Dropout(dropout_clf),
                nn.Linear(128, n_classes),
            )
            for state in self.AFFECTIVE_STATES
        })

    # Dégel progressif du backbone 
    def unfreeze_backbone(self, n_blocks: int = 2):
        """
        Dégèle les n_blocks derniers blocs d'EfficientNet pour le fine-tuning.
        Appeler après quelques epochs d'entraînement des couches supérieures.
        """
        children = list(self.cnn.children())
        for child in children[-n_blocks:]:
            for param in child.parameters():
                param.requires_grad = True
        print(f"[Model] {n_blocks} derniers blocs EfficientNet dégelés pour fine-tuning.")

    # Forward pass
    def forward(self, x: torch.Tensor) -> dict:
        """
        Args:
            x : Tensor de shape (Batch, T, C, H, W)
                B=batch, T=frames, C=canaux, H=hauteur, W=largeur

        Returns:
            dict : {"Boredom": logits, "Engagement": logits, ...}
                   Chaque logits est de shape (B, 4)
        """
        B, T, C, H, W = x.shape

        # Extraction CNN (frame par frame) 
        # Reshape (B*T, C, H, W) → CNN → Reshape (B, T, feat_dim)
        cnn_out = self.cnn(x.view(B * T, C, H, W))       # (B*T, 1792, 1, 1)
        cnn_out = cnn_out.view(B, T, self.feat_dim)       # (B, T, 1792)

        # Projection 
        proj = self.input_proj(cnn_out)                   # (B, T, hidden_size)

        #GRU Bidirectionnel
        gru_out, _ = self.gru(proj)                       # (B, T, hidden*2)

        # Multi-head Attention
        gru_norm = self.attn_norm(gru_out)
        attn_out, _ = self.attention(gru_norm, gru_norm, gru_norm)
        attn_out = attn_out + gru_out                     # Connexion résiduelle

        # Pooling temporel (moyenne)
        pooled = attn_out.mean(dim=1)                     # (B, hidden*2)

        # 4 Têtes de classification 
        return {state: head(pooled) for state, head in self.heads.items()}


# Weighted CrossEntropy Loss multi-tâche

class MultiTaskLoss(nn.Module):
    """
    Loss multi-tâche : somme pondérée des CrossEntropy pour chaque état affectif.
    Intègre class weights pour gérer le déséquilibre du dataset.

    Args:
        class_weights : dict {state: Tensor(4)} ou None.
        task_weights  : dict {state: float} pour pondérer les tâches entre elles.
        label_smoothing : float entre 0 et 0.2 (recommandé : 0.1).
    """

    STATES = ["Boredom", "Engagement", "Confusion", "Frustration"]

    def __init__(
        self,
        class_weights: dict = None,
        task_weights: dict = None,
        label_smoothing: float = 0.1,
    ):
        super().__init__()

        # Poids par état (pour le déséquilibre inter-tâches)
        self.task_weights = task_weights or {s: 1.0 for s in self.STATES}

        # CrossEntropy par état avec class weights
        self.criteria = nn.ModuleDict({
            state: nn.CrossEntropyLoss(
                weight=class_weights.get(state) if class_weights else None,
                label_smoothing=label_smoothing,
            )
            for state in self.STATES
        })

    def forward(self, outputs: dict, labels: torch.Tensor) -> tuple:
        """
        Args:
            outputs : dict de logits {state: (B, 4)}
            labels  : Tensor (B, 4) — colonnes dans l'ordre STATES

        Returns:
            total_loss, dict des losses individuelles
        """
        individual_losses = {}
        total_loss = 0.0

        for i, state in enumerate(self.STATES):
            loss = self.criteria[state](outputs[state], labels[:, i])
            individual_losses[state] = loss
            total_loss += self.task_weights[state] * loss

        return total_loss, individual_losses


# Utilitaire : compter les paramètres

def count_parameters(model: nn.Module) -> int:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Paramètres totaux     : {total:,}")
    print(f"Paramètres entraînables : {trainable:,}")
    return trainable
