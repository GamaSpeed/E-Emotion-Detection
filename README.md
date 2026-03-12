# 🎓 DAiSEE — Emotion Detection for E-Learning

Détection automatique des états affectifs des apprenants (Engagement, Ennui, Confusion, Frustration) à partir de vidéos, en utilisant le dataset **DAiSEE** et un modèle **EfficientNet-B4 + GRU Bidirectionnel**.

---

## 📁 Structure du Projet

```
daisee_project/
│
├── 📓 notebooks/
│   ├── 01_data_exploration.ipynb     # Visualisation & prétraitement
│   └── 02_training_evaluation.ipynb  # Entraînement & évaluation
│
├── 🐍 src/
│   ├── dataset.py                    # DAiSEEDataset class
│   ├── model.py                      # EfficientNet-B4 + GRU model
│   ├── train.py                      # Boucle d'entraînement
│   ├── evaluate.py                   # Métriques & évaluation
│   └── utils.py                      # Fonctions utilitaires
│
├── ⚙️ configs/
│   └── config.yaml                   # Hyperparamètres & chemins
│
├── 📊 outputs/
│   ├── checkpoints/                  # Poids du modèle sauvegardés
│   ├── logs/                         # Logs d'entraînement
│   └── figures/                      # Graphiques & visualisations
│
├── requirements.txt
└── README.md
```

---

## 🗂️ Dataset

[DAiSEE](https://people.iith.ac.in/vineethnb/resources/daisee/index.html) — Dataset for Affective States in E-Environments.

Placer le dataset dans le dossier suivant :
```
D:\Facial_Emotion_Detection\DAiSEE\
├── DataSet/
│   ├── Train/
│   ├── Validation/
│   └── Test/
└── Labels/
    ├── TrainLabels.csv
    ├── ValidationLabels.csv
    └── TestLabels.csv
```

---

## 🏗️ Modèle

```
Vidéo (16 frames)
      ↓
EfficientNet-B4  →  Features spatiales (1792-dim)
      ↓
GRU Bidirectionnel  →  Dynamique temporelle
      ↓
Multi-head Attention
      ↓
4 Têtes de Classification
├── Boredom     (Very Low / Low / High / Very High)
├── Engagement  (Very Low / Low / High / Very High)
├── Confusion   (Very Low / Low / High / Very High)
└── Frustration (Very Low / Low / High / Very High)
```

---

## 📈 Résultats Baselines (papier DAiSEE)

| État Affectif | InceptionNet | LRCN | **Notre Modèle** |
|---------------|-------------|------|-----------------|
| Engagement    | 47.1%       | 57.9% | **~68%** (cible) |
| Boredom       | 36.5%       | 53.7% | **~62%** (cible) |
| Confusion     | 70.3%       | 72.3% | **~78%** (cible) |
| Frustration   | 78.3%       | 73.5% | **~83%** (cible) |

---

## ⚙️ Installation

```bash
git clone https://github.com/ton-username/daisee-emotion.git
cd daisee-emotion

conda create -n daisee python=3.10 -y
conda activate daisee

pip install -r requirements.txt
```

---

## 🚀 Utilisation

### 1. Exploration des données
Ouvrir `notebooks/01_data_exploration.ipynb`

### 2. Entraînement & Évaluation
Ouvrir `notebooks/02_training_evaluation.ipynb`

---

## 📦 Dépendances principales

- PyTorch 2.x + CUDA
- torchvision, timm
- OpenCV, facenet-pytorch
- albumentations
- scikit-learn, matplotlib, seaborn

---

## 📄 Référence

> Gupta et al., *DAiSEE: Towards User Engagement Recognition in the Wild*, arXiv:1609.01885, 2022.
