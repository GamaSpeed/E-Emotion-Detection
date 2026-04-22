# EduSense — Détection d'États Affectifs en E-Learning

Détection automatique des états affectifs des apprenants (**Engagement, Ennui, Confusion, Frustration**) à partir d'un flux webcam en temps réel, en utilisant le dataset **DAiSEE** et un modèle **EfficientNet-B2 + GRU Bidirectionnel + Multi-head Attention**.

Déployé en production sur **Google Cloud Run** avec GPU **NVIDIA L4**.

---

## Résultats — Run 6 (modèle de production)

| État affectif | Accuracy | F1 Macro | Recall High | Notes |
|---------------|----------|----------|-------------|-------|
| Engagement    | 88 %     | 52 %     | **92 %**    | État majoritaire — excellente détection |
| Confusion     | 86 %     | 55 %     | 45 %        | Détection partielle acceptable |
| Frustration   | 95 %     | 52 %     | 4 %         | Limite structurelle — ratio 21,9:1 |
| Ennui         | 47 %     | 43 %     | 46 %        | Moins critique pédagogiquement |
| **Moyenne**   | **80,8 %** | **49,4 %** | —        | Mode binaire Low / High |

---

## Structure du Projet

```
E-Emotion-Detection/
│
├── notebooks/
│   ├── 01_data_exploration.ipynb       # Visualisation & statistiques DAiSEE
│   └── 02_training_evaluation.ipynb    # Entraînement, évaluation, export MLflow
│
├── src/
│   ├── dataset.py    # DAiSEEDataset — chargement, binarisation, oversampling, augmentations
│   ├── model.py      # EfficientNet-B2 + GRU Bidirectionnel + Multi-head Attention
│   ├── train.py      # MultiTaskFocalLoss, OneCycleLR, early stopping, mixup
│   ├── evaluate.py   # Accuracy, F1, Kappa, matrices de confusion
│   └── utils.py      # AverageMeter, set_seed, load_config, EarlyStopping
│
├── backend/
│   ├── main.py               # FastAPI — endpoints REST + WebSocket
│   ├── inference.py          # EmotionPredictor — buffer 16 frames, prédiction temps réel
│   ├── db/
│   │   ├── models.py         # SQLAlchemy ORM — users, sessions, predictions, alerts, logs
│   │   ├── schemas.py        # Pydantic — validation requêtes et réponses
│   │   ├── crud.py           # Opérations base de données
│   │   ├── auth.py           # JWT + bcrypt
│   │   └── routes_db.py      # Routes REST /auth, /sessions, /alerts, /export
│   └── ...
│
├── frontend/
│   └── src/
│       ├── StudentView.jsx   # Interface étudiant — webcam, jauges binaires
│       ├── TeacherView.jsx   # Dashboard enseignant — temps réel, alertes, historique
│       └── LoginPage.jsx     # Authentification JWT
│
├── configs/
│   └── config.yaml           # Hyperparamètres, chemins, paramètres d'entraînement
│
├── notebooks/outputs/
│   └── checkpoints/
│       └── best_model.pt     # Checkpoint Run 6 — 45,8 Mo
│
├── Dockerfile.backend         # pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime
├── Dockerfile.frontend        # Nginx + React/Vite
├── docker-compose.yml         # Déploiement local
├── nginx.conf                 # Proxy WebSocket + envsubst $PORT
├── requirements.txt           # numpy==1.26.4, bcrypt==4.0.1 fixés
└── README.md
```

---

## Dataset — DAiSEE

[DAiSEE](https://people.iith.ac.in/vineethnb/resources/daisee/index.html) — Dataset for Affective States in E-Environments (IIT Roorkee, 2016).

**9 068 clips vidéo · 112 sujets · 4 états · annotations sur 4 niveaux**

Placer le dataset à l'emplacement suivant :

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

### Déséquilibre des classes (jeu de test)

| État       | Ratio Low/High | Support Low | Support High |
|------------|---------------|-------------|--------------|
| Engagement | 0,05:1        | 88          | 1 696        |
| Ennui      | 3,4:1         | 1 407       | 377          |
| Confusion  | 9,8:1         | 1 627       | 157          |
| Frustration| **21,9:1**    | 1 704       | 80           |

> **Mode binaire** : les niveaux 0+1 sont fusionnés en Low, les niveaux 2+3 en High. Ce choix améliore le F1 macro de 26,8 % (4-classes) à 49,4 % et aligne la sortie sur le besoin pédagogique réel.

---

## Architecture du Modèle

```
Vidéo (16 frames · 224×224 px)
         │
         ▼
EfficientNet-B2  ──→  Features spatiales (1 408-dim / frame)
         │               freeze_backbone=True → unfreeze à l'epoch 10
         ▼
Projection  Linear(1408→256) + LayerNorm + ReLU + Dropout(0.2)
         │
         ▼
GRU Bidirectionnel  ──→  Dynamique temporelle (hidden=256 · 2 couches)
         │               Sortie : 512-dim (256 × 2 directions)
         ▼
Multi-head Attention (8 têtes · dropout=0.1 · connexion résiduelle)
         │
         ▼
Pooling temporel (moyenne sur 16 frames)
         │
         ▼
4 Têtes de classification indépendantes  Linear(512→128→2)
    ├── Boredom      →  Low / High
    ├── Engagement   →  Low / High
    ├── Confusion    →  Low / High
    └── Frustration  →  Low / High
```

---

## Stratégie d'Entraînement

### Fonction de perte — Focal Loss

```python
FL(p) = -α · (1 - p)^γ · log(p)
```

- **γ = 0,5** (Run 6) — dépriorisation modérée des exemples faciles
- **label_smoothing = 0,1** — réduit la sur-confiance, améliore la calibration sur les classes rares
- Implémentée dans `train.py` sous la classe `MultiTaskFocalLoss`

### Class Weights avec mécanisme CAPS

```python
w_i = clip(1 / count_i, 1/CAPS, 1.0)
```

Le paramètre CAPS plafonne le poids maximum pour éviter la divergence sur les classes ultra-minoritaires :

| État       | Ratio réel | CAPS (Run 6) | Effet |
|------------|-----------|-------------|-------|
| Ennui      | 3,4:1     | 3,0×        | Boost modéré |
| Engagement | 0,05:1    | **4,0×**    | Maintient Recall High à 92 % |
| Confusion  | 9,8:1     | 3,0×        | Boost modéré |
| Frustration| 21,9:1    | **4,0×**    | Plafond critique — au-delà l'entraînement diverge |

### Autres composants clés

| Composant | Configuration | Rôle |
|-----------|--------------|------|
| Optimiseur | AdamW — lr backbone ×0,1 | Fine-tuning différencié CNN / couches supérieures |
| Scheduler | OneCycleLR — warmup 15 % — cosinus | Montée rapide, descente douce |
| Accumulation | 4 steps (batch effectif = 32) | Stabilité gradient sous contrainte mémoire |
| Mixed Precision | AMP (GradScaler) | Réduit la VRAM, accélère le calcul |
| Gradient clipping | max_norm=1.0 | Prévient l'explosion des gradients (GRU) |
| Early stopping | patience=15 · sur F1 val | Surveille F1 hors Boredom (Engagement+Confusion+Frustration) |
| Mixup | alpha=0.2 · optionnel | Régularisation — désactivé en Run 6 |

---

## Expérimentation — Historique des Runs

| Run | Modifications principales | F1 moy. | Décision |
|-----|--------------------------|---------|----------|
| Run 2 | 4-classes, γ=0,5 | 26,8 % | Abandonné — passage en binaire |
| **Run 6** | **Binaire, γ=0,5, CAPS×4, lr=3×10⁻⁴** | **49,4 %** | **Référence — déployé** |
| Run 7 | CAPS×8, unfreeze=10 | 47,6 % | Régression |
| Run 8 | γ=1,0, smoothing=0,05, CAPS×2,5 | 48,4 % | < Run 6 |
| Run 9 | Confusion CAPS=2,5, Frustration CAPS=2,0 | 52 % val | Recall High Engagement effondré |
| Run 10 | Run 6 + Frustration CAPS=5× | 50,5 % | Instable (loss initiale 0,49) |
| Run 11 | Augmentations couleur | 47,4 % | Dégradation |

> **Enseignement clé** : le CAPS Engagement doit rester ≥ 4,0×. Réduire ce paramètre cause une régression systématique même lorsque d'autres hyperparamètres s'améliorent.

Suivi complet sur [DagsHub MLflow](https://dagshub.com/GamaSpeed/E-learning-Emotion-Detection).

---

## Installation

```bash
git clone https://github.com/GamaSpeed/E-Emotion-Detection.git
cd E-Emotion-Detection

# Environnement conda
conda create -n edusense python=3.10 -y
conda activate edusense
pip install -r requirements.txt
```

### Dépendances principales

```
torch==2.1.0
torchvision==0.16.0
numpy==1.26.4          # fixé — incompatibilité NumPy 2.x avec PyTorch 2.1.0
fastapi==0.115.0
uvicorn[standard]==0.30.0
websockets==12.0
sqlalchemy>=2.0
psycopg2-binary
python-jose[cryptography]
bcrypt==4.0.1          # fixé — incompatibilité passlib avec bcrypt ≥ 4.0
opencv-python-headless==4.10.0.84
pyyaml>=6.0
scikit-learn
mlflow
```

---

## Utilisation

### Entraînement

```bash
# 1. Configurer les chemins et hyperparamètres
# Éditer configs/config.yaml

# 2. Lancer via le notebook
jupyter notebook notebooks/02_training_evaluation.ipynb
```

Les runs sont automatiquement tracés sur DagsHub MLflow. Chaque run sauvegarde le meilleur checkpoint dans `notebooks/outputs/checkpoints/best_model.pt`.

### Inférence locale

```python
from src.model import EmotionModel
from backend.inference import EmotionPredictor

predictor = EmotionPredictor(
    model_path="notebooks/outputs/checkpoints/best_model.pt",
    model_class=EmotionModel,
    config=config,
    device="auto"  # cuda si disponible
)

# Ajouter des frames et obtenir une prédiction
ready = predictor.add_frame(frame_bgr)
if ready:
    result = predictor.predict()
    print(result["Engagement"])  # {"level": 1, "label": "Élevé", "confidence": 0.92}
```

---

## Déploiement

### Docker local

```bash
docker compose up --build
# Frontend : http://localhost:3000
# Backend  : http://localhost:8000
# Swagger  : http://localhost:8000/docs
```

### Production — Google Cloud Run (GPU L4)

```bash
# Build et push de l'image backend (CUDA)
docker build -f Dockerfile.backend -t us-central1-docker.pkg.dev/edusense-prod/edusense-repo/backend:latest .
docker push us-central1-docker.pkg.dev/edusense-prod/edusense-repo/backend:latest

# Déploiement avec GPU NVIDIA L4
gcloud run deploy edusense-backend \
  --image us-central1-docker.pkg.dev/edusense-prod/edusense-repo/backend:latest \
  --platform managed --region us-east4 \
  --gpu 1 --gpu-type nvidia-l4 \
  --memory 16Gi --cpu 4 --max-instances 1 \
  --set-env-vars DEVICE=cuda \
  --set-secrets DATABASE_URL=database-url:latest
```

**URLs de production :**
- Frontend : https://edusense-frontend-56uctf3xnq-uk.a.run.app
- Backend  : https://edusense-backend-728698401230.us-east4.run.app
- Health   : https://edusense-backend-728698401230.us-east4.run.app/health

### Stratégie de branches

| Branche | Usage | Spécificité |
|---------|-------|-------------|
| `main` | Entraînement ML, recherche | `weights=EfficientNet_B2_Weights.DEFAULT` |
| `production-gcp` | Déploiement Cloud Run | `weights=None` — poids dans le checkpoint |

---

## CI/CD — GitHub Actions

Le fichier `.github/workflows/cicd.yml` définit 4 jobs :

1. **test-backend** — lint flake8 + validation imports + test `/health`
2. **test-frontend** — `npm ci` + `vite build`
3. **build-and-push** — docker build + push Artifact Registry (déclenché sur `production-gcp` uniquement)
4. **deploy** — `gcloud run deploy` backend GPU + frontend + health check post-déploiement

Secret requis : `GCP_SA_KEY` (compte de service GCP avec rôles Cloud Run Admin, Artifact Registry Writer, Secret Manager Accessor, Cloud SQL Client).

---

## Notes techniques importantes

| Problème | Solution |
|----------|----------|
| `numpy._core` ImportError | Fixer `numpy==1.26.4` — NumPy 2.x incompatible avec PyTorch 2.1.0 |
| Hash error EfficientNet en production | Utiliser `weights=None` — les poids ImageNet sont dans le checkpoint |
| WebSocket coupé après 60s sur Cloud Run | Ping client toutes les 30s + `timeout=3600` sur le service |
| `bcrypt` incompatibilité passlib | Fixer `bcrypt==4.0.1` |
| `student_id` FOREIGN KEY violation | Déclarer `Optional[str] = None` dans `schemas.py` |
| `VITE_WS_URL` non injecté | Passer l'ARG Docker avant `npm run build` dans `Dockerfile.frontend` |

---

## Références

> Gupta, A., D'Cunha, A., Awasthi, K., & Balasubramanian, V. (2016). *DAiSEE: Towards User Engagement Recognition in the Wild*. arXiv:1609.01885.

> Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). *Focal Loss for Dense Object Detection*. IEEE ICCV.

> Tan, M., & Le, Q. V. (2019). *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks*. ICML 2019.

---

## Équipe

**Équipe Franklin** — Cité · Avril 2026

| Membre | Rôle |
|--------|------|
| Franklin | Chef de projet · ML & Déploiement |
| Joel | Backend · API & Base de données |
| Larissa | Frontend · Expérience utilisateur |
