"""
routes_db.py — Endpoints REST pour la base de données
À inclure dans main.py avec : app.include_router(router)

Endpoints :
  POST /auth/register       → créer un compte
  POST /auth/login          → obtenir un token JWT
  GET  /users/me            → profil courant
  GET  /students            → liste étudiants (prof seulement)
  GET  /sessions            → toutes les sessions (prof)
  GET  /sessions/{id}       → détail session + prédictions + alertes
  GET  /sessions/{id}/export → CSV pour le prof
  GET  /alerts/pending      → alertes non acquittées
  POST /alerts/{id}/ack     → acquitter une alerte
  GET  /logs                → logs système (prof)
"""

import csv
import io
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from backend.db.database import get_db
from backend.db import crud, schemas
from backend.db.auth import get_current_user, require_teacher
from backend.db.models import User

router = APIRouter()


# ── Auth ──────────────────────────────────────────────────────────────────────
@router.post("/auth/register", response_model=schemas.UserOut, status_code=201)
def register(data: schemas.UserCreate, db: Session = Depends(get_db)):
    if crud.get_user_by_email(db, data.email):
        raise HTTPException(status_code=400, detail="Email déjà utilisé")
    return crud.create_user(db, data)


@router.post("/auth/login", response_model=schemas.TokenResponse)
def login(data: schemas.LoginRequest, db: Session = Depends(get_db)):
    user = crud.get_user_by_email(db, data.email)
    if not user or not crud.verify_password(data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Email ou mot de passe incorrect")
    if not user.is_active:
        raise HTTPException(status_code=403, detail="Compte désactivé")

    from backend.db.auth import create_access_token
    token = create_access_token(user.id, user.role)
    crud.update_last_login(db, user)
    crud.write_log(db, "login", user_id=user.id)
    return schemas.TokenResponse(
        access_token=token,
        user=schemas.UserOut.model_validate(user),
    )


# ── Users ─────────────────────────────────────────────────────────────────────
@router.get("/users/me", response_model=schemas.UserOut)
def get_me(current_user: User = Depends(get_current_user)):
    return current_user


@router.get("/students", response_model=list[schemas.UserOut])
def list_students(
    db: Session = Depends(get_db),
    _: User = Depends(require_teacher),
):
    return crud.get_all_students(db)


# ── Sessions ──────────────────────────────────────────────────────────────────
@router.get("/sessions", response_model=list[schemas.SessionOut])
def list_sessions(
    db: Session = Depends(get_db),
    _: User = Depends(require_teacher),
):
    return crud.get_all_sessions(db)


@router.get("/sessions/active", response_model=list[schemas.SessionOut])
def active_sessions(
    db: Session = Depends(get_db),
    _: User = Depends(require_teacher),
):
    return crud.get_active_sessions(db)


@router.get("/sessions/{session_id}", response_model=schemas.SessionHistory)
def session_detail(
    session_id: str,
    db: Session = Depends(get_db),
    _: User = Depends(require_teacher),
):
    history = crud.get_session_history(db, session_id)
    if not history:
        raise HTTPException(status_code=404, detail="Session introuvable")
    return history


@router.get("/sessions/{session_id}/export")
def export_session_csv(
    session_id: str,
    db: Session = Depends(get_db),
    _: User = Depends(require_teacher),
):
    """Exporte l'historique complet d'une session en CSV."""
    history = crud.get_session_history(db, session_id)
    if not history:
        raise HTTPException(status_code=404, detail="Session introuvable")

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "timestamp", "engagement", "boredom", "confusion", "frustration",
        "engagement_conf", "boredom_conf", "confusion_conf", "frustration_conf",
        "engagement_score",
    ])
    for pred in history.predictions:
        writer.writerow([
            pred.timestamp.isoformat(),
            pred.engagement, pred.boredom, pred.confusion, pred.frustration,
            round(pred.engagement_conf, 3), round(pred.boredom_conf, 3),
            round(pred.confusion_conf, 3), round(pred.frustration_conf, 3),
            pred.engagement_score,
        ])

    output.seek(0)
    filename = f"session_{session_id[:8]}_{history.session.started_at.strftime('%Y%m%d')}.csv"
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


# ── Alerts ────────────────────────────────────────────────────────────────────
@router.get("/alerts/pending", response_model=list[schemas.AlertOut])
def pending_alerts(
    db: Session = Depends(get_db),
    _: User = Depends(require_teacher),
):
    return crud.get_unacknowledged_alerts(db)


@router.post("/alerts/{alert_id}/ack", response_model=schemas.AlertOut)
def acknowledge(
    alert_id: str,
    db: Session = Depends(get_db),
    _: User = Depends(require_teacher),
):
    alert = crud.acknowledge_alert(db, alert_id)
    if not alert:
        raise HTTPException(status_code=404, detail="Alerte introuvable")
    return alert


# ── Logs ──────────────────────────────────────────────────────────────────────
@router.get("/logs", response_model=list[schemas.LogOut])
def system_logs(
    limit: int = 100,
    db: Session = Depends(get_db),
    _: User = Depends(require_teacher),
):
    return crud.get_recent_logs(db, limit)
