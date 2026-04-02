"""
crud.py — Opérations base de données
Toutes les fonctions qui lisent ou écrivent dans la DB.
"""

import json
from datetime import datetime, timezone
from typing import Optional, List
from sqlalchemy.orm import Session
from sqlalchemy import desc
from passlib.context import CryptContext

from backend.db.models import User, Session as DBSession, Prediction, Alert, Log
from backend.db import schemas

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def utcnow():
    return datetime.now(timezone.utc)


# ── Auth ──────────────────────────────────────────────────────────────────────
def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


# ── Users ─────────────────────────────────────────────────────────────────────
def get_user_by_email(db: Session, email: str) -> Optional[User]:
    return db.query(User).filter(User.email == email).first()


def get_user_by_id(db: Session, user_id: str) -> Optional[User]:
    return db.query(User).filter(User.id == user_id).first()


def create_user(db: Session, data: schemas.UserCreate) -> User:
    user = User(
        email=data.email,
        hashed_password=hash_password(data.password),
        name=data.name,
        role=data.role,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def update_last_login(db: Session, user: User) -> User:
    user.last_login = utcnow()
    db.commit()
    db.refresh(user)
    return user


def get_all_students(db: Session) -> List[User]:
    return db.query(User).filter(User.role == "student", User.is_active == True).all()


# ── Sessions ──────────────────────────────────────────────────────────────────
def create_session(db: Session, student_id: str, client_id: str) -> DBSession:
    session = DBSession(student_id=student_id, client_id=client_id)
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


def close_session(db: Session, session_id: str, avg_score: Optional[float] = None) -> Optional[DBSession]:
    session = db.query(DBSession).filter(DBSession.id == session_id).first()
    if session:
        session.ended_at = utcnow()
        session.avg_engagement_score = avg_score
        db.commit()
        db.refresh(session)
    return session


def get_session_by_client_id(db: Session, client_id: str) -> Optional[DBSession]:
    return (db.query(DBSession)
            .filter(DBSession.client_id == client_id, DBSession.ended_at == None)
            .first())


def get_sessions_for_student(db: Session, student_id: str, limit: int = 20) -> List[DBSession]:
    return (db.query(DBSession)
            .filter(DBSession.student_id == student_id)
            .order_by(desc(DBSession.started_at))
            .limit(limit).all())


def get_all_sessions(db: Session, limit: int = 50) -> List[DBSession]:
    return (db.query(DBSession)
            .order_by(desc(DBSession.started_at))
            .limit(limit).all())


def get_active_sessions(db: Session) -> List[DBSession]:
    """Sessions actuellement ouvertes (ended_at IS NULL)."""
    return db.query(DBSession).filter(DBSession.ended_at == None).all()


# ── Predictions ───────────────────────────────────────────────────────────────
def save_prediction(db: Session, data: schemas.PredictionCreate) -> Prediction:
    pred = Prediction(**data.model_dump())
    db.add(pred)
    # Incrémenter le compteur de la session
    session = db.query(DBSession).filter(DBSession.id == data.session_id).first()
    if session:
        session.n_predictions += 1
    db.commit()
    db.refresh(pred)
    return pred


def get_predictions_for_session(db: Session, session_id: str) -> List[Prediction]:
    return (db.query(Prediction)
            .filter(Prediction.session_id == session_id)
            .order_by(Prediction.timestamp)
            .all())


def get_recent_predictions(db: Session, session_id: str, limit: int = 10) -> List[Prediction]:
    return (db.query(Prediction)
            .filter(Prediction.session_id == session_id)
            .order_by(desc(Prediction.timestamp))
            .limit(limit).all())


# ── Alerts ────────────────────────────────────────────────────────────────────
def create_alert(db: Session, session_id: str, state: str,
                 duration_s: Optional[float] = None) -> Alert:
    alert = Alert(session_id=session_id, state=state, level=1, duration_s=duration_s)
    db.add(alert)
    db.commit()
    db.refresh(alert)
    return alert


def acknowledge_alert(db: Session, alert_id: str) -> Optional[Alert]:
    alert = db.query(Alert).filter(Alert.id == alert_id).first()
    if alert:
        alert.acknowledged = True
        alert.acknowledged_at = utcnow()
        db.commit()
        db.refresh(alert)
    return alert


def get_alerts_for_session(db: Session, session_id: str) -> List[Alert]:
    return (db.query(Alert)
            .filter(Alert.session_id == session_id)
            .order_by(Alert.timestamp)
            .all())


def get_unacknowledged_alerts(db: Session) -> List[Alert]:
    return db.query(Alert).filter(Alert.acknowledged == False).all()


# ── Logs ──────────────────────────────────────────────────────────────────────
def write_log(db: Session, event_type: str,
              user_id: Optional[str] = None,
              details: Optional[dict] = None,
              ip_address: Optional[str] = None) -> Log:
    log = Log(
        user_id=user_id,
        event_type=event_type,
        details=json.dumps(details, ensure_ascii=False) if details else None,
        ip_address=ip_address,
    )
    db.add(log)
    db.commit()
    db.refresh(log)
    return log


def get_recent_logs(db: Session, limit: int = 100) -> List[Log]:
    return (db.query(Log)
            .order_by(desc(Log.timestamp))
            .limit(limit).all())


# ── Export session (rapport professeur) ───────────────────────────────────────
def get_session_history(db: Session, session_id: str) -> Optional[schemas.SessionHistory]:
    session = db.query(DBSession).filter(DBSession.id == session_id).first()
    if not session:
        return None
    return schemas.SessionHistory(
        session=session,
        predictions=get_predictions_for_session(db, session_id),
        alerts=get_alerts_for_session(db, session_id),
    )
