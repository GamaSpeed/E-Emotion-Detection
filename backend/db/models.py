"""
models.py — Modèles SQLAlchemy (ORM)

Tables :
  users       → professeurs et étudiants
  sessions    → une session par connexion étudiant
  predictions → une ligne toutes les 500ms par étudiant actif
  alerts      → déclenchées quand un état critique est détecté
  logs        → événements système (connexions, erreurs, actions)
"""

import uuid
from datetime import datetime, timezone
from sqlalchemy import (
    Column, String, Integer, Float, Boolean,
    DateTime, ForeignKey, Enum as SAEnum, Text,
)
from sqlalchemy.orm import relationship
from backend.db.database import Base


def utcnow():
    return datetime.now(timezone.utc)


def new_uuid():
    return str(uuid.uuid4())


# ── Table : users ─────────────────────────────────────────────────────────────
class User(Base):
    """Professeurs et étudiants — authentification JWT."""
    __tablename__ = "users"

    id            = Column(String(36), primary_key=True, default=new_uuid)
    email         = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    name          = Column(String(100), nullable=False)
    role          = Column(SAEnum("student", "teacher", name="user_role"), nullable=False)
    is_active     = Column(Boolean, default=True, nullable=False)
    created_at    = Column(DateTime(timezone=True), default=utcnow, nullable=False)
    last_login    = Column(DateTime(timezone=True), nullable=True)

    # Relations
    sessions      = relationship("Session", back_populates="student",
                                 foreign_keys="Session.student_id")
    logs          = relationship("Log", back_populates="user")


# ── Table : sessions ──────────────────────────────────────────────────────────
class Session(Base):
    """
    Une session = une connexion WebSocket d'un étudiant.
    Ouverte au ws.onopen, fermée au ws.onclose.
    """
    __tablename__ = "sessions"

    id            = Column(String(36), primary_key=True, default=new_uuid)
    student_id = Column(String(36), ForeignKey("users.id"), nullable=True, index=True)
    client_id     = Column(String(100), nullable=False)   # ex: "student_1742000000"
    started_at    = Column(DateTime(timezone=True), default=utcnow, nullable=False)
    ended_at      = Column(DateTime(timezone=True), nullable=True)
    n_predictions = Column(Integer, default=0, nullable=False)
    avg_engagement_score = Column(Float, nullable=True)   # calculé à la fermeture

    # Relations
    student       = relationship("User", back_populates="sessions",
                                 foreign_keys=[student_id])
    predictions   = relationship("Prediction", back_populates="session",
                                 cascade="all, delete-orphan")
    alerts        = relationship("Alert", back_populates="session",
                                 cascade="all, delete-orphan")


# ── Table : predictions ───────────────────────────────────────────────────────
class Prediction(Base):
    """
    Une prédiction = résultat du modèle toutes les ~500ms.
    Stocke les 4 états en binaire (0=Low, 1=High) + confiances.
    """
    __tablename__ = "predictions"

    id              = Column(String(36), primary_key=True, default=new_uuid)
    session_id      = Column(String(36), ForeignKey("sessions.id"), nullable=False, index=True)
    timestamp       = Column(DateTime(timezone=True), default=utcnow, nullable=False, index=True)

    # États binaires (0=Low, 1=High)
    engagement      = Column(Integer, nullable=False)
    boredom         = Column(Integer, nullable=False)
    confusion       = Column(Integer, nullable=False)
    frustration     = Column(Integer, nullable=False)

    # Confiances (0.0 → 1.0)
    engagement_conf = Column(Float, nullable=False)
    boredom_conf    = Column(Float, nullable=False)
    confusion_conf  = Column(Float, nullable=False)
    frustration_conf = Column(Float, nullable=False)

    # Score d'engagement agrégé (0-100)
    engagement_score = Column(Integer, nullable=False)

    # Relation
    session         = relationship("Session", back_populates="predictions")


# ── Table : alerts ────────────────────────────────────────────────────────────
class Alert(Base):
    """
    Alerte déclenchée quand un état critique persiste.
    Ex : confusion=High pendant plus de 30 secondes.
    """
    __tablename__ = "alerts"

    id          = Column(String(36), primary_key=True, default=new_uuid)
    session_id  = Column(String(36), ForeignKey("sessions.id"), nullable=False, index=True)
    timestamp   = Column(DateTime(timezone=True), default=utcnow, nullable=False)
    state       = Column(SAEnum("confusion", "frustration", "boredom",
                                name="alert_state"), nullable=False)
    level       = Column(Integer, nullable=False)          # toujours 1 (High) pour l'instant
    duration_s  = Column(Float, nullable=True)             # durée de l'état avant alerte
    acknowledged = Column(Boolean, default=False)          # prof a vu l'alerte
    acknowledged_at = Column(DateTime(timezone=True), nullable=True)

    # Relation
    session     = relationship("Session", back_populates="alerts")


# ── Table : logs ──────────────────────────────────────────────────────────────
class Log(Base):
    """
    Événements système pour auditabilité.
    Ex : connexion, déconnexion, erreur modèle, login/logout.
    """
    __tablename__ = "logs"

    id          = Column(String(36), primary_key=True, default=new_uuid)
    timestamp   = Column(DateTime(timezone=True), default=utcnow, nullable=False, index=True)
    user_id     = Column(String(36), ForeignKey("users.id"), nullable=True, index=True)
    event_type  = Column(SAEnum(
        "login", "logout", "ws_connect", "ws_disconnect",
        "prediction_error", "model_load", "alert_triggered",
        name="log_event"
    ), nullable=False)
    details     = Column(Text, nullable=True)   # JSON libre pour les détails
    ip_address  = Column(String(45), nullable=True)

    # Relation
    user        = relationship("User", back_populates="logs")
