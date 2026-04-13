"""
schemas.py — Schémas Pydantic
Validation des entrées API et sérialisation des réponses.
"""

from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, EmailStr, field_validator


# ── Users ─────────────────────────────────────────────────────────────────────
class UserCreate(BaseModel):
    email:    EmailStr
    password: str
    name:     str
    role:     str  # "student" | "teacher"

    @field_validator("role")
    @classmethod
    def role_must_be_valid(cls, v):
        if v not in ("student", "teacher"):
            raise ValueError("role doit être 'student' ou 'teacher'")
        return v

    @field_validator("password")
    @classmethod
    def password_min_length(cls, v):
        if len(v) < 6:
            raise ValueError("Le mot de passe doit contenir au moins 6 caractères")
        return v


class UserOut(BaseModel):
    id:         str
    email:      str
    name:       str
    role:       str
    is_active:  bool
    created_at: datetime
    last_login: Optional[datetime] = None

    model_config = {"from_attributes": True}


# ── Auth ──────────────────────────────────────────────────────────────────────
class LoginRequest(BaseModel):
    email:    EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type:   str = "bearer"
    user:         UserOut


# ── Sessions ──────────────────────────────────────────────────────────────────
class SessionOut(BaseModel):
    id:                  str
    student_id:          Optional[str] = None
    client_id:           str
    started_at:          datetime
    ended_at:            Optional[datetime] = None
    n_predictions:       int
    avg_engagement_score: Optional[float] = None

    model_config = {"from_attributes": True}


class SessionWithStudent(SessionOut):
    student: UserOut


# ── Predictions ───────────────────────────────────────────────────────────────
class PredictionCreate(BaseModel):
    session_id:       str
    engagement:       int
    boredom:          int
    confusion:        int
    frustration:      int
    engagement_conf:  float
    boredom_conf:     float
    confusion_conf:   float
    frustration_conf: float
    engagement_score: int


class PredictionOut(PredictionCreate):
    id:        str
    timestamp: datetime

    model_config = {"from_attributes": True}


# ── Alerts ────────────────────────────────────────────────────────────────────
class AlertOut(BaseModel):
    id:              str
    session_id:      str
    timestamp:       datetime
    state:           str
    level:           int
    duration_s:      Optional[float] = None
    acknowledged:    bool
    acknowledged_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


# ── Logs ──────────────────────────────────────────────────────────────────────
class LogOut(BaseModel):
    id:         str
    timestamp:  datetime
    user_id:    Optional[str] = None
    event_type: str
    details:    Optional[str] = None
    ip_address: Optional[str] = None

    model_config = {"from_attributes": True}


# ── Historique session (vue professeur) ───────────────────────────────────────
class SessionHistory(BaseModel):
    session:     SessionWithStudent
    predictions: List[PredictionOut]
    alerts:      List[AlertOut]