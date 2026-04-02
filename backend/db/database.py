"""
database.py — Connexion SQLAlchemy
Compatible SQLite (local/VirtualBox) et PostgreSQL (Google Cloud SQL)

Changer DATABASE_URL dans .env pour basculer entre les deux :
  SQLite    : sqlite:///./data/edusense.db
  PostgreSQL: postgresql://user:pass@host/edusense
"""

import os
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, DeclarativeBase

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./data/edusense.db")

# ── Options spécifiques selon le moteur ───────────────────────────────────────
connect_args = {}
if DATABASE_URL.startswith("sqlite"):
    # SQLite ne supporte pas le multithreading natif — nécessaire pour FastAPI
    connect_args["check_same_thread"] = False

engine = create_engine(
    DATABASE_URL,
    connect_args=connect_args,
    # Pool adapté : SQLite n'a pas besoin de pool, PostgreSQL oui
    pool_pre_ping=True,
)

# Activer les foreign keys sur SQLite (désactivées par défaut)
if DATABASE_URL.startswith("sqlite"):
    @event.listens_for(engine, "connect")
    def enable_sqlite_fk(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


def get_db():
    """Dépendance FastAPI — injecte une session DB dans chaque requête."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Crée toutes les tables si elles n'existent pas encore."""
    import os
    # Créer le répertoire data/ si SQLite
    if DATABASE_URL.startswith("sqlite"):
        db_path = DATABASE_URL.replace("sqlite:///", "")
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

    from backend.db import models  # noqa — importer pour enregistrer les modèles
    Base.metadata.create_all(bind=engine)
