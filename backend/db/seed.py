"""
seed.py — Créer les comptes de démonstration
À exécuter une seule fois après init_db() :
  python -m backend.db.seed
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from backend.db.database import SessionLocal, init_db
from backend.db import crud, schemas


DEMO_USERS = [
    {"email": "prof@edusense.ai",      "password": "prof123",     "name": "Dr. Martin",   "role": "teacher"},
    {"email": "etudiant@edusense.ai",  "password": "etudiant123", "name": "Alex Dupont",  "role": "student"},
    {"email": "alice@edusense.ai",     "password": "etudiant123", "name": "Alice Martin", "role": "student"},
    {"email": "baptiste@edusense.ai",  "password": "etudiant123", "name": "Baptiste Leroy","role": "student"},
    {"email": "camille@edusense.ai",   "password": "etudiant123", "name": "Camille Dubois","role": "student"},
]


def seed():
    print("[Seed] Initialisation de la base de données...")
    init_db()

    db = SessionLocal()
    created = 0
    try:
        for user_data in DEMO_USERS:
            existing = crud.get_user_by_email(db, user_data["email"])
            if existing:
                print(f"[Seed] Existe déjà : {user_data['email']}")
                continue
            user = crud.create_user(db, schemas.UserCreate(**user_data))
            print(f"[Seed] Créé : {user.role} — {user.email}")
            created += 1
    finally:
        db.close()

    print(f"\n[Seed] Terminé — {created} utilisateur(s) créé(s).")
    print("[Seed] Connexion prof    : prof@edusense.ai / prof123")
    print("[Seed] Connexion étudiant: etudiant@edusense.ai / etudiant123")


if __name__ == "__main__":
    seed()
