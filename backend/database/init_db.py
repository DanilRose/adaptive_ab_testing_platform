# backend/database/init_db.py

from __future__ import annotations

import logging

from sqlalchemy.orm import Session

from backend.database.models import UserORM
from backend.database.session import Base, engine
from backend.auth.service import get_password_hash


logger = logging.getLogger(__name__)


def create_tables() -> None:
    Base.metadata.create_all(bind=engine)


def seed_default_users() -> None:
    with Session(engine) as db:
        defaults = [
            {
                "username": "developer",
                "role": "developer",
                "full_name": "Разработчик",
                "password": "dev123",
            },
            {
                "username": "analyst",
                "role": "analyst",
                "full_name": "Аналитик",
                "password": "analyst123",
            },
            {
                "username": "manager",
                "role": "manager",
                "full_name": "Проект-менеджер",
                "password": "manager123",
            },
        ]

        for item in defaults:
            existing = db.query(UserORM).filter(UserORM.username == item["username"]).first()
            if existing:
                continue
            db.add(
                UserORM(
                    username=item["username"],
                    role=item["role"],
                    full_name=item["full_name"],
                    hashed_password=get_password_hash(item["password"]),
                )
            )

        db.commit()


def bootstrap_database() -> None:
    create_tables()
    seed_default_users()
    logger.info("Database bootstrap completed")
