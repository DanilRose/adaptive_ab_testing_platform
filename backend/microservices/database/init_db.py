# backend/database/init_db.py

from __future__ import annotations

import logging

from sqlalchemy import text
from sqlalchemy.orm import Session

from backend.microservices.database import crud
from backend.microservices.database.models import UserORM
from backend.microservices.database.session import Base, engine
from backend.microservices.database.migration_script import migrate_ab_tests_table
from backend.microservices.auth_core.service import get_password_hash


logger = logging.getLogger(__name__)

_EXTRA_INDEXES = [
    "CREATE INDEX IF NOT EXISTS ix_ab_tests_status_created_at ON ab_tests (status, created_at)",
    "CREATE INDEX IF NOT EXISTS ix_ab_tests_dataset_id ON ab_tests (dataset_id)",
    "CREATE INDEX IF NOT EXISTS ix_generated_data_data_type_created_at ON generated_data (data_type, created_at)",
    "CREATE INDEX IF NOT EXISTS ix_test_sessions_test_id ON test_sessions (test_id)",
    "CREATE INDEX IF NOT EXISTS ix_test_sessions_user_id ON test_sessions (user_id)",
    "CREATE INDEX IF NOT EXISTS ix_ab_test_time_series_test_variant_users ON ab_test_time_series (test_id, variant, users_processed)",
    "CREATE INDEX IF NOT EXISTS ix_checkpoints_name_created_at ON checkpoints (name, created_at)",
]


def create_tables() -> None:
    Base.metadata.create_all(bind=engine, checkfirst=True)
    with engine.connect() as conn:
        for stmt in _EXTRA_INDEXES:
            conn.execute(text(stmt))
        conn.commit()


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

    # create_all() не изменяет уже существующие таблицы.
    # Догоняем эволюцию схемы (в т.ч. ab_tests.extra_config) через идемпотентные SQL-миграции.
    try:
        migrate_ab_tests_table()
    except Exception as exc:
        logger.warning("Schema migration step failed during bootstrap: %s", exc)

    seed_default_users()

    # Идемпотентная инициализация системных шаблонов:
    # - на чистой БД создаст все шаблоны из кода;
    # - на существующей БД синхронизирует только системные шаблоны (created_by='system').
    try:
        with Session(engine) as db:
            seeded_count = crud.seed_default_templates(db)
        logger.info("Default templates synchronized: %s", seeded_count)
    except Exception as exc:
        logger.warning("Default templates sync failed during bootstrap: %s", exc)

    logger.info("Database bootstrap completed")
