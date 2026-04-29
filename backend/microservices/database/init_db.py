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
        # Миграция legacy тест-аккаунтов:
        # analyst -> gubanovaaa, manager -> Ivanovii, удаление старого developer.
        analyst_user = db.query(UserORM).filter(UserORM.username == "analyst").first()
        if analyst_user and not db.query(UserORM).filter(UserORM.username == "gubanovaaa").first():
            analyst_user.username = "gubanovaaa"

        manager_user = db.query(UserORM).filter(UserORM.username == "manager").first()
        if manager_user and not db.query(UserORM).filter(UserORM.username == "Ivanovii").first():
            manager_user.username = "Ivanovii"

        legacy_developer_user = db.query(UserORM).filter(UserORM.username == "developer").first()
        if legacy_developer_user:
            db.delete(legacy_developer_user)

        defaults = [
            {
                "username": "timohinds",
                "role": "user",
                "full_name": "Разработчик",
                "job_title": "developer",
                "permissions": [
                    "Администрирование",
                    "GAN_менеджер_обучение",
                    "GAN_менеджер_генерация_данных",
                    "GAN_менеджер_редактирование",
                    "AB_тесты_создание",
                    "AB_тесты_управление",
                    "AB_тесты_удаление_и_архивация",
                    "Шаблоны_просмотр",
                    "Шаблоны_редактирование",
                    "Шаблоны_удаление",
                    "Просмотр_результатов_тестов",
                    "Экспорт_результатов",
                ],
                "password": "dev123",
            },
            {
                "username": "gubanovaaa",
                "role": "user",
                "full_name": "Аналитик",
                "job_title": "analyst",
                "permissions": [
                    "GAN_менеджер_генерация_данных",
                    "AB_тесты_создание",
                    "Просмотр_результатов_тестов",
                    "Шаблоны_просмотр",
                ],
                "password": "analyst123",
            },
            {
                "username": "Ivanovii",
                "role": "user",
                "full_name": "Проект-менеджер",
                "job_title": "project_manager",
                "permissions": [
                    "Просмотр_результатов_тестов",
                    "Экспорт_результатов",
                    "GAN_менеджер_обучение",
                ],
                "password": "manager123",
            },
        ]

        for item in defaults:
            existing = db.query(UserORM).filter(UserORM.username == item["username"]).first()
            if existing:
                # Для уже существующих пользователей НЕ перетираем персональные поля
                # (full_name и др.), иначе после рестарта Docker профиль «сбрасывается».
                # Поддерживаем только базовую совместимость роли и прав.
                existing.role = item["role"]
                existing.permissions_json = item.get("permissions", [])
                continue

            db.add(
                UserORM(
                    username=item["username"],
                    role=item["role"],
                    full_name=item["full_name"],
                    job_title=item.get("job_title"),
                    permissions_json=item.get("permissions", []),
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
