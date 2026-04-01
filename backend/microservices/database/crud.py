from __future__ import annotations

from typing import Any, Optional

from sqlalchemy import func, select, update, delete
from sqlalchemy.orm import Session
from sqlalchemy.ext.asyncio import AsyncSession

from backend.microservices.database.models import (
    ABTestORM,
    CheckpointORM,
    GeneratedDataORM,
    UserORM,
    ABTestTimeSeriesORM,
    TemplateORM,
    AssignmentAuditORM,
    UserAssignmentORM,
    MetricEventORM,
)


def get_user_by_username(db: Session, username: str) -> Optional[UserORM]:
    return db.query(UserORM).filter(UserORM.username == username).first()


def create_ab_test(
    db: Session,
    *,
    test_id: str,
    test_name: str,
    description: Optional[str],
    variants: list[str],
    primary_metric: str,
    metric_type: str,
    sample_size: Optional[int],
    confidence_level: float,
    power: float,
    min_effect_size: float,
    created_by_user_id: Optional[int],
    dataset_id: Optional[int] = None,
    real_world_duration_days: int = 14,
    simulation_duration_minutes: int = 20,
    traffic_split_type: str = "fixed",
    analysis_mode: str = "fixed_experiment",
    status: str = "prepared",  # Изменено с "active" на "prepared"
    extra_config: Optional[dict] = None,
    guardrails_config: Optional[dict] = None,
    analysis_validity: str = "valid_for_inference",
    do_commit: bool = True,
) -> ABTestORM:
    # min_effect_size приходит в долях (0.1 = 10%), mde_percent хранится в процентах.
    mde_percent = float(min_effect_size) * 100.0

    entity = ABTestORM(
        test_id=test_id,
        test_name=test_name,
        description=description,
        status=status,
        dataset_id=dataset_id,
        variants=variants,
        primary_metric=primary_metric,
        metric_type=metric_type,
        sample_size=sample_size,
        confidence_level=confidence_level,
        power=power,
        min_effect_size=min_effect_size,
        mde_percent=mde_percent,
        real_world_duration_days=real_world_duration_days,
        simulation_duration_minutes=simulation_duration_minutes,
        traffic_split_type=traffic_split_type,
        created_by_user_id=created_by_user_id,
        analysis_mode=analysis_mode,
        extra_config=extra_config,
        guardrails_config=guardrails_config,
        analysis_validity=analysis_validity,
    )
    db.add(entity)
    if do_commit:
        db.commit()
        db.refresh(entity)
    return entity


def list_ab_tests(db: Session, limit: int = 100) -> list[ABTestORM]:
    return db.query(ABTestORM).order_by(ABTestORM.created_at.desc()).limit(limit).all()


def get_ab_test_by_test_id(db: Session, test_id: str) -> Optional[ABTestORM]:
    return db.query(ABTestORM).filter(ABTestORM.test_id == test_id).first()


def update_ab_test_status(db: Session, test_id: str, status: str) -> Optional[ABTestORM]:
    entity = get_ab_test_by_test_id(db, test_id)
    if not entity:
        return None
    entity.status = status
    db.commit()
    db.refresh(entity)
    return entity


def create_generated_data(
    db: Session,
    *,
    data_type: str,
    sample_count: int,
    file_path: Optional[str],
    schema_json: Optional[dict[str, Any]],
    preview_json: Optional[list[dict[str, Any]]],
    extra_metadata: Optional[dict[str, Any]],
    ab_test_id: Optional[int] = None,
    do_commit: bool = True,
) -> GeneratedDataORM:
    entity = GeneratedDataORM(
        ab_test_id=ab_test_id,
        data_type=data_type,
        sample_count=sample_count,
        file_path=file_path,
        schema_json=schema_json,
        preview_json=preview_json,
        extra_metadata=extra_metadata,
    )
    db.add(entity)
    if do_commit:
        db.commit()
        db.refresh(entity)
    return entity


def list_generated_data(db: Session, limit: int = 100) -> list[GeneratedDataORM]:
    return db.query(GeneratedDataORM).order_by(GeneratedDataORM.created_at.desc()).limit(limit).all()


def get_latest_generated_data_by_type(db: Session, data_type: str) -> Optional[GeneratedDataORM]:
    return (
        db.query(GeneratedDataORM)
        .filter(GeneratedDataORM.data_type == data_type)
        .order_by(GeneratedDataORM.created_at.desc())
        .first()
    )


def upsert_checkpoint(
    db: Session,
    *,
    name: str,
    file_path: str,
    version: Optional[str] = None,
    epoch: Optional[int] = None,
    metrics_json: Optional[dict[str, Any]] = None,
    ab_test_id: Optional[int] = None,
    do_commit: bool = True,
) -> CheckpointORM:
    checkpoint = db.query(CheckpointORM).filter(CheckpointORM.file_path == file_path).first()
    if checkpoint is None:
        checkpoint = CheckpointORM(
            name=name,
            file_path=file_path,
            version=version,
            epoch=epoch,
            metrics_json=metrics_json,
            ab_test_id=ab_test_id,
        )
        db.add(checkpoint)
    else:
        checkpoint.name = name
        checkpoint.version = version
        checkpoint.epoch = epoch
        checkpoint.metrics_json = metrics_json
        checkpoint.ab_test_id = ab_test_id

    if do_commit:
        db.commit()
        db.refresh(checkpoint)
    return checkpoint


def list_checkpoints(db: Session, limit: int = 100, *, only_with_binary: bool = False, exclude_binary: bool = False) -> list[CheckpointORM]:
    query = db.query(CheckpointORM)
    if only_with_binary:
        query = query.filter(CheckpointORM.file_path.isnot(None), CheckpointORM.file_path != "")
    return query.order_by(CheckpointORM.created_at.desc()).limit(limit).all()


def get_checkpoint_by_name(db: Session, name: str) -> Optional[CheckpointORM]:
    return db.query(CheckpointORM).filter(CheckpointORM.name == name).first()


def get_checkpoint_by_file_path(db: Session, file_path: str) -> Optional[CheckpointORM]:
    return db.query(CheckpointORM).filter(CheckpointORM.file_path == file_path).first()


def get_test(db: Session, test_id: str) -> Optional[ABTestORM]:
    return get_ab_test_by_test_id(db, test_id)


def update_test_status(db: Session, test_id: str, status: str) -> Optional[ABTestORM]:
    return update_ab_test_status(db, test_id, status)


def update_test_simulation_status(db: Session, test_id: str, simulation_status: str) -> Optional[ABTestORM]:
    """Обновляет статус симуляции теста"""
    entity = get_ab_test_by_test_id(db, test_id)
    if not entity:
        return None
    entity.simulation_status = simulation_status
    db.commit()
    db.refresh(entity)
    return entity


def get_tests_by_status(db: Session, status: str, limit: int = 100) -> list[ABTestORM]:
    """Получает тесты по статусу"""
    return db.query(ABTestORM).filter(ABTestORM.status == status).order_by(ABTestORM.created_at.desc()).limit(limit).all()


def get_all_tests(db: Session, limit: int = 100) -> list[ABTestORM]:
    """Получает все тесты"""
    return db.query(ABTestORM).order_by(ABTestORM.created_at.desc()).limit(limit).all()


def get_all_checkpoints(db: Session) -> list[CheckpointORM]:
    return db.query(CheckpointORM).order_by(CheckpointORM.created_at.desc()).all()


def get_generated_data_by_id(db: Session, item_id: int) -> Optional[GeneratedDataORM]:
    return db.query(GeneratedDataORM).filter(GeneratedDataORM.id == item_id).first()


def delete_generated_data_by_id(db: Session, item_id: int) -> bool:
    entity = get_generated_data_by_id(db, item_id)
    if entity is None:
        return False
    db.delete(entity)
    db.commit()
    return True


def delete_checkpoint_by_id(db: Session, checkpoint_id: int) -> bool:
    checkpoint = db.query(CheckpointORM).filter(CheckpointORM.id == checkpoint_id).first()
    if checkpoint is None:
        return False
    db.delete(checkpoint)
    db.commit()
    return True


# ============================================================================
# АСИНХРОННЫЕ CRUD ФУНКЦИИ
# ============================================================================

async def async_get_user_by_username(db: AsyncSession, username: str) -> Optional[UserORM]:
    result = await db.execute(select(UserORM).filter(UserORM.username == username))
    return result.scalar_one_or_none()


async def async_list_ab_tests(db: AsyncSession, limit: int = 100) -> list[ABTestORM]:
    result = await db.execute(
        select(ABTestORM).order_by(ABTestORM.created_at.desc()).limit(limit)
    )
    return list(result.scalars().all())


async def async_get_ab_test_by_test_id(db: AsyncSession, test_id: str) -> Optional[ABTestORM]:
    result = await db.execute(select(ABTestORM).filter(ABTestORM.test_id == test_id))
    return result.scalar_one_or_none()


async def async_list_generated_data(db: AsyncSession, limit: int = 100) -> list[GeneratedDataORM]:
    result = await db.execute(
        select(GeneratedDataORM).order_by(GeneratedDataORM.created_at.desc()).limit(limit)
    )
    return list(result.scalars().all())


async def async_get_latest_generated_data_by_type(db: AsyncSession, data_type: str) -> Optional[GeneratedDataORM]:
    result = await db.execute(
        select(GeneratedDataORM)
        .filter(GeneratedDataORM.data_type == data_type)
        .order_by(GeneratedDataORM.created_at.desc())
        .limit(1)
    )
    return result.scalar_one_or_none()


async def async_get_generated_data_by_id(db: AsyncSession, item_id: int) -> Optional[GeneratedDataORM]:
    result = await db.execute(select(GeneratedDataORM).filter(GeneratedDataORM.id == item_id))
    return result.scalar_one_or_none()


async def async_list_checkpoints(db: AsyncSession, limit: int = 100, *, only_with_binary: bool = False) -> list[CheckpointORM]:
    query = select(CheckpointORM)
    if only_with_binary:
        query = query.where(CheckpointORM.file_path.is_not(None), CheckpointORM.file_path != "")
    query = query.order_by(CheckpointORM.created_at.desc()).limit(limit)
    result = await db.execute(query)
    return list(result.scalars().all())


async def async_delete_generated_data_by_id(db: AsyncSession, item_id: int) -> bool:
    result = await db.execute(delete(GeneratedDataORM).where(GeneratedDataORM.id == item_id))
    await db.commit()
    return result.rowcount > 0


async def async_delete_checkpoint_by_id(db: AsyncSession, checkpoint_id: int) -> bool:
    result = await db.execute(delete(CheckpointORM).where(CheckpointORM.id == checkpoint_id))
    await db.commit()
    return result.rowcount > 0


async def async_create_generated_data(
    db: AsyncSession,
    *,
    data_type: str,
    sample_count: int,
    file_path: Optional[str],
    schema_json: Optional[dict[str, Any]],
    preview_json: Optional[list[dict[str, Any]]],
    extra_metadata: Optional[dict[str, Any]],
    ab_test_id: Optional[int] = None,
) -> GeneratedDataORM:
    """Асинхронная версия создания записи generated_data"""
    entity = GeneratedDataORM(
        ab_test_id=ab_test_id,
        data_type=data_type,
        sample_count=sample_count,
        file_path=file_path,
        schema_json=schema_json,
        preview_json=preview_json,
        extra_metadata=extra_metadata,
    )
    db.add(entity)
    await db.commit()
    await db.refresh(entity)
    return entity


# ============================================================================
# CRUD функции для ABTestTimeSeriesORM (временные ряды A/B тестов)
# ============================================================================

def create_ab_test_time_series(
    db: Session,
    *,
    test_id: str,
    users_processed: int,
    variant: str,
    cumulative_metric: float,
    mean_metric: float,
    sample_size: int,
    p_value: Optional[float] = None,
    confidence_interval_lower: Optional[float] = None,
    confidence_interval_upper: Optional[float] = None,
    do_commit: bool = True,
) -> ABTestTimeSeriesORM:
    """Создает запись временного ряда для A/B теста"""
    entity = ABTestTimeSeriesORM(
        test_id=test_id,
        users_processed=users_processed,
        variant=variant,
        cumulative_metric=cumulative_metric,
        mean_metric=mean_metric,
        sample_size=sample_size,
        p_value=p_value,
        confidence_interval_lower=confidence_interval_lower,
        confidence_interval_upper=confidence_interval_upper,
    )
    db.add(entity)
    if do_commit:
        db.commit()
        db.refresh(entity)
    return entity


def get_ab_test_time_series(
    db: Session,
    test_id: str,
    limit: int = 1000
) -> list[ABTestTimeSeriesORM]:
    """Получает временные ряды для теста"""
    return (
        db.query(ABTestTimeSeriesORM)
        .filter(ABTestTimeSeriesORM.test_id == test_id)
        .order_by(ABTestTimeSeriesORM.users_processed)
        .limit(limit)
        .all()
    )


async def async_create_ab_test_time_series(
    db: AsyncSession,
    *,
    test_id: str,
    users_processed: int,
    variant: str,
    cumulative_metric: float,
    mean_metric: float,
    sample_size: int,
    p_value: Optional[float] = None,
    confidence_interval_lower: Optional[float] = None,
    confidence_interval_upper: Optional[float] = None,
) -> ABTestTimeSeriesORM:
    """Асинхронная версия создания записи временного ряда"""
    entity = ABTestTimeSeriesORM(
        test_id=test_id,
        users_processed=users_processed,
        variant=variant,
        cumulative_metric=cumulative_metric,
        mean_metric=mean_metric,
        sample_size=sample_size,
        p_value=p_value,
        confidence_interval_lower=confidence_interval_lower,
        confidence_interval_upper=confidence_interval_upper,
    )
    db.add(entity)
    await db.commit()
    await db.refresh(entity)
    return entity


# ============================================================================
# CRUD функции для шаблонов (TemplateORM)
# ============================================================================

def create_template(
    db: Session,
    *,
    name: str,
    description: Optional[str],
    template_type: str,
    config_json: dict[str, Any],
    tags: Optional[list[str]] = None,
    created_by: Optional[str] = None,
    do_commit: bool = True,
) -> TemplateORM:
    """Создание нового шаблона"""
    entity = TemplateORM(
        name=name,
        description=description,
        template_type=template_type,
        config_json=config_json,
        tags=tags,
        created_by=created_by,
    )
    db.add(entity)
    if do_commit:
        db.commit()
        db.refresh(entity)
    return entity


def list_templates(
    db: Session,
    template_type: Optional[str] = None,
    limit: int = 100,
) -> list[TemplateORM]:
    """Список шаблонов, опционально фильтрованный по типу"""
    query = db.query(TemplateORM)
    if template_type:
        query = query.filter(TemplateORM.template_type == template_type)
    return query.order_by(TemplateORM.updated_at.desc()).limit(limit).all()


def get_template_by_id(db: Session, template_id: int) -> Optional[TemplateORM]:
    """Получение шаблона по ID"""
    return db.query(TemplateORM).filter(TemplateORM.id == template_id).first()


def update_template(
    db: Session,
    template_id: int,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    config_json: Optional[dict[str, Any]] = None,
    tags: Optional[list[str]] = None,
    do_commit: bool = True,
) -> Optional[TemplateORM]:
    """Обновление шаблона"""
    entity = get_template_by_id(db, template_id)
    if not entity:
        return None
    if name is not None:
        entity.name = name
    if description is not None:
        entity.description = description
    if config_json is not None:
        entity.config_json = config_json
    if tags is not None:
        entity.tags = tags
    if do_commit:
        db.commit()
        db.refresh(entity)
    return entity


def delete_template_by_id(db: Session, template_id: int) -> bool:
    """Удаление шаблона по ID"""
    entity = get_template_by_id(db, template_id)
    if not entity:
        return False
    db.delete(entity)
    db.commit()
    return True


def seed_default_templates(db: Session) -> int:
    """Создаёт стандартные шаблоны, если их ещё нет в БД. Возвращает кол-во созданных."""
    existing = db.query(TemplateORM).count()
    if existing > 0:
        return 0

    defaults = [
        # --- GAN конфиги ---
        TemplateORM(
            name="GAN — Базовая конфигурация",
            description="Стандартная конфигурация GAN для большинства задач. 50 эпох, WGAN-GP режим включён.",
            template_type="gan_config",
            tags=["базовый", "wgan-gp"],
            created_by="system",
            config_json={
                "epochs": 50,
                "real_data_samples": 50000,
                "save_checkpoint": True,
                "checkpoint_name": "baseline_wgan",
                "LATENT_DIM": 128,
                "BATCH_SIZE": 256,
                "LEARNING_RATE": 0.0002,
                "DROPOUT_RATE": 0.3,
                "LAMBDA_GP": 10,
                "N_CRITIC": 5,
                "GENERATOR_LAYERS": "256,512,256",
                "DISCRIMINATOR_LAYERS": "256,512,256",
                "USE_WGAN_GP": True,
            },
        ),
        TemplateORM(
            name="GAN — Быстрое обучение",
            description="Упрощённая конфигурация для быстрого обучения и прототипирования. 20 эпох, меньший батч.",
            template_type="gan_config",
            tags=["быстрый", "прототип"],
            created_by="system",
            config_json={
                "epochs": 20,
                "real_data_samples": 20000,
                "save_checkpoint": True,
                "checkpoint_name": "fast_proto",
                "LATENT_DIM": 64,
                "BATCH_SIZE": 128,
                "LEARNING_RATE": 0.0003,
                "DROPOUT_RATE": 0.2,
                "LAMBDA_GP": 10,
                "N_CRITIC": 3,
                "GENERATOR_LAYERS": "128,256,128",
                "DISCRIMINATOR_LAYERS": "128,256,128",
                "USE_WGAN_GP": True,
            },
        ),
        TemplateORM(
            name="GAN — Высокое качество",
            description="Максимальное качество генерации. 100 эпох, глубокая сеть. Долго обучается.",
            template_type="gan_config",
            tags=["высокое-качество", "production"],
            created_by="system",
            config_json={
                "epochs": 100,
                "real_data_samples": 100000,
                "save_checkpoint": True,
                "checkpoint_name": "high_quality_gan",
                "LATENT_DIM": 256,
                "BATCH_SIZE": 512,
                "LEARNING_RATE": 0.0001,
                "DROPOUT_RATE": 0.4,
                "LAMBDA_GP": 10,
                "N_CRITIC": 5,
                "GENERATOR_LAYERS": "512,1024,512,256",
                "DISCRIMINATOR_LAYERS": "512,1024,512,256",
                "USE_WGAN_GP": True,
            },
        ),
        # --- Синтетические данные ---
        TemplateORM(
            name="Синтетика — Мобильные пользователи (iOS)",
            description="10 000 пользователей iOS из Москвы и Санкт-Петербурга. Подходит для мобильных UI/UX экспериментов.",
            template_type="synthetic_data",
            tags=["мобильные", "ios", "москва", "спб"],
            created_by="system",
            config_json={
                "num_users": 10000,
                "evaluation_metrics": True,
                "dataset_name": "mobile_ios_users",
                "filters": {
                    "devices": ["Mobile"],
                    "os": ["iOS"],
                    "cities": ["Москва", "Санкт-Петербург"],
                },
            },
        ),
        TemplateORM(
            name="Синтетика — Десктоп пользователи (Windows)",
            description="15 000 desktop-пользователей Windows. Подходит для тестов веб-интерфейса и desktop-функций.",
            template_type="synthetic_data",
            tags=["десктоп", "windows"],
            created_by="system",
            config_json={
                "num_users": 15000,
                "evaluation_metrics": True,
                "dataset_name": "desktop_windows_users",
                "filters": {
                    "devices": ["Desktop"],
                    "os": ["Windows"],
                },
            },
        ),
        TemplateORM(
            name="Синтетика — Платёжеспособные пользователи",
            description="8 000 пользователей с высоким доходом и email-подпиской. Для проверки monetization-гипотез.",
            template_type="synthetic_data",
            tags=["premium", "высокий-доход", "email"],
            created_by="system",
            config_json={
                "num_users": 8000,
                "evaluation_metrics": True,
                "dataset_name": "premium_users",
                "filters": {
                    "email_subscribed": True,
                    "user_types": ["shopper", "returning"],
                    "numeric_ranges": {
                        "income": {"min": 90000, "max": 200000},
                    },
                },
            },
        ),
        TemplateORM(
            name="Синтетика — Большая выборка (смешанная аудитория)",
            description="30 000 пользователей без фильтров — репрезентативная выборка для большинства A/B тестов.",
            template_type="synthetic_data",
            tags=["большая-выборка", "репрезентативная"],
            created_by="system",
            config_json={
                "num_users": 30000,
                "evaluation_metrics": True,
                "dataset_name": "mixed_audience_30k",
            },
        ),
        # --- A/B тесты ---
        # ВАЖНО: sampleSize явно задан <= 2000 (влезает в датасет 5000 записей).
        # minEffectSize = 0.3 (30%) — крупный эффект, малая выборка.
        # variantEffects гарантируют реальный, видимый эффект на графиках.
        TemplateORM(
            name="A/B тест — Конверсия кнопки (бинарная)",
            description=(
                "Классический продуктовый A/B тест по метрике conversion (0/1). "
                "Подходит для финальной проверки гипотезы перед релизом."
            ),
            template_type="ab_test",
            tags=["конверсия", "бинарная", "кнопка", "product"],
            created_by="system",
            config_json={
                "testName": "Тест конверсии кнопки",
                "variants": "A, B",
                "primaryMetric": "conversion",
                "metricType": "binary",
                "description": "Проверяем влияние нового CTA на конверсию",
                "confidenceLevel": 0.95,
                "power": 0.8,
                "minEffectSize": 0.1,
                "sampleSize": 12000,
                "trafficSplitType": "fixed",
                "analysisMode": "fixed_experiment",
                "simulationDurationMinutes": 25,
                "variantEffects": {
                    "B": {"conversion": 1.12}
                },
            },
        ),
        TemplateORM(
            name="A/B тест — Доход пользователя (непрерывная метрика)",
            description=(
                "Тест на изменение revenue (continuous). Подходит для оценки влияния новой механики монетизации."
            ),
            template_type="ab_test",
            tags=["доход", "непрерывная", "revenue", "product"],
            created_by="system",
            config_json={
                "testName": "Тест влияния на доход",
                "variants": "A, B",
                "primaryMetric": "revenue",
                "metricType": "continuous",
                "description": "Проверяем влияние новой функции на средний доход",
                "confidenceLevel": 0.95,
                "power": 0.8,
                "minEffectSize": 0.08,
                "sampleSize": 10000,
                "trafficSplitType": "fixed",
                "analysisMode": "fixed_experiment",
                "simulationDurationMinutes": 25,
                "variantEffects": {
                    "B": {"revenue": 1.10}
                },
            },
        ),
        TemplateORM(
            name="A/B тест — CTR три варианта (A/B/C)",
            description=(
                "Тест трёх вариантов баннера по метрике CTR. Используется для выбора лучшего креатива."
            ),
            template_type="ab_test",
            tags=["ctr", "баннер", "три-варианта", "product"],
            created_by="system",
            config_json={
                "testName": "A/B/C тест баннеров",
                "variants": "A, B, C",
                "primaryMetric": "ctr",
                "metricType": "ratio",
                "description": "Сравниваем три варианта баннера по CTR",
                "confidenceLevel": 0.95,
                "power": 0.8,
                "minEffectSize": 0.08,
                "sampleSize": 15000,
                "trafficSplitType": "fixed",
                "analysisMode": "fixed_experiment",
                "simulationDurationMinutes": 30,
                "variantEffects": {
                    "B": {"ctr": 1.08},
                    "C": {"ctr": 1.15},
                },
            },
        ),
        TemplateORM(
            name="A/B тест — Адаптивная стратегия (исследование)",
            description=(
                "Исследовательский bandit-режим для быстрого скрининга гипотез. "
                "Для финального решения по внедрению обязателен повторный fixed_experiment тест."
            ),
            template_type="ab_test",
            tags=["адаптивный", "bandit", "exploration"],
            created_by="system",
            config_json={
                "testName": "Адаптивный тест (исследование)",
                "variants": "A, B",
                "primaryMetric": "conversion",
                "metricType": "binary",
                "description": "Быстрый exploratory тест с адаптивным распределением трафика",
                "confidenceLevel": 0.95,
                "power": 0.8,
                "minEffectSize": 0.1,
                "sampleSize": 8000,
                "trafficSplitType": "adaptive",
                "analysisMode": "adaptive_bandit",
                "simulationDurationMinutes": 20,
                "variantEffects": {
                    "B": {"conversion": 1.12}
                },
            },
        ),
    ]
    db.add_all(defaults)
    db.commit()
    return len(defaults)


def create_assignment_audit(
    db: Session,
    *,
    test_id: str,
    session_id: str,
    user_id: str,
    variant: str,
    splitter_type: str,
    analysis_mode: str,
    traffic_split_type: str,
    hash_bucket: Optional[int] = None,
    hash_space_size: Optional[int] = None,
    seed: Optional[int] = None,
    assignment_metadata: Optional[dict[str, Any]] = None,
    do_commit: bool = True,
) -> AssignmentAuditORM:
    entity = AssignmentAuditORM(
        test_id=test_id,
        session_id=session_id,
        user_id=user_id,
        variant=variant,
        splitter_type=splitter_type,
        analysis_mode=analysis_mode,
        traffic_split_type=traffic_split_type,
        hash_bucket=hash_bucket,
        hash_space_size=hash_space_size,
        seed=seed,
        assignment_metadata=assignment_metadata,
    )
    db.add(entity)
    if do_commit:
        db.commit()
        db.refresh(entity)
    return entity


def get_assignment_audit_for_test(db: Session, test_id: str, limit: int = 1000) -> list[AssignmentAuditORM]:
    return (
        db.query(AssignmentAuditORM)
        .filter(AssignmentAuditORM.test_id == test_id)
        .order_by(AssignmentAuditORM.created_at.desc())
        .limit(limit)
        .all()
    )


def get_user_assignment(db: Session, *, test_id: str, user_id: str) -> Optional[UserAssignmentORM]:
    return (
        db.query(UserAssignmentORM)
        .filter(UserAssignmentORM.test_id == test_id, UserAssignmentORM.user_id == user_id)
        .first()
    )


def upsert_user_assignment(
    db: Session,
    *,
    test_id: str,
    user_id: str,
    variant: str,
    splitter_type: str,
    hash_bucket: Optional[int] = None,
    hash_space_size: Optional[int] = None,
    seed: Optional[int] = None,
    assignment_metadata: Optional[dict[str, Any]] = None,
    do_commit: bool = True,
) -> UserAssignmentORM:
    entity = get_user_assignment(db, test_id=test_id, user_id=user_id)
    if entity is None:
        entity = UserAssignmentORM(
            test_id=test_id,
            user_id=user_id,
            variant=variant,
            splitter_type=splitter_type,
            hash_bucket=hash_bucket,
            hash_space_size=hash_space_size,
            seed=seed,
            assignment_metadata=assignment_metadata,
        )
        db.add(entity)
    else:
        entity.variant = variant
        entity.splitter_type = splitter_type
        entity.hash_bucket = hash_bucket
        entity.hash_space_size = hash_space_size
        entity.seed = seed
        entity.assignment_metadata = assignment_metadata

    if do_commit:
        db.commit()
        db.refresh(entity)

    return entity


def create_metric_event_if_absent(
    db: Session,
    *,
    event_id: str,
    session_id: str,
    test_id: str,
    metric_name: str,
    value: float,
    do_commit: bool = True,
) -> tuple[MetricEventORM, bool]:
    existing = db.query(MetricEventORM).filter(MetricEventORM.event_id == event_id).first()
    if existing is not None:
        return existing, False

    entity = MetricEventORM(
        event_id=event_id,
        session_id=session_id,
        test_id=test_id,
        metric_name=metric_name,
        value=value,
    )
    db.add(entity)

    if do_commit:
        db.commit()
        db.refresh(entity)

    return entity, True
