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


def list_users(db: Session, limit: int = 500) -> list[UserORM]:
    return db.query(UserORM).order_by(UserORM.id.asc()).limit(limit).all()


def update_user_role(db: Session, user_id: int, role: str) -> Optional[UserORM]:
    # legacy-поле role сохраняем только для обратной совместимости,
    # в новой модели доступ определяется permissions_json.
    user = db.query(UserORM).filter(UserORM.id == user_id).first()
    if user is None:
        return None
    user.role = role
    db.commit()
    db.refresh(user)
    return user


def update_user_permissions(db: Session, user_id: int, permissions: list[str]) -> Optional[UserORM]:
    user = db.query(UserORM).filter(UserORM.id == user_id).first()
    if user is None:
        return None

    deduplicated = list(dict.fromkeys([p.strip() for p in permissions if p and p.strip()]))
    user.permissions_json = deduplicated
    db.commit()
    db.refresh(user)
    return user


def create_user(
    db: Session,
    *,
    username: str,
    hashed_password: str,
    full_name: str,
    role: str = "user",
    job_title: Optional[str] = None,
    permissions: Optional[list[str]] = None,
    email: Optional[str] = None,
    phone: Optional[str] = None,
    avatar_url: Optional[str] = None,
) -> UserORM:
    entity = UserORM(
        username=username,
        hashed_password=hashed_password,
        full_name=full_name,
        role=role,
        job_title=job_title,
        permissions_json=list(dict.fromkeys([p.strip() for p in (permissions or []) if p and p.strip()])),
        email=email,
        phone=phone,
        avatar_url=avatar_url,
    )
    db.add(entity)
    db.commit()
    db.refresh(entity)
    return entity


def update_user_profile(
    db: Session,
    *,
    user_id: int,
    full_name: str,
    email: Optional[str],
    phone: Optional[str],
    avatar_url: Optional[str],
) -> Optional[UserORM]:
    user = db.query(UserORM).filter(UserORM.id == user_id).first()
    if user is None:
        return None

    user.full_name = full_name
    user.email = email
    user.phone = phone
    user.avatar_url = avatar_url
    db.commit()
    db.refresh(user)
    return user


def update_user_avatar_blob(
    db: Session,
    *,
    user_id: int,
    avatar_blob: bytes,
    mime_type: str,
) -> Optional[UserORM]:
    user = db.query(UserORM).filter(UserORM.id == user_id).first()
    if user is None:
        return None

    user.avatar_blob = avatar_blob
    user.avatar_mime_type = mime_type
    db.commit()
    db.refresh(user)
    return user


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
    """
    Полностью заменяет системные шаблоны (created_by='system') на новую расширенную коллекцию.

    Важно:
    - старые системные шаблоны удаляются полностью;
    - пользовательские шаблоны (created_by != 'system') не затрагиваются;
    - создаётся 24 новых шаблона, покрывающих A/B/C+ тесты, adaptive/bandit,
      early stopping, генерацию синтетики и GAN-конфиги.
    """
    db.query(TemplateORM).filter(TemplateORM.created_by == "system").delete(synchronize_session=False)

    defaults = [
        # -----------------------------
        # GAN CONFIG TEMPLATES (8)
        # -----------------------------
        TemplateORM(
            name="GAN — Быстрый smoke-train (15 эпох)",
            description="Минимальная конфигурация для быстрой проверки пайплайна обучения и генерации.",
            template_type="gan_config",
            tags=["gan", "smoke", "быстрый"],
            created_by="system",
            config_json={
                "epochs": 15,
                "real_data_samples": 15000,
                "save_checkpoint": True,
                "checkpoint_name": "gan_smoke",
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
            name="GAN — Balanced baseline (50 эпох)",
            description="Сбалансированный production-ready baseline для большинства задач синтетики.",
            template_type="gan_config",
            tags=["gan", "baseline", "balanced"],
            created_by="system",
            config_json={
                "epochs": 50,
                "real_data_samples": 50000,
                "save_checkpoint": True,
                "checkpoint_name": "gan_balanced",
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
            name="GAN — High fidelity (90 эпох)",
            description="Конфиг для максимального качества распределений, когда время обучения не критично.",
            template_type="gan_config",
            tags=["gan", "high-quality", "production"],
            created_by="system",
            config_json={
                "epochs": 90,
                "real_data_samples": 120000,
                "save_checkpoint": True,
                "checkpoint_name": "gan_high_fidelity",
                "LATENT_DIM": 256,
                "BATCH_SIZE": 512,
                "LEARNING_RATE": 0.0001,
                "DROPOUT_RATE": 0.35,
                "LAMBDA_GP": 10,
                "N_CRITIC": 5,
                "GENERATOR_LAYERS": "512,1024,512,256",
                "DISCRIMINATOR_LAYERS": "512,1024,512,256",
                "USE_WGAN_GP": True,
            },
        ),
        TemplateORM(
            name="GAN — Low-resource CPU friendly",
            description="Конфигурация с пониженной нагрузкой для dev-стендов и ограниченных машин.",
            template_type="gan_config",
            tags=["gan", "cpu", "dev"],
            created_by="system",
            config_json={
                "epochs": 25,
                "real_data_samples": 12000,
                "save_checkpoint": True,
                "checkpoint_name": "gan_cpu_friendly",
                "LATENT_DIM": 48,
                "BATCH_SIZE": 96,
                "LEARNING_RATE": 0.00035,
                "DROPOUT_RATE": 0.25,
                "LAMBDA_GP": 8,
                "N_CRITIC": 3,
                "GENERATOR_LAYERS": "96,192,96",
                "DISCRIMINATOR_LAYERS": "96,192,96",
                "USE_WGAN_GP": True,
            },
        ),
        TemplateORM(
            name="GAN — Marketing events skew",
            description="Усиленная вариативность для датасетов с кампанийными всплесками и сезонностью.",
            template_type="gan_config",
            tags=["gan", "marketing", "seasonality"],
            created_by="system",
            config_json={
                "epochs": 60,
                "real_data_samples": 70000,
                "save_checkpoint": True,
                "checkpoint_name": "gan_marketing_skew",
                "LATENT_DIM": 160,
                "BATCH_SIZE": 320,
                "LEARNING_RATE": 0.00018,
                "DROPOUT_RATE": 0.28,
                "LAMBDA_GP": 12,
                "N_CRITIC": 5,
                "GENERATOR_LAYERS": "320,640,320",
                "DISCRIMINATOR_LAYERS": "320,640,320",
                "USE_WGAN_GP": True,
            },
        ),
        TemplateORM(
            name="GAN — Revenue-sensitive training",
            description="Фокус на стабильности непрерывных метрик (ARPU/Revenue) для monetization тестов.",
            template_type="gan_config",
            tags=["gan", "revenue", "monetization"],
            created_by="system",
            config_json={
                "epochs": 70,
                "real_data_samples": 85000,
                "save_checkpoint": True,
                "checkpoint_name": "gan_revenue_sensitive",
                "LATENT_DIM": 192,
                "BATCH_SIZE": 384,
                "LEARNING_RATE": 0.00015,
                "DROPOUT_RATE": 0.32,
                "LAMBDA_GP": 10,
                "N_CRITIC": 6,
                "GENERATOR_LAYERS": "384,768,384",
                "DISCRIMINATOR_LAYERS": "384,768,384",
                "USE_WGAN_GP": True,
            },
        ),
        TemplateORM(
            name="GAN — Exploration-heavy",
            description="Повышенная стохастичность для генерации «широкого» пространства пользовательских профилей.",
            template_type="gan_config",
            tags=["gan", "exploration", "diversity"],
            created_by="system",
            config_json={
                "epochs": 45,
                "real_data_samples": 45000,
                "save_checkpoint": True,
                "checkpoint_name": "gan_exploration",
                "LATENT_DIM": 224,
                "BATCH_SIZE": 256,
                "LEARNING_RATE": 0.00022,
                "DROPOUT_RATE": 0.4,
                "LAMBDA_GP": 9,
                "N_CRITIC": 4,
                "GENERATOR_LAYERS": "256,512,512,256",
                "DISCRIMINATOR_LAYERS": "256,512,512,256",
                "USE_WGAN_GP": True,
            },
        ),
        TemplateORM(
            name="GAN — Stable long-run",
            description="Долгое стабильное обучение для регулярного nightly синтетического репликационного контура.",
            template_type="gan_config",
            tags=["gan", "nightly", "stable"],
            created_by="system",
            config_json={
                "epochs": 120,
                "real_data_samples": 140000,
                "save_checkpoint": True,
                "checkpoint_name": "gan_stable_longrun",
                "LATENT_DIM": 128,
                "BATCH_SIZE": 512,
                "LEARNING_RATE": 0.00008,
                "DROPOUT_RATE": 0.25,
                "LAMBDA_GP": 10,
                "N_CRITIC": 5,
                "GENERATOR_LAYERS": "512,512,256",
                "DISCRIMINATOR_LAYERS": "512,512,256",
                "USE_WGAN_GP": True,
            },
        ),

        # -----------------------------
        # SYNTHETIC DATA TEMPLATES (8)
        # -----------------------------
        TemplateORM(
            name="Синтетика — Универсальная 20k",
            description="Репрезентативная аудитория без фильтров для большинства A/B и A/B/C экспериментов.",
            template_type="synthetic_data",
            tags=["synthetic", "universal", "20k"],
            created_by="system",
            config_json={
                "num_users": 20000,
                "evaluation_metrics": True,
                "dataset_name": "synthetic_universal_20k",
            },
        ),
        TemplateORM(
            name="Синтетика — Mobile first (iOS+Android)",
            description="Мобильная аудитория для UI/UX гипотез и пуш-воронок.",
            template_type="synthetic_data",
            tags=["synthetic", "mobile", "ux"],
            created_by="system",
            config_json={
                "num_users": 18000,
                "evaluation_metrics": True,
                "dataset_name": "synthetic_mobile_first",
                "filters": {
                    "devices": ["Mobile"],
                    "os": ["iOS", "Android"],
                    "push_enabled": True,
                },
            },
        ),
        TemplateORM(
            name="Синтетика — Desktop checkout focus",
            description="Desktop-аудитория для тестов корзины, чекаута и платежных сценариев.",
            template_type="synthetic_data",
            tags=["synthetic", "desktop", "checkout"],
            created_by="system",
            config_json={
                "num_users": 16000,
                "evaluation_metrics": True,
                "dataset_name": "synthetic_desktop_checkout",
                "filters": {
                    "devices": ["Desktop"],
                    "browsers": ["Chrome", "Edge", "Firefox"],
                    "user_types": ["shopper", "returning"],
                },
            },
        ),
        TemplateORM(
            name="Синтетика — Premium сегмент",
            description="Аудитория с высоким доходом для monetization-экспериментов и upsell.",
            template_type="synthetic_data",
            tags=["synthetic", "premium", "revenue"],
            created_by="system",
            config_json={
                "num_users": 12000,
                "evaluation_metrics": True,
                "dataset_name": "synthetic_premium_segment",
                "filters": {
                    "email_subscribed": True,
                    "user_types": ["shopper", "returning"],
                    "numeric_ranges": {
                        "income": {"min": 90000, "max": 250000},
                    },
                },
            },
        ),
        TemplateORM(
            name="Синтетика — New users activation",
            description="Новые пользователи для тестов onboarding, first-session и активации.",
            template_type="synthetic_data",
            tags=["synthetic", "new-users", "activation"],
            created_by="system",
            config_json={
                "num_users": 14000,
                "evaluation_metrics": True,
                "dataset_name": "synthetic_new_users_activation",
                "filters": {
                    "user_types": ["new"],
                    "traffic_sources": ["ads", "organic"],
                },
            },
        ),
        TemplateORM(
            name="Синтетика — Weekend traffic",
            description="Трафик выходных для гипотез с поведенческой сезонностью.",
            template_type="synthetic_data",
            tags=["synthetic", "weekend", "seasonality"],
            created_by="system",
            config_json={
                "num_users": 10000,
                "evaluation_metrics": True,
                "dataset_name": "synthetic_weekend_traffic",
                "filters": {
                    "is_weekend": True,
                    "traffic_sources": ["organic", "social", "ads"],
                },
            },
        ),
        TemplateORM(
            name="Синтетика — Retention campaign",
            description="Возвращающиеся пользователи для CRM/retention экспериментов.",
            template_type="synthetic_data",
            tags=["synthetic", "retention", "crm"],
            created_by="system",
            config_json={
                "num_users": 15000,
                "evaluation_metrics": True,
                "dataset_name": "synthetic_retention_campaign",
                "filters": {
                    "user_types": ["returning"],
                    "email_subscribed": True,
                    "push_enabled": True,
                },
            },
        ),
        TemplateORM(
            name="Синтетика — Stress test 60k",
            description="Большой датасет для нагрузочных симуляций и много-вариантных тестов (A/B/C/D/E).",
            template_type="synthetic_data",
            tags=["synthetic", "stress", "large"],
            created_by="system",
            config_json={
                "num_users": 60000,
                "evaluation_metrics": True,
                "dataset_name": "synthetic_stress_60k",
            },
        ),

        # -----------------------------
        # AB TEST TEMPLATES (8)
        # -----------------------------
        TemplateORM(
            name="A/B — Базовый фиксированный тест конверсии",
            description="Классический фиксированный A/B-тест по конверсии для финального статистического вывода.",
            template_type="ab_test",
            tags=["ab", "conversion", "fixed"],
            created_by="system",
            config_json={
                "testName": "AB Базовый тест конверсии",
                "variants": "A, B",
                "primaryMetric": "conversion",
                "metricType": "binary",
                "description": "Базовый продуктовый фиксированный эксперимент",
                "confidenceLevel": 0.95,
                "power": 0.8,
                "minEffectSize": 0.08,
                "sampleSize": 14000,
                "trafficSplitType": "fixed",
                "analysisMode": "fixed_experiment",
                "simulationDurationMinutes": 25,
                "variantEffects": {"B": {"conversion": 1.10}},
                "earlyStoppingEnabled": False,
                "early_stopping_enabled": False,
            },
        ),
        TemplateORM(
            name="A/B/C — Фиксированный тест вариантов CTA",
            description="Трёхвариантный A/B/C-тест для выбора лучшего CTA по конверсии.",
            template_type="ab_test",
            tags=["abc", "cta", "conversion"],
            created_by="system",
            config_json={
                "testName": "ABC Фиксированный тест CTA",
                "variants": "A, B, C",
                "primaryMetric": "conversion",
                "metricType": "binary",
                "description": "Сравнение трёх CTA",
                "confidenceLevel": 0.95,
                "power": 0.85,
                "minEffectSize": 0.07,
                "sampleSize": 22000,
                "trafficSplitType": "fixed",
                "analysisMode": "fixed_experiment",
                "simulationDurationMinutes": 30,
                "variantEffects": {
                    "B": {"conversion": 1.07},
                    "C": {"conversion": 1.13},
                },
                "earlyStoppingEnabled": False,
                "early_stopping_enabled": False,
            },
        ),
        TemplateORM(
            name="A/B/C/D — Эксперимент лендинга",
            description="Мультивариантный фиксированный тест с 4 вариантами для лендинга и hero-блока.",
            template_type="ab_test",
            tags=["abcd", "landing", "fixed"],
            created_by="system",
            config_json={
                "testName": "ABCD Фиксированный лендинг-тест",
                "variants": "A, B, C, D",
                "primaryMetric": "ctr",
                "metricType": "ratio",
                "description": "Тест 4 вариантов hero+CTA",
                "confidenceLevel": 0.95,
                "power": 0.8,
                "minEffectSize": 0.06,
                "sampleSize": 26000,
                "trafficSplitType": "fixed",
                "analysisMode": "fixed_experiment",
                "simulationDurationMinutes": 35,
                "variantEffects": {
                    "B": {"ctr": 1.05},
                    "C": {"ctr": 1.09},
                    "D": {"ctr": 1.12},
                },
                "earlyStoppingEnabled": False,
                "early_stopping_enabled": False,
            },
        ),
        TemplateORM(
            name="A/B — Фиксированный тест выручки с защитными ограничениями",
            description="Фиксированный тест выручки + защитные ограничения по задержке и доле ошибок для безопасного релиза.",
            template_type="ab_test",
            tags=["ab", "revenue", "guardrails"],
            created_by="system",
            config_json={
                "testName": "AB Выручка с защитными ограничениями",
                "variants": "A, B",
                "primaryMetric": "revenue",
                "metricType": "continuous",
                "description": "Monetization гипотеза с защитными метриками",
                "confidenceLevel": 0.95,
                "power": 0.85,
                "minEffectSize": 0.05,
                "sampleSize": 24000,
                "trafficSplitType": "fixed",
                "analysisMode": "fixed_experiment",
                "simulationDurationMinutes": 40,
                "variantEffects": {"B": {"revenue": 1.08}},
                "guardrailsConfig": {
                    "latency_ms": {"threshold": 5, "direction": "max_increase"},
                    "error_rate": {"threshold": 0.01, "direction": "max_increase"}
                },
                "earlyStoppingEnabled": False,
                "early_stopping_enabled": False,
            },
        ),
        TemplateORM(
            name="A/B — Успех с ранней остановкой",
            description="Фиксированный A/B-тест с включённой ранней остановкой для быстрого завершения при выраженном приросте.",
            template_type="ab_test",
            tags=["ab", "early-stop", "success",],
            created_by="system",
            config_json={
                "testName": "AB Ранняя остановка при успехе",
                "variants": "A, B",
                "primaryMetric": "conversion",
                "metricType": "binary",
                "description": "Сценарий с ранней остановкой по успеху",
                "confidenceLevel": 0.95,
                "power": 0.8,
                "minEffectSize": 0.06,
                "sampleSize": 18000,
                "trafficSplitType": "fixed",
                "analysisMode": "fixed_experiment",
                "simulationDurationMinutes": 30,
                "variantEffects": {"B": {"conversion": 1.18}},
                "earlyStoppingEnabled": True,
                "early_stopping_enabled": True,
            },
        ),
        TemplateORM(
            name="A/B/C — Адаптивное исследование",
            description="Исследовательский режим adaptive_bandit для быстрого перераспределения трафика на лидирующий вариант.",
            template_type="ab_test",
            tags=["abc", "adaptive", "bandit"],
            created_by="system",
            config_json={
                "testName": "ABC Адаптивное исследование",
                "variants": "A, B, C",
                "primaryMetric": "ctr",
                "metricType": "ratio",
                "description": "Bandit-скрининг 3 креативов",
                "confidenceLevel": 0.95,
                "power": 0.8,
                "minEffectSize": 0.07,
                "sampleSize": 16000,
                "trafficSplitType": "adaptive",
                "analysisMode": "adaptive_bandit",
                "simulationDurationMinutes": 25,
                "variantEffects": {
                    "B": {"ctr": 1.06},
                    "C": {"ctr": 1.14},
                },
                "earlyStoppingEnabled": False,
                "early_stopping_enabled": False,
            },
        ),
        TemplateORM(
            name="A/B/C/D/E — Адаптивный портфель",
            description="Пятивариантный адаптивный тест для портфеля гипотез (только исследовательский режим).",
            template_type="ab_test",
            tags=["abcde", "adaptive", "portfolio"],
            created_by="system",
            config_json={
                "testName": "ABCDE Адаптивный портфель",
                "variants": "A, B, C, D, E",
                "primaryMetric": "conversion",
                "metricType": "binary",
                "description": "Мультиарм bandit для портфеля UX-гипотез",
                "confidenceLevel": 0.95,
                "power": 0.75,
                "minEffectSize": 0.08,
                "sampleSize": 32000,
                "trafficSplitType": "adaptive",
                "analysisMode": "adaptive_bandit",
                "simulationDurationMinutes": 45,
                "variantEffects": {
                    "B": {"conversion": 1.03},
                    "C": {"conversion": 1.05},
                    "D": {"conversion": 1.10},
                    "E": {"conversion": 1.07},
                },
                "earlyStoppingEnabled": False,
                "early_stopping_enabled": False,
            },
        ),
        TemplateORM(
            name="A/B — Контрольная группа + ранняя остановка + защитные ограничения",
            description="Полный сценарий: фиксированный статистический вывод, логика контрольной группы через прирост B, ранняя остановка и защитные ограничения.",
            template_type="ab_test",
            tags=["ab", "full-flow", "early-stop", "guardrails"],
            created_by="system",
            config_json={
                "testName": "AB Полный сценарий",
                "variants": "A, B",
                "primaryMetric": "revenue",
                "metricType": "continuous",
                "description": "Полнофункциональный шаблон для промышленного сценария",
                "confidenceLevel": 0.95,
                "power": 0.9,
                "minEffectSize": 0.04,
                "sampleSize": 28000,
                "trafficSplitType": "fixed",
                "analysisMode": "fixed_experiment",
                "simulationDurationMinutes": 50,
                "variantEffects": {"B": {"revenue": 1.09, "conversion": 1.03}},
                "guardrailsConfig": {
                    "latency_ms": {"threshold": 3, "direction": "max_increase"},
                    "error_rate": {"threshold": 0.005, "direction": "max_increase"}
                },
                "earlyStoppingEnabled": True,
                "early_stopping_enabled": True,
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
