from __future__ import annotations

from typing import Any, Optional

from sqlalchemy import func, select, update, delete
from sqlalchemy.orm import Session
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.dialects.postgresql import JSONB

from backend.database.models import ABTestORM, CheckpointORM, GeneratedDataORM, UserORM, ABTestTimeSeriesORM


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
    status: str = "prepared",  # Изменено с "active" на "prepared"
    do_commit: bool = True,
) -> ABTestORM:
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
        real_world_duration_days=real_world_duration_days,
        simulation_duration_minutes=simulation_duration_minutes,
        traffic_split_type=traffic_split_type,
        created_by_user_id=created_by_user_id,
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
        metrics_json = CheckpointORM.metrics_json.cast(JSONB)
        binary_value = metrics_json["binary"].astext
        query = query.filter(
            metrics_json.has_key("binary"),
            func.length(func.coalesce(binary_value, "")) > 0,
        )
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
    query = select(CheckpointORM).order_by(CheckpointORM.created_at.desc()).limit(limit)
    if only_with_binary:
        # Для async нужно использовать jsonb операторы через text()
        from sqlalchemy import text
        query = query.where(
            text("metrics_json IS NOT NULL AND metrics_json->>'binary' IS NOT NULL AND length(metrics_json->>'binary') > 0")
        )
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
