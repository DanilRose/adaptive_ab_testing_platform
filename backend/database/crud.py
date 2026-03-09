from __future__ import annotations

from typing import Any, Optional

from sqlalchemy import func
from sqlalchemy.orm import Session
from sqlalchemy.dialects.postgresql import JSONB

from backend.database.models import ABTestORM, CheckpointORM, GeneratedDataORM, UserORM


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
    status: str = "active",
) -> ABTestORM:
    entity = ABTestORM(
        test_id=test_id,
        test_name=test_name,
        description=description,
        status=status,
        variants=variants,
        primary_metric=primary_metric,
        metric_type=metric_type,
        sample_size=sample_size,
        confidence_level=confidence_level,
        power=power,
        min_effect_size=min_effect_size,
        created_by_user_id=created_by_user_id,
    )
    db.add(entity)
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
