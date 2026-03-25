from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd
from sqlalchemy.orm import Session

from backend.database.session import SessionLocal

from backend.database import crud
from backend.microservices.shared.storage import (
    load_checkpoint_binary,
    load_dataset_records,
    save_checkpoint_binary,
    save_dataset_records,
)


class DatasetPersistenceService:
    """Сервис сохранения/чтения датасетов с внешним storage вместо JSON в БД."""

    @staticmethod
    def persist_dataset(
        *,
        db: Session,
        data_type: str,
        dataframe: pd.DataFrame,
        generated_by: str,
        dataset_name: Optional[str] = None,
        include_evaluation: Optional[bool] = None,
    ):
        metadata: Dict[str, Any] = {
            "generated_by": generated_by,
            "dataset_name": dataset_name,
            "records_count": int(len(dataframe)),
        }
        if include_evaluation is not None:
            metadata["include_evaluation"] = include_evaluation

        entity = crud.create_generated_data(
            db,
            data_type=data_type,
            sample_count=len(dataframe),
            file_path=None,
            schema_json={col: str(dtype) for col, dtype in dataframe.dtypes.items()},
            preview_json=dataframe.head(10).to_dict("records"),
            extra_metadata=metadata,
            do_commit=True,
        )

        records_path = save_dataset_records(entity.id, dataframe.to_dict("records"))
        entity.file_path = records_path
        entity.extra_metadata = {
            **(entity.extra_metadata or {}),
            "records_storage": "file",
            "records_path": records_path,
        }
        db.commit()
        db.refresh(entity)
        return entity

    @staticmethod
    def load_dataset_records_for_entity(entity) -> list[dict[str, Any]]:
        metadata = entity.extra_metadata or {}
        if metadata.get("records_storage") == "file":
            records = load_dataset_records(entity.file_path)
            if records:
                return records
        return metadata.get("records") or entity.preview_json or []


class CheckpointStorageService:
    """Сервис хранения бинарных чекпоинтов на файловой системе."""

    @staticmethod
    def save_checkpoint_bytes(
        *,
        db: Session,
        checkpoint_name: str,
        checkpoint_bytes: bytes,
        epoch: Optional[int],
        trained_by: str,
        final_g_loss: Optional[float],
        final_d_loss: Optional[float],
    ):
        file_path = save_checkpoint_binary(checkpoint_name, checkpoint_bytes)
        return crud.upsert_checkpoint(
            db,
            name=checkpoint_name,
            file_path=file_path,
            version="1.0",
            epoch=epoch,
            metrics_json={
                "trained_by": trained_by,
                "size": len(checkpoint_bytes),
                "final_g_loss": final_g_loss,
                "final_d_loss": final_d_loss,
            },
        )

    @staticmethod
    def save_checkpoint_bytes_with_local_session(
        *,
        checkpoint_name: str,
        checkpoint_bytes: bytes,
        epoch: Optional[int],
        trained_by: str,
        final_g_loss: Optional[float],
        final_d_loss: Optional[float],
    ):
        with SessionLocal() as db:
            return CheckpointStorageService.save_checkpoint_bytes(
                db=db,
                checkpoint_name=checkpoint_name,
                checkpoint_bytes=checkpoint_bytes,
                epoch=epoch,
                trained_by=trained_by,
                final_g_loss=final_g_loss,
                final_d_loss=final_d_loss,
            )

    @staticmethod
    def load_checkpoint_bytes(checkpoint) -> bytes:
        return load_checkpoint_binary(checkpoint.file_path)
