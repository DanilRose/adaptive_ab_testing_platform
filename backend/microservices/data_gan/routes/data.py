from datetime import datetime
from typing import Dict, Optional, List
import os
import tempfile

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.api.platform_instance import get_platform
from backend.auth.models import User
from backend.auth.service import require_role
from backend.database import crud
from backend.database.session import get_db
from backend.microservices.ab_testing import ABTestLifecycleService
from backend.microservices.data_gan import (
    DatasetPersistenceService,
    CheckpointStorageService,
    DataGANLifecycleService,
)
from backend.microservices.shared import SimpleTTLCache
from backend.services.gan_integration import gan_service
from backend.services.traffic_generator.data_generator import RealisticDataGenerator

platform = get_platform()

router = APIRouter(prefix="/api/v1/data", tags=["Data Generation"])

data_generator = RealisticDataGenerator()


# Глобальный кэш для статусов
_status_cache = SimpleTTLCache(ttl_seconds=10)  # 10 секунд TTL


class DataGenerationRequest(BaseModel):
    num_samples: int = Field(1000, ge=100, le=100000, description="Количество samples")
    save_to_file: bool = Field(False, description="Сохранить в файл")
    include_evaluation: bool = Field(True, description="Включить оценку качества")
    filters: Optional[dict] = Field(None, description="Фильтры генерации, совпадающие с возможностями генератора")


class GANTrainingRequest(BaseModel):
    epochs: int = Field(50, ge=10, le=500, description="Количество эпох")
    real_data_samples: int = Field(50000, ge=1000, le=100000, description="Samples для обучения")
    save_checkpoint: bool = Field(True, description="Сохранить чекпоинт")
    checkpoint_name: Optional[str] = Field(None, description="Пользовательское имя для чекпоинта")
    gan_config: Optional[dict] = Field(None, description="Конфигурация GAN (эпохи, размер шума и т.д.)")


class SyntheticDataRequest(BaseModel):
    num_users: int = Field(10000, ge=100, le=100000, description="Количество пользователей")
    evaluation_metrics: bool = Field(True, description="Рассчитать метрики качества")
    filters: Optional[dict] = Field(None, description="Фильтры генерации (по устройствам, городам и т.д.)")
    dataset_name: Optional[str] = Field(None, description="Название набора синтетических данных")


class DatasetListItem(BaseModel):
    id: int
    dataset_name: Optional[str]
    data_type: str
    sample_count: int
    created_at: datetime
    has_full_records: bool


class LoadCheckpointRequest(BaseModel):
    checkpoint_name: str


@router.post("/generate-real", summary="Сгенерировать реальные данные")
async def generate_real_data(
    request: DataGenerationRequest,
    current_user: User = Depends(require_role("developer", "analyst")),
    db: Session = Depends(get_db),
):
    try:
        real_data = await run_in_threadpool(
            data_generator.generate_dataset,
            request.num_samples,
            request.filters,
        )

        result = {
            "generated_samples": len(real_data),
            "features": list(real_data.columns),
            "data_preview": real_data.head(10).to_dict("records"),
        }

        if request.include_evaluation:
            stats = real_data.describe().to_dict()
            result["statistics"] = stats

        if request.save_to_file:
            result["warning"] = "save_to_file=true проигнорирован: данные сохраняются только в БД"

        DatasetPersistenceService.persist_dataset(
            db=db,
            data_type="real",
            dataframe=real_data,
            generated_by=current_user.username,
            include_evaluation=request.include_evaluation,
        )
        
        # Инвалидируем кэш
        _status_cache.invalidate_prefix("generated_history_")
        _status_cache.invalidate("dataset_stats")

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка генерации данных: {str(e)}")


@router.post("/train-gan", summary="Обучить GAN модель")
async def train_gan_model(
    request: GANTrainingRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(require_role("developer", "analyst")),
):
    try:
        DataGANLifecycleService.enqueue_training(
            background_tasks=background_tasks,
            epochs=request.epochs,
            real_data_samples=request.real_data_samples,
            save_checkpoint=request.save_checkpoint,
            checkpoint_name=request.checkpoint_name,
            gan_config=request.gan_config,
            trained_by=current_user.username,
            data_generator=data_generator,
            gan_service=gan_service,
            status_cache=_status_cache,
        )

        return {
            "status": "training_started",
            "epochs": request.epochs,
            "real_data_samples": request.real_data_samples,
            "message": "GAN модель начала обучение в фоновом режиме",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка обучения GAN: {str(e)}")


@router.get("/gan-status", summary="Статус GAN модели")
async def get_gan_status(current_user: User = Depends(require_role("developer", "analyst"))):
    try:
        # Проверяем кэш
        cached = _status_cache.get("gan_status")
        if cached:
            return cached
        
        status = gan_service.get_status()
        _status_cache.set("gan_status", status)
        return status

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка получения статуса: {str(e)}")


@router.post("/generate-synthetic", summary="Сгенерировать синтетические данные")
async def generate_synthetic_data(
    request: SyntheticDataRequest,
    current_user: User = Depends(require_role("developer", "analyst")),
    db: Session = Depends(get_db),
):
    try:
        if not gan_service.is_trained:
            raise HTTPException(status_code=400, detail="GAN модель не обучена. Сначала обучите модель.")

        synthetic_data = await run_in_threadpool(
            gan_service.generate_synthetic_data,
            request.num_users,
            request.filters,
            request.dataset_name,
        )

        if synthetic_data is None:
            raise HTTPException(status_code=500, detail="Ошибка генерации синтетических данных")

        result = {
            "synthetic_samples": len(synthetic_data),
            "features": list(synthetic_data.columns),
            "synthetic_preview": synthetic_data.head(10).to_dict("records"),
        }

        DatasetPersistenceService.persist_dataset(
            db=db,
            data_type="synthetic",
            dataframe=synthetic_data,
            generated_by=current_user.username,
            dataset_name=request.dataset_name,
        )
        
        # Инвалидируем кэш
        _status_cache.invalidate_prefix("generated_history_")
        _status_cache.invalidate("dataset_stats")

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка генерации синтетических данных: {str(e)}")


@router.get("/dataset-stats", summary="Статистика datasets")
async def get_dataset_stats(current_user: User = Depends(require_role("developer", "analyst"))):
    try:
        # Проверяем кэш
        cached = _status_cache.get("dataset_stats")
        if cached:
            return cached
        
        # Кешированные опции фильтров - не нужно генерировать данные
        filter_options = data_generator.get_filter_options()
        
        result = filter_options
        _status_cache.set("dataset_stats", result)
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка получения статистики: {str(e)}")


@router.get("/generated-history", summary="История сгенерированных данных")
async def get_generated_history(
    limit: int = Query(50, ge=1, le=200, description="Максимум записей"),
    offset: int = Query(0, ge=0, description="Смещение"),
    data_type: Optional[str] = Query(None, description="Фильтр по типу данных (real|synthetic)"),
    current_user: User = Depends(require_role("developer", "analyst")),
    db: Session = Depends(get_db),
):
    try:
        # Проверяем кэш
        cache_key = f"generated_history_{limit}_{offset}_{data_type}"
        cached = _status_cache.get(cache_key)
        if cached:
            return cached
        
        rows = crud.list_generated_data(db, limit=limit)
        
        # Применяем фильтры и offset
        if data_type:
            rows = [r for r in rows if r.data_type == data_type]
        
        total_count = len(rows)
        rows = rows[offset:offset+limit]
        
        result = {
            "items": [
                {
                    "id": r.id,
                    "data_type": r.data_type,
                    "sample_count": r.sample_count,
                    "file_path": r.file_path,
                    "storage": "database_only",
                    "dataset_name": (r.extra_metadata or {}).get("dataset_name"),
                    "preview_json": r.preview_json,
                    # Убираем огромный records из extra_metadata для быстрой загрузки
                    "extra_metadata": {
                        "generated_by": (r.extra_metadata or {}).get("generated_by"),
                        "dataset_name": (r.extra_metadata or {}).get("dataset_name"),
                        "include_evaluation": (r.extra_metadata or {}).get("include_evaluation"),
                        "records_count": int((r.extra_metadata or {}).get("records_count") or r.sample_count or 0),
                    },
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                }
                for r in rows
            ],
            "count": len(rows),
            "total": total_count,
            "offset": offset,
            "limit": limit,
            "has_more": offset + limit < total_count,
        }
        
        _status_cache.set(cache_key, result)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка истории generated data: {str(e)}")


@router.get("/gan-checkpoints", summary="Список доступных чекпоинтов")
async def get_gan_checkpoints(
    current_user: User = Depends(require_role("developer", "analyst")),
    db: Session = Depends(get_db),
):
    try:
        # Проверяем кэш
        cached = _status_cache.get("gan_checkpoints")
        if cached:
            return cached
        
        checkpoints = crud.list_checkpoints(db, limit=100, only_with_binary=True)

        # Возвращаем минимальную информацию без бинарных данных для быстрой загрузки
        result = {
            "checkpoints": [
                {
                    "id": c.id,
                    "name": c.name,
                    "file_path": c.file_path,
                    "version": c.version,
                    "epoch": c.epoch,
                    "metrics": {
                        "trained_by": (c.metrics_json or {}).get("trained_by"),
                        "loaded_by": (c.metrics_json or {}).get("loaded_by"),
                        "size": (c.metrics_json or {}).get("size"),
                        "final_g_loss": (c.metrics_json or {}).get("final_g_loss"),
                        "final_d_loss": (c.metrics_json or {}).get("final_d_loss"),
                    },
                    "created_at": c.created_at.isoformat() if c.created_at else None,
                }
                for c in checkpoints
            ],
            "count": len(checkpoints),
        }
        
        _status_cache.set("gan_checkpoints", result)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка получения чекпоинтов: {str(e)}")


@router.post("/gan-load-checkpoint", summary="Загрузить чекпоинт")
async def load_gan_checkpoint(
    request: LoadCheckpointRequest,
    current_user: User = Depends(require_role("developer", "analyst")),
    db: Session = Depends(get_db),
):
    try:
        checkpoint = crud.get_checkpoint_by_name(db, request.checkpoint_name) or crud.get_checkpoint_by_file_path(db, request.checkpoint_name)
        if checkpoint is None:
            raise HTTPException(status_code=404, detail="Чекпоинт не найден в БД")

        checkpoint_bytes = CheckpointStorageService.load_checkpoint_bytes(checkpoint)
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp_file:
            tmp_file.write(checkpoint_bytes)
            tmp_file_path = tmp_file.name

        success = await run_in_threadpool(gan_service.load_pretrained_model, tmp_file_path)
        os.unlink(tmp_file_path)

        if success:
            crud.upsert_checkpoint(
                db,
                name=checkpoint.name,
                file_path=checkpoint.file_path,
                version=checkpoint.version,
                epoch=checkpoint.epoch,
                metrics_json={**(checkpoint.metrics_json or {}), "loaded_by": current_user.username},
            )
            
            # Инвалидируем кэш
            _status_cache.invalidate("gan_checkpoints")
            _status_cache.invalidate("gan_status")
            
            return {
                "status": "success",
                "message": f"Модель загружена из {checkpoint.name}",
                "is_trained": gan_service.is_trained,
            }
        raise HTTPException(status_code=400, detail="Не удалось загрузить модель")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка загрузки модели: {str(e)}")


@router.post("/stop-gan-training", summary="Остановить обучение GAN")
async def stop_gan_training(
    current_user: User = Depends(require_role("developer", "analyst")),
):
    try:
        success = gan_service.stop_training()
        if success:
            return {"status": "stopping", "message": "Запрос на остановку обучения отправлен"}
        return {"status": "not_training", "message": "GAN не находится в процессе обучения"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка остановки обучения: {str(e)}")


@router.post("/resume-gan-training", summary="Возобновить обучение GAN")
async def resume_gan_training(
    current_user: User = Depends(require_role("developer", "analyst")),
):
    try:
        success = gan_service.resume_training()
        if success:
            return {"status": "resumed", "message": "Обучение возобновлено"}
        return {"status": "cannot_resume", "message": "Невозможно возобновить обучение"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка возобновления обучения: {str(e)}")


@router.post("/reset-gan-training", summary="Сбросить обучение GAN")
async def reset_gan_training(
    current_user: User = Depends(require_role("developer", "analyst")),
):
    try:
        success = gan_service.reset_training()
        if success:
            return {"status": "reset", "message": "Обучение GAN сброшено"}
        return {"status": "error", "message": "Не удалось сбросить обучение"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка сброса обучения: {str(e)}")


@router.delete("/generated-history/{item_id}", summary="Удалить запись синтетического датасета")
async def delete_generated_history_item(
    item_id: int,
    current_user: User = Depends(require_role("developer", "analyst")),
    db: Session = Depends(get_db),
):
    try:
        success = crud.delete_generated_data_by_id(db, item_id)
        if not success:
            raise HTTPException(status_code=404, detail="Запись не найдена")
        
        # Инвалидируем кэш
        _status_cache.invalidate_prefix("generated_history_")
        _status_cache.invalidate("dataset_stats")
        
        return {"status": "deleted", "id": item_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка удаления записи: {str(e)}")


@router.delete("/gan-checkpoints/{checkpoint_id}", summary="Удалить чекпоинт")
async def delete_gan_checkpoint(
    checkpoint_id: int,
    current_user: User = Depends(require_role("developer", "analyst")),
    db: Session = Depends(get_db),
):
    try:
        success = crud.delete_checkpoint_by_id(db, checkpoint_id)
        if not success:
            raise HTTPException(status_code=404, detail="Чекпоинт не найден")
        
        # Инвалидируем кэш
        _status_cache.invalidate("gan_checkpoints")
        
        return {"status": "deleted", "id": checkpoint_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка удаления чекпоинта: {str(e)}")


@router.post("/run-ab-test-simulation", summary="Запустить симуляцию A/B теста")
async def run_ab_test_simulation(
    request: dict,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(require_role("developer", "analyst")),
    db: Session = Depends(get_db),
):
    """Совместимый endpoint: делегирует запуск в единый orchestration-контур /tests/{test_id}/start-simulation."""
    try:
        test_id = request.get("test_id")
        if not test_id:
            raise HTTPException(status_code=400, detail="test_id is required")

        prepared = ABTestLifecycleService.prepare_simulation_start(
            db=db,
            test_id=test_id,
            dataset_id=request.get("dataset_id"),
            user_count=request.get("user_count"),
            strategy=request.get("strategy", "fixed"),
            simulation_minutes=request.get("simulation_minutes"),
            variant_effects=request.get("variant_effects"),
        )

        simulation_meta = ABTestLifecycleService.enqueue_simulation(
            background_tasks=background_tasks,
            platform=platform,
            test_id=test_id,
            prepared=prepared,
        )

        return {
            "status": "simulation_started",
            "message": f"Симуляция A/B теста запущена для {test_id} с {simulation_meta['user_count']} пользователями",
            **simulation_meta,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка запуска симуляции: {str(e)}")


@router.get("/generated-history/{item_id}/full", summary="Получить полный датасет")
async def get_full_dataset(
    item_id: int,
    current_user: User = Depends(require_role("developer", "analyst")),
    db: Session = Depends(get_db),
):
    try:
        dataset = crud.get_generated_data_by_id(db, item_id)
        if not dataset:
            raise HTTPException(status_code=404, detail="Датасет не найден")
        
        records = DatasetPersistenceService.load_dataset_records_for_entity(dataset)

        return {
            "id": dataset.id,
            "data_type": dataset.data_type,
            "sample_count": dataset.sample_count,
            "dataset_name": (dataset.extra_metadata or {}).get("dataset_name"),
            "records": records,
            "preview_json": dataset.preview_json,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка получения датасета: {str(e)}")
