from datetime import datetime
from typing import Dict, Optional
import os
import tempfile
import torch

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.api.platform_instance import get_platform
from backend.auth.models import User
from backend.auth.service import require_role
from backend.database import crud
from backend.database.session import SessionLocal, get_db
from backend.services.gan_integration import gan_service
from backend.services.traffic_generator.data_generator import RealisticDataGenerator
from backend.gan.config import GANConfig

platform = get_platform()

router = APIRouter(prefix="/api/v1/data", tags=["Data Generation"])

data_generator = RealisticDataGenerator()


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


class LoadCheckpointRequest(BaseModel):
    checkpoint_name: str


@router.post("/generate-real", summary="Сгенерировать реальные данные")
async def generate_real_data(
    request: DataGenerationRequest,
    current_user: User = Depends(require_role("developer", "analyst")),
    db: Session = Depends(get_db),
):
    try:
        real_data = data_generator.generate_dataset(request.num_samples, filters=request.filters)

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

        crud.create_generated_data(
            db,
            data_type="real",
            sample_count=len(real_data),
            file_path=None,
            schema_json={col: str(dtype) for col, dtype in real_data.dtypes.items()},
            preview_json=real_data.head(10).to_dict("records"),
            extra_metadata={
                "generated_by": current_user.username,
                "include_evaluation": request.include_evaluation,
                "records": real_data.to_dict("records"),
            },
        )

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
        base_config = GANConfig()
        effective_config = {}
        for key, value in base_config.__dict__.items():
            if isinstance(value, torch.device):
                effective_config[key] = str(value)
            else:
                effective_config[key] = value
        
        if request.gan_config:
            for key, value in request.gan_config.items():
                if key in effective_config:
                    effective_config[key] = value

        effective_config["EPOCHS"] = request.epochs

        real_data = data_generator.generate_dataset(request.real_data_samples)

        def train_in_background(username: str, checkpoint_name_override: Optional[str]):
            try:
                result = gan_service.train_gan(
                    real_data, 
                    request.epochs, 
                    config_overrides=effective_config
                )
                
                import time
                max_wait = request.epochs * 10  
                waited = 0
                while gan_service.current_status.startswith("training") and waited < max_wait:
                    time.sleep(1)
                    waited += 1
                
                if request.save_checkpoint and gan_service.gan_model and gan_service.is_trained:
                    checkpoint_payload = {
                        "epoch": gan_service.current_epoch,
                        "generator_state_dict": gan_service.gan_model.generator.state_dict(),
                        "discriminator_state_dict": gan_service.gan_model.discriminator.state_dict(),
                        "optimizer_G_state_dict": gan_service.gan_model.optimizer_G.state_dict(),
                        "optimizer_D_state_dict": gan_service.gan_model.optimizer_D.state_dict(),
                        "g_losses": gan_service.gan_model.g_losses,
                        "d_losses": gan_service.gan_model.d_losses,
                        "feature_info": gan_service.gan_model.feature_info,
                        "processed_columns": gan_service.gan_model.processed_columns,
                        "scalers": gan_service.gan_model.scalers,
                    }
                    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp_file:
                        import torch
                        torch.save(checkpoint_payload, tmp_file.name)
                        tmp_file_path = tmp_file.name

                    with open(tmp_file_path, "rb") as f:
                        checkpoint_bytes = f.read()
                    os.unlink(tmp_file_path)

                    checkpoint_name = checkpoint_name_override or f"gan_trained_{datetime.now().strftime('%Y%m%d_%H%M')}.pth"
                    with SessionLocal() as db:
                        crud.upsert_checkpoint(
                            db,
                            name=checkpoint_name,
                            file_path=f"db://checkpoints/{checkpoint_name}",
                            version="1.0",
                            epoch=gan_service.current_epoch,
                            metrics_json={
                                "trained_by": username,
                                "size": len(checkpoint_bytes),
                                "binary": checkpoint_bytes.hex(),
                                "final_g_loss": gan_service.gan_model.g_losses[-1] if gan_service.gan_model.g_losses else None,
                                "final_d_loss": gan_service.gan_model.d_losses[-1] if gan_service.gan_model.d_losses else None,
                            },
                        )
                    # Устанавливаем имя чекпоинта и статус после успешного сохранения
                    checkpoint_name_clean = checkpoint_name[:-4] if checkpoint_name.endswith('.pth') else checkpoint_name
                    gan_service.loaded_checkpoint_name = checkpoint_name_clean
                    gan_service.current_status = "checkpoint_loaded"
                    print(f" Чекпоинт '{checkpoint_name}' успешно сохранен в БД")
            except Exception as e:
                print(f" Background training error: {e}")
                import traceback
                traceback.print_exc()

        background_tasks.add_task(train_in_background, current_user.username, request.checkpoint_name)

        return {
            "status": "training_started",
            "epochs": request.epochs,
            "real_data_samples": len(real_data),
            "message": "GAN модель начала обучение в фоновом режиме",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка обучения GAN: {str(e)}")


@router.get("/gan-status", summary="Статус GAN модели")
async def get_gan_status(current_user: User = Depends(require_role("developer", "analyst"))):
    try:
        status = gan_service.get_status()
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

        synthetic_data = gan_service.generate_synthetic_data(
            request.num_users,
            filters=request.filters,
            dataset_name=request.dataset_name,
        )

        if synthetic_data is None:
            raise HTTPException(status_code=500, detail="Ошибка генерации синтетических данных")

        result = {
            "synthetic_samples": len(synthetic_data),
            "features": list(synthetic_data.columns),
            "synthetic_preview": synthetic_data.head(10).to_dict("records"),
        }

        crud.create_generated_data(
            db,
            data_type="synthetic",
            sample_count=len(synthetic_data),
            file_path=None,
            schema_json={col: str(dtype) for col, dtype in synthetic_data.dtypes.items()},
            preview_json=synthetic_data.head(10).to_dict("records"),
            extra_metadata={
                "generated_by": current_user.username,
                "dataset_name": request.dataset_name,
                "records": synthetic_data.to_dict("records"),
            },
        )

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка генерации синтетических данных: {str(e)}")


@router.get("/dataset-stats", summary="Статистика datasets")
async def get_dataset_stats(current_user: User = Depends(require_role("developer", "analyst"))):
    try:
        # Кешированные опции фильтров - не нужно генерировать данные
        filter_options = data_generator.get_filter_options()
        
        # Возвращаем только опции фильтров без генерации реальных данных
        return filter_options

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка получения статистики: {str(e)}")


@router.get("/generated-history", summary="История сгенерированных данных")
async def get_generated_history(
    limit: int = 50,
    current_user: User = Depends(require_role("developer", "analyst")),
    db: Session = Depends(get_db),
):
    try:
        rows = crud.list_generated_data(db, limit=limit)
        return {
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
                        "records_count": len((r.extra_metadata or {}).get("records", [])),
                    },
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                }
                for r in rows
            ],
            "count": len(rows),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка истории generated data: {str(e)}")


@router.get("/gan-checkpoints", summary="Список доступных чекпоинтов")
async def get_gan_checkpoints(
    current_user: User = Depends(require_role("developer", "analyst")),
    db: Session = Depends(get_db),
):
    try:
        checkpoints = crud.list_checkpoints(db, limit=100, only_with_binary=True)
        
        # Возвращаем минимальную информацию без бинарных данных для быстрой загрузки
        return {
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

        payload_hex = (checkpoint.metrics_json or {}).get("binary")
        if not payload_hex:
            raise HTTPException(status_code=400, detail="В БД отсутствует бинарный payload чекпоинта")

        checkpoint_bytes = bytes.fromhex(payload_hex)
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp_file:
            tmp_file.write(checkpoint_bytes)
            tmp_file_path = tmp_file.name

        success = gan_service.load_pretrained_model(tmp_file_path)
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
        return {"status": "deleted", "id": checkpoint_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка удаления чекпоинта: {str(e)}")


@router.post("/run-ab-test-simulation", summary="Запустить симуляцию A/B теста")
async def run_ab_test_simulation(
    request: dict,
    current_user: User = Depends(require_role("developer", "analyst")),
):
    try:
        from backend.services.ab_test_simulator import ABTestSimulator

        simulator = ABTestSimulator(platform)
        simulator.simulate_test(
            request["test_id"],
            None,
            request.get("user_count", 1000),
        )
        return {"status": "simulation_started", "message": "Симуляция A/B теста запущена"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка симуляции: {str(e)}")


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
        
        return {
            "id": dataset.id,
            "data_type": dataset.data_type,
            "sample_count": dataset.sample_count,
            "dataset_name": (dataset.extra_metadata or {}).get("dataset_name"),
            "records": (dataset.extra_metadata or {}).get("records", []),
            "preview_json": dataset.preview_json,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка получения датасета: {str(e)}")
