# backend/api/routes/data.py
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime
import asyncio
from pydantic import BaseModel

from backend.services.traffic_generator.data_generator import RealisticDataGenerator
from backend.services.gan_integration import gan_service
from backend.services.evaluator import GANEvaluator
from backend.ab_testing.managers import AdaptiveABTestingPlatform

platform = AdaptiveABTestingPlatform() 

router = APIRouter(prefix="/api/v1/data", tags=["Data Generation"])

data_generator = RealisticDataGenerator()

class DataGenerationRequest(BaseModel):
    num_samples: int = Field(1000, ge=100, le=100000, description="Количество samples")
    save_to_file: bool = Field(False, description="Сохранить в файл")
    include_evaluation: bool = Field(True, description="Включить оценку качества")

class GANTrainingRequest(BaseModel):
    epochs: int = Field(50, ge=10, le=500, description="Количество эпох")
    real_data_samples: int = Field(50000, ge=1000, le=100000, description="Samples для обучения")
    save_checkpoint: bool = Field(True, description="Сохранить чекпоинт")

class SyntheticDataRequest(BaseModel):
    num_users: int = Field(10000, ge=100, le=100000, description="Количество пользователей")
    evaluation_metrics: bool = Field(True, description="Рассчитать метрики качества")

class LoadCheckpointRequest(BaseModel):
    checkpoint_name: str

@router.post("/generate-real", summary="Сгенерировать реальные данные")
async def generate_real_data(request: DataGenerationRequest):
    try:
        real_data = data_generator.generate_dataset(request.num_samples)
        
        result = {
            "generated_samples": len(real_data),
            "features": list(real_data.columns),
            "data_preview": real_data.head(10).to_dict('records')
        }
        
        if request.include_evaluation:
            stats = real_data.describe().to_dict()
            result["statistics"] = stats
        
        if request.save_to_file:
            filename = f"real_data_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
            real_data.to_csv(filename, index=False)
            result["saved_file"] = filename
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка генерации данных: {str(e)}")

@router.post("/train-gan", summary="Обучить GAN модель")
async def train_gan_model(request: GANTrainingRequest, background_tasks: BackgroundTasks):
    try:
        # Генерация данных для обучения
        real_data = data_generator.generate_dataset(request.real_data_samples)
        
        def train_in_background():
            try:
                gan_service.train_gan(real_data, request.epochs)
                if request.save_checkpoint:
                    gan_service.gan_model._save_checkpoint(f"gan_trained_{datetime.now().strftime('%Y%m%d_%H%M')}")
            except Exception as e:
                print(f"Background training error: {e}")
        
        background_tasks.add_task(train_in_background)
        
        return {
            "status": "training_started",
            "epochs": request.epochs,
            "real_data_samples": len(real_data),
            "message": "GAN модель начала обучение в фоновом режиме"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка обучения GAN: {str(e)}")

@router.get("/gan-status", summary="Статус GAN модели")
async def get_gan_status():
    try:
        status = gan_service.get_status()
        return status
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка получения статуса: {str(e)}")

@router.post("/generate-synthetic", summary="Сгенерировать синтетические данные")
async def generate_synthetic_data(request: SyntheticDataRequest):
    try:
        if not gan_service.is_trained:
            raise HTTPException(status_code=400, detail="GAN модель не обучена. Сначала обучите модель.")
        
        # Генерация синтетических данных
        synthetic_data = gan_service.generate_synthetic_data(request.num_users)
        
        if synthetic_data is None:
            raise HTTPException(status_code=500, detail="Ошибка генерации синтетических данных")
        
        result = {
            "synthetic_samples": len(synthetic_data),
            "features": list(synthetic_data.columns),
            "synthetic_preview": synthetic_data.head(10).to_dict('records')
        }
        
        # ВРЕМЕННО ОТКЛЮЧАЕМ EVALUATION METRICS
        # if request.evaluation_metrics:
        #     real_data = data_generator.generate_dataset(min(10000, request.num_users))
        #     evaluator = GANEvaluator(real_data, synthetic_data)
        #     quality_metrics = evaluator.evaluate_quality()
        #     result["quality_metrics"] = quality_metrics
        #     result["fid_score"] = evaluator.calculate_fid_score()
        
        # Сохранение данных
        synth_filename = f"synthetic_data_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
        synthetic_data.to_csv(synth_filename, index=False)
        result["saved_file"] = synth_filename
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка генерации синтетических данных: {str(e)}")

@router.post("/load-pretrained", summary="Загрузить предобученную модель")
async def load_pretrained_model(checkpoint_path: str):
    try:
        success = gan_service.load_pretrained_model(checkpoint_path)
        
        if success:
            return {
                "status": "success",
                "message": f"Модель загружена из {checkpoint_path}",
                "is_trained": gan_service.is_trained
            }
        else:
            raise HTTPException(status_code=400, detail="Не удалось загрузить модель")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка загрузки модели: {str(e)}")

@router.get("/dataset-stats", summary="Статистика datasets")
async def get_dataset_stats():
    try:
        sample_real_data = data_generator.generate_dataset(1000)
        real_stats = sample_real_data.describe().to_dict()
        
        return {
            "real_data_statistics": real_stats,
            "available_features": list(sample_real_data.columns),
            "data_types": {col: str(dtype) for col, dtype in sample_real_data.dtypes.items()}
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка получения статистики: {str(e)}")
    
@router.get("/gan-checkpoints", summary="Список доступных чекпоинтов")
async def get_gan_checkpoints():
    try:
        print("🔍 DEBUG: Getting GAN status for checkpoints...")
        status = gan_service.get_status()
        print(f"🔍 DEBUG: GAN status checkpoints: {status.get('checkpoints', [])}")
        print(f"🔍 DEBUG: Available checkpoints count: {status.get('available_checkpoints', 0)}")
        
        return {
            "checkpoints": status.get("checkpoints", []),
            "count": status.get("available_checkpoints", 0)
        }
    except Exception as e:
        print(f"❌ DEBUG: Error in get_gan_checkpoints: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка получения чекпоинтов: {str(e)}")

@router.post("/gan-load-checkpoint", summary="Загрузить чекпоинт")
async def load_gan_checkpoint(request: LoadCheckpointRequest):  
    try:
        success = gan_service.load_pretrained_model(request.checkpoint_name)
        
        if success:
            return {
                "status": "success",
                "message": f"Модель загружена из {request.checkpoint_name}",
                "is_trained": gan_service.is_trained
            }
        else:
            raise HTTPException(status_code=400, detail="Не удалось загрузить модель")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка загрузки модели: {str(e)}")
    
@router.post("/run-ab-test-simulation", summary="Запустить симуляцию A/B теста")
async def run_ab_test_simulation(request: dict):
    try:
        from backend.services.ab_test_simulator import ABTestSimulator
        from backend.api.routes.tests import platform 
        
        simulator = ABTestSimulator(platform)
        simulator.simulate_test(
            request['test_id'], 
            None,
            request.get('user_count', 1000)
        )
        return {"status": "simulation_started", "message": "Симуляция A/B теста запущена"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка симуляции: {str(e)}")