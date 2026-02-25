# backend/api/routes/tests.py

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, List, Optional
from pydantic import BaseModel, Field, validator
import uuid

from backend.ab_testing.core import MetricType, TestConfig
from backend.ab_testing.managers import AdaptiveABTestingPlatform

router = APIRouter(prefix="/api/v1/tests", tags=["A/B Tests"])

platform = AdaptiveABTestingPlatform()

class TestCreateRequest(BaseModel):
    test_name: str = Field(..., description="Название теста")
    variants: List[str] = Field(..., description="Варианты теста (A, B, C...)")
    primary_metric: str = Field(..., description="Основная метрика")
    metric_type: str = Field(..., description="Тип метрики")
    description: Optional[str] = Field(None, description="Описание теста")
    sample_size: Optional[int] = Field(None, description="Размер выборки")
    confidence_level: float = Field(0.95, ge=0.8, le=0.99, description="Уровень доверия")
    power: float = Field(0.8, ge=0.5, le=0.95, description="Мощность теста")
    min_effect_size: float = Field(0.1, ge=0.01, le=1.0, description="Минимальный размер эффекта")
    
    @validator('variants')
    def validate_variants(cls, v):
        print(f"🔍 DEBUG Pydantic validator: variants={v}, type={type(v)}")
        if not isinstance(v, list):
            raise ValueError(f"Variants must be list, got {type(v)}")
        return v

class UserAssignmentRequest(BaseModel):
    user_id: str = Field(..., description="ID пользователя")
    user_context: Optional[Dict] = Field(None, description="Контекст пользователя")

class MetricRecordRequest(BaseModel):
    session_id: str = Field(..., description="ID сессии")
    metric_name: str = Field(..., description="Название метрики")
    value: float = Field(..., description="Значение метрики")

class SessionCompleteRequest(BaseModel):
    session_id: str = Field(..., description="ID сессии")
    final_metrics: Optional[Dict[str, float]] = Field(None, description="Финальные метрики")

class TestStopRequest(BaseModel):
    reason: str = Field("Manual stop", description="Причина остановки")

@router.post("/", summary="Создать новый A/B тест")
async def create_test(request: dict):
    try:
        test_id = f"test_{uuid.uuid4().hex[:8]}"
        
        config = TestConfig(
            test_id=test_id,
            variants=request.get('variants'),
            primary_metric=request.get('primary_metric'),
            metric_type=MetricType(request.get('metric_type')),
            sample_size=request.get('sample_size'),
            confidence_level=request.get('confidence_level', 0.95),
            power=request.get('power', 0.8),
            min_effect_size=request.get('min_effect_size', 0.1)
        )
        platform.test_registry.register_test(config, "api_user", request.get('description', ''))
        platform.test_manager.create_test(config)   
        return {
            "test_id": test_id,
            "status": "created", 
            "message": f"A/B тест '{request.get('test_name')}' успешно создан"
        }
    except Exception as e:
        import traceback
        print(f"❌ DEBUG: Full traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/{test_id}/assign", summary="Назначить пользователя в тест")
async def assign_user(test_id: str, request: UserAssignmentRequest):
    try:
        assignment = platform.assign_user_to_test(
            test_id=test_id,
            user_id=request.user_id,
            user_context=request.user_context
        )
        
        return {
            "assignment": assignment,
            "message": f"Пользователь {request.user_id} назначен в вариант {assignment['variant']}"
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/metrics/record", summary="Записать метрику пользователя")
async def record_metric(request: MetricRecordRequest):
    try:
        platform.record_user_metric(
            session_id=request.session_id,
            metric_name=request.metric_name,
            value=request.value
        )
        
        return {
            "status": "recorded",
            "message": f"Метрика '{request.metric_name}' записана для сессии {request.session_id}"
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/sessions/complete", summary="Завершить сессию пользователя")
async def complete_session(request: SessionCompleteRequest):
    try:
        platform.complete_user_session(
            session_id=request.session_id,
            final_metrics=request.final_metrics
        )
        
        return {
            "status": "completed",
            "message": f"Сессия {request.session_id} завершена"
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/{test_id}/results", summary="Получить результаты теста")
async def get_test_results(test_id: str):
    try:
        results = platform.get_test_results(test_id)
        return results
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/{test_id}/stop", summary="Остановить тест")
async def stop_test(test_id: str, request: TestStopRequest):
    try:
        result = platform.stop_test(test_id, request.reason)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/", summary="Получить список активных тестов")
async def get_active_tests():
    try:
        active_tests = platform.test_registry.get_active_tests()
        return {
            "active_tests": active_tests,
            "count": len(active_tests)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/stats/platform", summary="Статистика платформы")
async def get_platform_stats():
    try:
        stats = platform.get_platform_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/history", summary="История завершенных тестов")
async def get_test_history(limit: int = 50):
    try:
        history = platform.test_registry.get_test_history(limit)
        return {
            "test_history": history,
            "count": len(history)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.delete("/{test_id}", summary="Удалить тест")
async def delete_test(test_id: str):
    try:
        # В реальном приложении добавить проверку прав
        if test_id in platform.test_manager.active_tests:
            platform.stop_test(test_id, "Manual deletion")
        
        return {
            "status": "deleted",
            "message": f"Тест {test_id} удален"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))