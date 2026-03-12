# backend/api/routes/tests.py

from typing import Dict, List, Optional
import uuid

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field, validator
from sqlalchemy.orm import Session

from backend.ab_testing.core import MetricType, TestConfig
from backend.api.platform_instance import get_platform
from backend.auth.models import User
from backend.auth.service import get_current_user, require_role
from backend.database import crud
from backend.database.session import get_db

router = APIRouter(prefix="/api/v1/tests", tags=["A/B Tests"])

platform = get_platform()


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
    dataset_id: Optional[int] = Field(None, description="ID синтетического датасета")
    simulation_user_count: Optional[int] = Field(None, ge=100, description="Количество пользователей для симуляции")
    simulation_duration_minutes: Optional[int] = Field(None, ge=1, le=180, description="Длительность симуляции в минутах")
    traffic_split_type: str = Field("fixed", description="Стратегия трафика: fixed | adaptive")

    @validator("variants")
    def validate_variants(cls, v):
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


class TestPauseRequest(BaseModel):
    reason: Optional[str] = Field(None, description="Причина паузы")


class TestDeleteRequest(BaseModel):
    move_to_archived: bool = Field(True, description="Переместить в архив (True) или в подготовленные (False)")


class StartSimulationRequest(BaseModel):
    dataset_id: Optional[int] = Field(None, description="ID синтетического датасета")
    user_count: Optional[int] = Field(None, ge=100, description="Количество пользователей для симуляции")
    strategy: str = Field("fixed", description="Стратегия трафика: fixed | adaptive")
    simulation_minutes: Optional[int] = Field(None, ge=1, le=180, description="Длительность симуляции в минутах")


@router.post("/", summary="Создать новый A/B тест")
async def create_test(
    request: TestCreateRequest,
    current_user: User = Depends(require_role("developer", "analyst")),
    db: Session = Depends(get_db),
):
    try:
        test_id = f"test_{uuid.uuid4().hex[:8]}"

        # Валидация: variants должен быть списком строк
        if not isinstance(request.variants, list):
            raise ValueError(f"variants должен быть списком, получен {type(request.variants)}")

        if not all(isinstance(v, str) for v in request.variants):
            raise ValueError("Все варианты должны быть строками")

        config = TestConfig(
            test_id=test_id,
            variants=request.variants,
            primary_metric=request.primary_metric,
            metric_type=MetricType(request.metric_type),
            sample_size=request.sample_size,
            confidence_level=request.confidence_level,
            power=request.power,
            min_effect_size=request.min_effect_size,
        )

        platform.test_registry.register_test(config, current_user.username, request.description or "")
        platform.test_manager.create_test(config)

        # Создаем тест в БД со статусом "prepared" и test_name из запроса
        crud.create_ab_test(
            db,
            test_id=test_id,
            test_name=request.test_name,  # Используем user-friendly имя из запроса
            description=request.description,
            variants=request.variants,
            primary_metric=request.primary_metric,
            metric_type=request.metric_type,
            sample_size=request.sample_size,
            confidence_level=request.confidence_level,
            power=request.power,
            min_effect_size=request.min_effect_size,
            dataset_id=request.dataset_id,
            simulation_duration_minutes=request.simulation_duration_minutes or 20,
            traffic_split_type=request.traffic_split_type,
            created_by_user_id=current_user.id,
            status="prepared",  # Новый тест создается в статусе "prepared"
        )

        return {
            "test_id": test_id,
            "test_name": request.test_name,  # Возвращаем user-friendly имя
            "status": "prepared",
            "message": f"A/B тест '{request.test_name}' успешно создан и готов к запуску",
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{test_id}/assign", summary="Назначить пользователя в тест")
async def assign_user(
    test_id: str,
    request: UserAssignmentRequest,
    current_user: User = Depends(require_role("developer", "analyst")),
):
    try:
        assignment = platform.assign_user_to_test(
            test_id=test_id,
            user_id=request.user_id,
            user_context=request.user_context,
        )

        return {
            "assignment": assignment,
            "message": f"Пользователь {request.user_id} назначен в вариант {assignment['variant']}",
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/metrics/record", summary="Записать метрику пользователя")
async def record_metric(
    request: MetricRecordRequest,
    current_user: User = Depends(require_role("developer", "analyst")),
):
    try:
        platform.record_user_metric(
            session_id=request.session_id,
            metric_name=request.metric_name,
            value=request.value,
        )

        return {
            "status": "recorded",
            "message": f"Метрика '{request.metric_name}' записана для сессии {request.session_id}",
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/sessions/complete", summary="Завершить сессию пользователя")
async def complete_session(
    request: SessionCompleteRequest,
    current_user: User = Depends(require_role("developer", "analyst")),
):
    try:
        platform.complete_user_session(
            session_id=request.session_id,
            final_metrics=request.final_metrics,
        )

        return {
            "status": "completed",
            "message": f"Сессия {request.session_id} завершена",
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{test_id}/results", summary="Получить результаты теста")
async def get_test_results(test_id: str, current_user: User = Depends(get_current_user)):
    try:
        results = platform.get_test_results(test_id)
        return results
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{test_id}/stop", summary="Остановить тест")
async def stop_test(
    test_id: str,
    request: TestStopRequest,
    current_user: User = Depends(require_role("developer", "analyst")),
    db: Session = Depends(get_db),
):
    try:
        result = platform.stop_test(test_id, request.reason)
        crud.update_ab_test_status(db, test_id=test_id, status="archived")
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/", summary="Получить список всех тестов с разделением по статусам")
async def get_all_tests_endpoint(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    try:
        # Получаем все тесты из БД
        all_db_tests = crud.get_all_tests(db, limit=500)

        # Разделяем по статусам
        prepared_tests = []
        active_tests = []
        paused_tests = []
        completed_tests = []
        archived_tests = []

        for t in all_db_tests:
            test_data = {
                "test_id": t.test_id,
                "test_name": t.test_name,  # Возвращаем user-friendly имя
                "description": t.description,
                "status": t.status,
                "simulation_status": t.simulation_status,
                "variants": t.variants,
                "primary_metric": t.primary_metric,
                "metric_type": t.metric_type,
                "sample_size": t.sample_size,
                "confidence_level": t.confidence_level,
                "power": t.power,
                "min_effect_size": t.min_effect_size,
                "total_users": t.total_users,
                "completion_percentage": t.completion_percentage,
                "archive_reason": t.archive_reason,
                "created_at": t.created_at.isoformat() if t.created_at else None,
                "updated_at": t.updated_at.isoformat() if t.updated_at else None,
            }

            if t.status == "prepared":
                prepared_tests.append(test_data)
            elif t.status == "active":
                active_tests.append(test_data)
            elif t.status == "paused":
                paused_tests.append(test_data)
            elif t.status == "completed":
                completed_tests.append(test_data)
            elif t.status == "archived":
                archived_tests.append(test_data)

        return {
            "prepared_tests": prepared_tests,
            "active_tests": active_tests,
            "paused_tests": paused_tests,
            "completed_tests": completed_tests,
            "archived_tests": archived_tests,
            "counts": {
                "prepared": len(prepared_tests),
                "active": len(active_tests),
                "paused": len(paused_tests),
                "completed": len(completed_tests),
                "archived": len(archived_tests),
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/stats/platform", summary="Статистика платформы")
async def get_platform_stats(current_user: User = Depends(get_current_user)):
    try:
        stats = platform.get_platform_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/history", summary="История завершенных тестов")
async def get_test_history(limit: int = 50, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    try:
        history = platform.test_registry.get_test_history(limit)
        archived = [t for t in crud.list_ab_tests(db, limit=limit) if t.status == "archived"]
        return {
            "test_history": history,
            "db_history": [
                {
                    "test_id": t.test_id,
                    "test_name": t.test_name,
                    "status": t.status,
                    "updated_at": t.updated_at.isoformat() if t.updated_at else None,
                }
                for t in archived
            ],
            "count": len(history),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/{test_id}", summary="Удалить тест")
async def delete_test(
    test_id: str,
    current_user: User = Depends(require_role("developer", "analyst")),
    db: Session = Depends(get_db),
):
    """Permanently delete a test"""
    try:
        try:
            platform.stop_test(test_id, "Manual deletion")
        except Exception:
            pass
        test = crud.get_test(db, test_id)
        if not test:
            raise HTTPException(status_code=404, detail="Test not found")

        crud.update_test_status(db, test_id, "deleted")

        return {"message": "Test deleted successfully", "test_id": test_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ============================================================================
# НОВЫЕ ENDPOINTS ДЛЯ УПРАВЛЕНИЯ ТЕСТАМИ
# ============================================================================

@router.post("/{test_id}/start-simulation", summary="Запустить симуляцию теста")
async def start_simulation(
    test_id: str,
    request: StartSimulationRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(require_role("developer", "analyst")),
    db: Session = Depends(get_db),
):
    """
    Запуск симуляции A/B теста.
    Переводит тест из статуса 'prepared' в 'active'.
    """
    try:
        from backend.services.ab_test_simulator import run_ab_test_simulation as run_sim_v2

        # Проверяем существование теста
        test = crud.get_test(db, test_id)
        if not test:
            raise HTTPException(status_code=404, detail="Тест не найден")

        # Проверяем, что тест в статусе prepared или paused
        if test.status not in ["prepared", "paused"]:
            raise HTTPException(
                status_code=400,
                detail=f"Нельзя запустить симуляцию. Текущий статус: {test.status}. Тест должен быть в статусе 'prepared' или 'paused'"
            )

        # Получаем dataset_id: request -> тест -> последний synthetic
        dataset_id = request.dataset_id or test.dataset_id
        if not dataset_id:
            latest_dataset = crud.get_latest_generated_data_by_type(db, "synthetic")
            if not latest_dataset:
                raise HTTPException(
                    status_code=400,
                    detail="Нет синтетических данных! Сначала сгенерируйте данные в GAN Manager или привяжите датасет к тесту."
                )
            dataset_id = latest_dataset.id

        user_count = request.user_count or test.sample_size or 1000
        strategy = request.strategy or test.traffic_split_type or "fixed"
        simulation_minutes = request.simulation_minutes or test.simulation_duration_minutes or 20

        # Обновляем статус теста на 'active' и simulation_status на 'running'
        crud.update_test_status(db, test_id, "active")
        crud.update_test_simulation_status(db, test_id, "running")
        
        # Сохраняем dataset_id в тесте (важно для симуляции)
        test.dataset_id = dataset_id
        db.commit()

        # Запускаем симуляцию асинхронно
        async def run_simulation_task():
            try:
                results = await run_sim_v2(
                    test_id=test_id,
                    dataset_id=dataset_id,
                    user_count=user_count,
                    real_world_days=test.real_world_duration_days or 14,
                    simulation_minutes=simulation_minutes,
                    strategy=strategy
                )

                # После завершения обновляем статус
                crud.update_test_simulation_status(db, test_id, None)  # Убираем simulation_status
                crud.update_test_status(db, test_id, "completed")  # Устанавливаем статус completed
                platform.refresh_test_from_database(test_id)
                platform.force_update_test_statistics(test_id)

                print(f"✅ Simulation completed for test {test_id}")
                return results
            except Exception as e:
                print(f"❌ Simulation failed: {str(e)}")
                # При ошибке останавливаем тест
                crud.update_test_simulation_status(db, test_id, None)
                crud.update_test_status(db, test_id, "prepared")
                raise

        background_tasks.add_task(run_simulation_task)

        return {
            "status": "simulation_started",
            "message": f"Симуляция запущена для теста {test_id}",
            "test_id": test_id,
            "dataset_id": dataset_id,
            "user_count": user_count,
            "simulation_minutes": simulation_minutes,
            "strategy": strategy,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{test_id}/pause", summary="Поставить тест на паузу")
async def pause_test(
    test_id: str,
    request: Optional[TestPauseRequest] = None,
    current_user: User = Depends(require_role("developer", "analyst")),
    db: Session = Depends(get_db),
):
    """
    Ставит тест на паузу.
    Переводит тест из статуса 'active' в 'paused'.
    """
    try:
        test = crud.get_test(db, test_id)
        if not test:
            raise HTTPException(status_code=404, detail="Тест не найден")

        if test.status != "active":
            raise HTTPException(
                status_code=400,
                detail=f"Нельзя поставить на паузу. Текущий статус: {test.status}. Тест должен быть в статусе 'active'"
            )

        # Обновляем статус на 'paused' и ставим симуляцию на паузу
        crud.update_test_status(db, test_id, "paused")
        crud.update_test_simulation_status(db, test_id, "paused")

        print(f"⏸️ Test {test_id} paused by user {current_user.username}")

        return {
            "status": "paused",
            "message": f"Тест {test_id} поставлен на паузу",
            "test_id": test_id,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{test_id}/resume", summary="Продолжить тест")
async def resume_test(
    test_id: str,
    current_user: User = Depends(require_role("developer", "analyst")),
    db: Session = Depends(get_db),
):
    """
    Продолжает тест после паузы.
    Переводит тест из статуса 'paused' в 'active'.
    """
    try:
        test = crud.get_test(db, test_id)
        if not test:
            raise HTTPException(status_code=404, detail="Тест не найден")

        if test.status != "paused":
            raise HTTPException(
                status_code=400,
                detail=f"Нельзя продолжить. Текущий статус: {test.status}. Тест должен быть в статусе 'paused'"
            )

        # Обновляем статус на 'active' и возвращаем симуляцию в running
        crud.update_test_status(db, test_id, "active")
        crud.update_test_simulation_status(db, test_id, "running")

        print(f"▶️ Test {test_id} resumed by user {current_user.username}")

        return {
            "status": "resumed",
            "message": f"Тест {test_id} продолжен",
            "test_id": test_id,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{test_id}/delete-with-option", summary="Удалить тест с опцией архивирования")
async def delete_test_with_option(
    test_id: str,
    request: TestDeleteRequest,
    current_user: User = Depends(require_role("developer", "analyst")),
    db: Session = Depends(get_db),
):
    """
    Удаляет тест с опцией перемещения в архив или в подготовленные.
    """
    try:
        test = crud.get_test(db, test_id)
        if not test:
            raise HTTPException(status_code=404, detail="Тест не найден")

        if request.move_to_archived:
            # Перемещаем в архив
            crud.update_test_status(db, test_id, "archived")
            print(f"🗄️ Test {test_id} moved to archive by user {current_user.username}")
            return {
                "status": "archived",
                "message": f"Тест {test_id} перемещен в архив",
                "test_id": test_id,
            }
        else:
            # Перемещаем в подготовленные (если тест был активным или на паузе)
            crud.update_test_status(db, test_id, "prepared")
            print(f"📁 Test {test_id} moved to prepared by user {current_user.username}")
            return {
                "status": "prepared",
                "message": f"Тест {test_id} перемещен в подготовленные",
                "test_id": test_id,
            }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{test_id}/archive", summary="Переместить тест в архив")
async def archive_test(
    test_id: str,
    reason: Optional[str] = None,
    current_user: User = Depends(require_role("developer", "analyst")),
    db: Session = Depends(get_db),
):
    """
    Перемещает тест в архив.
    """
    try:
        test = crud.get_test(db, test_id)
        if not test:
            raise HTTPException(status_code=404, detail="Тест не найден")

        crud.update_test_status(db, test_id, "archived")
        if reason:
            test.archive_reason = reason
            db.commit()

        print(f"🗄️ Test {test_id} archived by user {current_user.username}")

        return {
            "status": "archived",
            "message": f"Тест {test_id} перемещен в архив",
            "test_id": test_id,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/{test_id}/permanent", summary="Полностью удалить тест из архива")
async def permanently_delete_test(
    test_id: str,
    current_user: User = Depends(require_role("developer", "analyst")),
    db: Session = Depends(get_db),
):
    """
    Полностью удаляет тест из базы данных.
    """
    try:
        test = crud.get_test(db, test_id)
        if not test:
            raise HTTPException(status_code=404, detail="Тест не найден")

        if test.status != "archived":
            raise HTTPException(
                status_code=400,
                detail=f"Нельзя удалить. Тест должен быть в архиве. Текущий статус: {test.status}"
            )

        # Удаляем тест
        db.delete(test)
        db.commit()

        print(f"🗑️ Test {test_id} permanently deleted by user {current_user.username}")

        return {
            "status": "deleted",
            "message": f"Тест {test_id} полностью удален",
            "test_id": test_id,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))