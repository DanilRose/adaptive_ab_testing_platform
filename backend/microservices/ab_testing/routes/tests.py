# backend/api/routes/tests.py

from typing import Dict, List, Optional
import uuid

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from pydantic import BaseModel, Field, validator
from sqlalchemy.orm import Session

from backend.microservices.ab_testing_core.core import MetricType, TestConfig
from backend.microservices.ab_testing_core.statistics import SampleSizeCalculator
from backend.microservices.ab_testing.service import ABPlatformProvider
from backend.microservices.auth_core.models import User
from backend.microservices.auth_core.service import get_current_user, require_permission
from backend.microservices.database import crud
from backend.microservices.database.session import get_db
from backend.microservices.ab_testing import ABTestLifecycleService

router = APIRouter(prefix="/api/v1/tests", tags=["A/B Tests"])

platform = ABPlatformProvider.get()


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
    dataset_id: int = Field(..., description="ID синтетического датасета")
    simulation_user_count: Optional[int] = Field(None, ge=100, description="Количество пользователей для симуляции")
    simulation_duration_minutes: Optional[int] = Field(None, ge=1, le=180, description="Длительность симуляции в минутах")
    traffic_split_type: str = Field("fixed", description="Стратегия трафика: fixed | adaptive")
    variant_effects: Optional[Dict] = Field(None, description="Эффекты вариантов, например: {'B': {'conversion': 1.15}}")
    analysis_mode: str = Field("fixed_experiment", description="Режим анализа: fixed_experiment | adaptive_bandit")
    guardrails_config: Optional[Dict] = Field(
        None,
        description=(
            "Guardrails-конфиг, например: "
            "{'latency_ms': {'threshold': 5, 'direction': 'max_increase'}}"
        ),
    )
    early_stopping_enabled: bool = Field(
        False,
        description="Включить раннюю остановку симуляции (sequential success/futility)",
    )

    @validator("variants")
    def validate_variants(cls, v):
        if not isinstance(v, list):
            raise ValueError(f"Variants must be list, got {type(v)}")
        return v

    @validator("analysis_mode")
    def validate_analysis_mode(cls, v: str):
        allowed = {"fixed_experiment", "adaptive_bandit"}
        if v not in allowed:
            raise ValueError(f"analysis_mode должен быть одним из: {', '.join(sorted(allowed))}")
        return v


class UserAssignmentRequest(BaseModel):
    user_id: str = Field(..., description="ID пользователя")
    user_context: Optional[Dict] = Field(None, description="Контекст пользователя")


class MetricRecordRequest(BaseModel):
    session_id: str = Field(..., description="ID сессии")
    metric_name: str = Field(..., description="Название метрики")
    value: float = Field(..., description="Значение метрики")
    event_id: Optional[str] = Field(None, description="Уникальный ID события для идемпотентности")


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
    variant_effects: Optional[Dict] = Field(None, description="Эффекты вариантов: {'B': {'conversion': 1.15}} = +15% к конверсии")


class TestArchiveRequest(BaseModel):
    reason: Optional[str] = Field(None, description="Причина архивирования")


def _default_baseline_for_metric(metric_type: MetricType, primary_metric: str) -> tuple[float, float]:
    pm = (primary_metric or "").lower()

    if metric_type == MetricType.BINARY:
        # Типичный baseline для conversion-like метрик
        return 0.10, 0.30

    # Небольшие эвристики для непрерывных/ratio метрик
    if any(k in pm for k in ["revenue", "gmv", "amount", "income"]):
        return 100.0, 60.0
    if any(k in pm for k in ["ctr", "cr", "rate", "ratio"]):
        return 0.10, 0.10

    return 1.0, 1.0


def _resolve_sample_size(
    *,
    metric_type: MetricType,
    primary_metric: str,
    explicit_sample_size: Optional[int],
    min_effect_size_fraction: float,
    confidence_level: float,
    power: float,
) -> int:
    if explicit_sample_size and explicit_sample_size > 0:
        return int(explicit_sample_size)

    alpha = max(1e-6, 1.0 - float(confidence_level))
    mde_percent = max(0.1, float(min_effect_size_fraction) * 100.0)

    baseline_mean, baseline_std = _default_baseline_for_metric(metric_type, primary_metric)

    if metric_type == MetricType.BINARY:
        return int(
            SampleSizeCalculator.calculate_sample_size_for_binary(
                baseline_conversion=max(1e-4, min(0.99, baseline_mean)),
                mde_percent=mde_percent,
                alpha=alpha,
                power=float(power),
            )
        )

    # Для continuous/ratio используем общий расчёт по MDE%
    return int(
        SampleSizeCalculator.calculate_sample_size(
            baseline_mean=max(1e-6, baseline_mean),
            baseline_std=max(1e-6, baseline_std),
            mde_percent=mde_percent,
            alpha=alpha,
            power=float(power),
            two_tailed=True,
        )
    )


@router.post("/", summary="Создать новый A/B тест")
async def create_test(
    request: TestCreateRequest,
    current_user: User = Depends(require_permission("AB_тесты_создание")),
    db: Session = Depends(get_db),
):
    try:
        test_id = f"test_{uuid.uuid4().hex[:8]}"

        # Валидация: variants должен быть списком строк
        if not isinstance(request.variants, list):
            raise ValueError(f"variants должен быть списком, получен {type(request.variants)}")

        if not all(isinstance(v, str) for v in request.variants):
            raise ValueError("Все варианты должны быть строками")

        metric_type = MetricType(request.metric_type)
        resolved_sample_size = _resolve_sample_size(
            metric_type=metric_type,
            primary_metric=request.primary_metric,
            explicit_sample_size=request.sample_size,
            min_effect_size_fraction=request.min_effect_size,
            confidence_level=request.confidence_level,
            power=request.power,
        )

        dataset = crud.get_generated_data_by_id(db, int(request.dataset_id))
        if dataset is None:
            raise ValueError(f"Датасет {request.dataset_id} не найден")
        if dataset.data_type != "synthetic":
            raise ValueError(
                f"Датасет {request.dataset_id} имеет тип '{dataset.data_type}'. Для A/B тестов разрешены только synthetic datasets"
            )
        # Если расчётный sample_size превышает размер датасета — автоматически ограничиваем.
        # Минимум 200 записей необходимо для статистически значимого результата.
        if dataset.sample_count < 200:
            raise ValueError(
                f"Датасет {request.dataset_id} слишком мал: доступно {dataset.sample_count} записей, "
                f"минимум 200 для A/B теста."
            )
        if resolved_sample_size > dataset.sample_count:
            # Автоматически снижаем sample_size до размера датасета
            resolved_sample_size = dataset.sample_count

        config = TestConfig(
            test_id=test_id,
            variants=request.variants,
            primary_metric=request.primary_metric,
            metric_type=metric_type,
            sample_size=resolved_sample_size,
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
            sample_size=resolved_sample_size,
            confidence_level=request.confidence_level,
            power=request.power,
            min_effect_size=request.min_effect_size,
            dataset_id=request.dataset_id,
            simulation_duration_minutes=request.simulation_duration_minutes or 20,
            traffic_split_type=request.traffic_split_type,
            analysis_mode=request.analysis_mode,
            analysis_validity=("exploration_only" if request.analysis_mode == "adaptive_bandit" or request.traffic_split_type == "adaptive" else "valid_for_inference"),
            created_by_user_id=current_user.id,
            status="prepared",  # Новый тест создается в статусе "prepared"
            extra_config={
                "variant_effects": request.variant_effects,
                "early_stopping_enabled": bool(request.early_stopping_enabled),
            },
            guardrails_config=request.guardrails_config,
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
    current_user: User = Depends(require_permission("AB_тесты_управление")),
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
    current_user: User = Depends(require_permission("AB_тесты_управление")),
):
    try:
        record_result = platform.record_user_metric(
            session_id=request.session_id,
            metric_name=request.metric_name,
            value=request.value,
            event_id=request.event_id,
        )

        return {
            "status": "recorded" if record_result.get("deduplicated") is not True else "duplicate_ignored",
            "deduplicated": bool(record_result.get("deduplicated", False)),
            "event_id": record_result.get("event_id"),
            "message": f"Метрика '{request.metric_name}' обработана для сессии {request.session_id}",
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/sessions/complete", summary="Завершить сессию пользователя")
async def complete_session(
    request: SessionCompleteRequest,
    current_user: User = Depends(require_permission("AB_тесты_управление")),
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
    current_user: User = Depends(require_permission("AB_тесты_удаление_и_архивация")),
    db: Session = Depends(get_db),
):
    try:
        result = platform.stop_test(test_id, request.reason)
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
                "analysis_mode": t.analysis_mode,
                "analysis_validity": t.analysis_validity,
                "guardrails_status": t.guardrails_status,
                "dataset_id": t.dataset_id,
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
    current_user: User = Depends(require_permission("AB_тесты_удаление_и_архивация")),
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
    current_user: User = Depends(require_permission("AB_тесты_управление")),
    db: Session = Depends(get_db),
):
    """
    Запуск симуляции A/B теста.
    Переводит тест из статуса 'prepared' в 'active'.
    """
    try:
        prepared = ABTestLifecycleService.prepare_simulation_start(
            db=db,
            test_id=test_id,
            dataset_id=request.dataset_id,
            user_count=request.user_count,
            strategy=request.strategy,
            simulation_minutes=request.simulation_minutes,
            variant_effects=request.variant_effects,
        )

        simulation_meta = ABTestLifecycleService.enqueue_simulation(
            background_tasks=background_tasks,
            platform=platform,
            test_id=test_id,
            prepared=prepared,
        )

        return {
            "status": "simulation_started",
            "message": f"Симуляция запущена для теста {test_id}",
            **simulation_meta,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{test_id}/pause", summary="Поставить тест на паузу")
async def pause_test(
    test_id: str,
    request: Optional[TestPauseRequest] = None,
    current_user: User = Depends(require_permission("AB_тесты_управление")),
    db: Session = Depends(get_db),
):
    """
    Ставит тест на паузу.
    Переводит тест из статуса 'active' в 'paused'.
    """
    try:
        result = ABTestLifecycleService.pause_test(db, test_id=test_id)
        print(f"⏸️ Test {test_id} paused by user {current_user.username}")
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{test_id}/resume", summary="Продолжить тест")
async def resume_test(
    test_id: str,
    current_user: User = Depends(require_permission("AB_тесты_управление")),
    db: Session = Depends(get_db),
):
    """
    Продолжает тест после паузы.
    Переводит тест из статуса 'paused' в 'active'.
    """
    try:
        result = ABTestLifecycleService.resume_test(db, test_id=test_id)
        print(f"▶️ Test {test_id} resumed by user {current_user.username}")
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{test_id}/delete-with-option", summary="Удалить тест с опцией архивирования")
async def delete_test_with_option(
    test_id: str,
    request: TestDeleteRequest,
    current_user: User = Depends(require_permission("AB_тесты_удаление_и_архивация")),
    db: Session = Depends(get_db),
):
    """
    Удаляет тест с опцией перемещения в архив или в подготовленные.
    """
    try:
        result = ABTestLifecycleService.move_test(db, test_id=test_id, move_to_archived=request.move_to_archived)
        if request.move_to_archived:
            print(f"🗄️ Test {test_id} moved to archive by user {current_user.username}")
        else:
            print(f"📁 Test {test_id} moved to prepared by user {current_user.username}")
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{test_id}/archive", summary="Переместить тест в архив")
async def archive_test(
    test_id: str,
    request: Optional[TestArchiveRequest] = None,
    reason: Optional[str] = Query(None, description="Причина архивирования (legacy query-param)"),
    current_user: User = Depends(require_permission("AB_тесты_удаление_и_архивация")),
    db: Session = Depends(get_db),
):
    """
    Перемещает тест в архив.
    Поддерживает reason как в body, так и в query (backward compatibility).
    """
    try:
        final_reason = (request.reason if request and request.reason is not None else reason)
        result = ABTestLifecycleService.archive_test(db, test_id=test_id, reason=final_reason)
        print(f"🗄️ Test {test_id} archived by user {current_user.username}")
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/{test_id}/permanent", summary="Полностью удалить тест из архива")
async def permanently_delete_test(
    test_id: str,
    current_user: User = Depends(require_permission("AB_тесты_удаление_и_архивация")),
    db: Session = Depends(get_db),
):
    """
    Полностью удаляет тест из базы данных.
    """
    try:
        result = ABTestLifecycleService.permanently_delete_test(db, test_id=test_id)
        print(f"🗑️ Test {test_id} permanently deleted by user {current_user.username}")
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))