from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from fastapi import BackgroundTasks

from fastapi import HTTPException
from sqlalchemy.orm import Session

from backend.database import crud
from backend.database.session import SessionLocal
from backend.microservices.ab_testing.service import ABSimulationOrchestrator


class ABTestLifecycleService:
    """Сервис жизненного цикла A/B тестов и оркестрации симуляций."""

    @staticmethod
    def _ensure_test_exists(db: Session, test_id: str):
        test = crud.get_test(db, test_id)
        if not test:
            raise HTTPException(status_code=404, detail="Тест не найден")
        return test

    @staticmethod
    def pause_test(db: Session, *, test_id: str) -> Dict[str, Any]:
        test = ABTestLifecycleService._ensure_test_exists(db, test_id)
        if test.status != "active":
            raise HTTPException(
                status_code=400,
                detail=f"Нельзя поставить на паузу. Текущий статус: {test.status}. Тест должен быть в статусе 'active'",
            )

        crud.update_test_status(db, test_id, "paused")
        crud.update_test_simulation_status(db, test_id, "paused")
        return {
            "status": "paused",
            "message": f"Тест {test_id} поставлен на паузу",
            "test_id": test_id,
        }

    @staticmethod
    def resume_test(db: Session, *, test_id: str) -> Dict[str, Any]:
        test = ABTestLifecycleService._ensure_test_exists(db, test_id)
        if test.status != "paused":
            raise HTTPException(
                status_code=400,
                detail=f"Нельзя продолжить. Текущий статус: {test.status}. Тест должен быть в статусе 'paused'",
            )

        crud.update_test_status(db, test_id, "active")
        crud.update_test_simulation_status(db, test_id, "running")
        return {
            "status": "resumed",
            "message": f"Тест {test_id} продолжен",
            "test_id": test_id,
        }

    @staticmethod
    def move_test(db: Session, *, test_id: str, move_to_archived: bool) -> Dict[str, Any]:
        ABTestLifecycleService._ensure_test_exists(db, test_id)
        if move_to_archived:
            crud.update_test_status(db, test_id, "archived")
            return {
                "status": "archived",
                "message": f"Тест {test_id} перемещен в архив",
                "test_id": test_id,
            }

        crud.update_test_status(db, test_id, "prepared")
        return {
            "status": "prepared",
            "message": f"Тест {test_id} перемещен в подготовленные",
            "test_id": test_id,
        }

    @staticmethod
    def archive_test(db: Session, *, test_id: str, reason: Optional[str]) -> Dict[str, Any]:
        test = ABTestLifecycleService._ensure_test_exists(db, test_id)
        crud.update_test_status(db, test_id, "archived")
        if reason:
            test.archive_reason = reason
            db.commit()

        return {
            "status": "archived",
            "message": f"Тест {test_id} перемещен в архив",
            "test_id": test_id,
        }

    @staticmethod
    def permanently_delete_test(db: Session, *, test_id: str) -> Dict[str, Any]:
        test = ABTestLifecycleService._ensure_test_exists(db, test_id)
        if test.status != "archived":
            raise HTTPException(
                status_code=400,
                detail=f"Нельзя удалить. Тест должен быть в архиве. Текущий статус: {test.status}",
            )

        db.delete(test)
        db.commit()

        return {
            "status": "deleted",
            "message": f"Тест {test_id} полностью удален",
            "test_id": test_id,
        }

    @staticmethod
    def prepare_simulation_start(
        db: Session,
        *,
        test_id: str,
        dataset_id: Optional[int],
        user_count: Optional[int],
        strategy: Optional[str],
        simulation_minutes: Optional[int],
        variant_effects: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        resolved = ABSimulationOrchestrator.resolve_simulation_params(
            db=db,
            test_id=test_id,
            dataset_id=dataset_id,
            user_count=user_count,
            strategy=strategy,
            simulation_minutes=simulation_minutes,
            variant_effects=variant_effects,
        )

        test = resolved["test"]
        resolved_dataset_id = int(resolved["dataset_id"])
        resolved_user_count = int(resolved["user_count"])
        resolved_strategy = str(resolved["strategy"])
        resolved_minutes = int(resolved["simulation_minutes"])
        resolved_variant_effects = resolved.get("variant_effects")

        ABSimulationOrchestrator.mark_simulation_started(db, test_id=test_id, dataset_id=resolved_dataset_id)

        return {
            "test": test,
            "dataset_id": resolved_dataset_id,
            "user_count": resolved_user_count,
            "strategy": resolved_strategy,
            "simulation_minutes": resolved_minutes,
            "variant_effects": resolved_variant_effects,
        }

    @staticmethod
    def enqueue_simulation(
        *,
        background_tasks: BackgroundTasks,
        platform: Any,
        test_id: str,
        prepared: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Единая точка постановки задачи симуляции в фон (без дублирования в роутинге)."""
        test = prepared["test"]
        dataset_id = int(prepared["dataset_id"])
        user_count = int(prepared["user_count"])
        strategy = str(prepared["strategy"])
        simulation_minutes = int(prepared["simulation_minutes"])
        variant_effects = prepared.get("variant_effects")

        background_tasks.add_task(
            ABTestLifecycleService.run_simulation_task,
            test_id=test_id,
            dataset_id=dataset_id,
            user_count=user_count,
            real_world_days=test.real_world_duration_days or 14,
            simulation_minutes=simulation_minutes,
            strategy=strategy,
            variant_effects=variant_effects,
            on_success=lambda: (
                platform.refresh_test_from_database(test_id),
                platform.force_update_test_statistics(test_id),
            ),
            on_failure=lambda: None,
        )

        return {
            "test_id": test_id,
            "dataset_id": dataset_id,
            "user_count": user_count,
            "simulation_minutes": simulation_minutes,
            "strategy": strategy,
        }

    @staticmethod
    def run_simulation_task(
        *,
        test_id: str,
        dataset_id: int,
        user_count: int,
        real_world_days: int,
        simulation_minutes: int,
        strategy: str,
        variant_effects: Optional[Dict[str, Any]],
        on_success: Optional[Callable[[], None]] = None,
        on_failure: Optional[Callable[[], None]] = None,
    ) -> None:
        try:
            ABSimulationOrchestrator.run_simulation_sync_wrapper(
                test_id=test_id,
                dataset_id=dataset_id,
                user_count=user_count,
                real_world_days=real_world_days,
                simulation_minutes=simulation_minutes,
                strategy=strategy,
                variant_effects=variant_effects,
            )

            with SessionLocal() as task_db:
                crud.update_test_simulation_status(task_db, test_id, None)
                crud.update_test_status(task_db, test_id, "completed")

            if on_success:
                on_success()
        except Exception:
            with SessionLocal() as task_db:
                crud.update_test_simulation_status(task_db, test_id, None)
                crud.update_test_status(task_db, test_id, "prepared")
            if on_failure:
                on_failure()
