from __future__ import annotations

import asyncio
import threading
from typing import Any, Dict, Optional

from fastapi import HTTPException
from sqlalchemy.orm import Session

from backend.microservices.ab_testing_core.managers import AdaptiveABTestingPlatform
from backend.microservices.database import crud
from backend.microservices.services.ab_test_simulator import run_ab_test_simulation


class ABPlatformProvider:
    """Потокобезопасный провайдер singleton-платформы A/B."""

    _instance: Optional[AdaptiveABTestingPlatform] = None
    _lock = threading.Lock()

    @classmethod
    def get(cls) -> AdaptiveABTestingPlatform:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = AdaptiveABTestingPlatform()
        return cls._instance


class ABSimulationOrchestrator:
    """Единая точка запуска симуляций (устраняет дубли route-логики)."""

    @staticmethod
    def resolve_simulation_params(
        *,
        db: Session,
        test_id: str,
        dataset_id: Optional[int],
        user_count: Optional[int],
        strategy: Optional[str],
        simulation_minutes: Optional[int],
        variant_effects: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        test = crud.get_test(db, test_id)
        if not test:
            raise HTTPException(status_code=404, detail="Тест не найден")

        if test.status not in ["prepared", "paused"]:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Нельзя запустить симуляцию. Текущий статус: {test.status}. "
                    "Тест должен быть в статусе 'prepared' или 'paused'"
                ),
            )

        resolved_dataset_id = dataset_id or test.dataset_id
        if not resolved_dataset_id:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Для запуска симуляции требуется synthetic dataset. "
                    "Сначала сгенерируйте данные через GAN и привяжите dataset_id к тесту."
                ),
            )

        dataset = crud.get_generated_data_by_id(db, int(resolved_dataset_id))
        if dataset is None:
            raise HTTPException(status_code=400, detail=f"Датасет {resolved_dataset_id} не найден")
        if dataset.data_type != "synthetic":
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Датасет {resolved_dataset_id} имеет тип '{dataset.data_type}'. "
                    "Для A/B симуляции разрешены только synthetic datasets."
                ),
            )

        resolved_user_count = int(user_count or test.sample_size or 1000)
        resolved_strategy = str(strategy or test.traffic_split_type or "fixed")
        resolved_minutes = int(simulation_minutes or test.simulation_duration_minutes or 20)

        resolved_variant_effects = variant_effects
        if resolved_variant_effects is None and test.extra_config:
            resolved_variant_effects = test.extra_config.get("variant_effects")

        if dataset.sample_count < resolved_user_count:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Недостаточно синтетических данных в dataset {resolved_dataset_id}: "
                    f"нужно {resolved_user_count}, доступно {dataset.sample_count}. "
                    "Сгенерируйте больше данных через GAN или уменьшите sample_size/user_count."
                ),
            )

        return {
            "test": test,
            "dataset_id": int(resolved_dataset_id),
            "user_count": resolved_user_count,
            "strategy": resolved_strategy,
            "simulation_minutes": resolved_minutes,
            "variant_effects": resolved_variant_effects,
        }

    @staticmethod
    def mark_simulation_started(db: Session, *, test_id: str, dataset_id: int) -> None:
        crud.update_test_status(db, test_id, "active")
        crud.update_test_simulation_status(db, test_id, "running")
        test = crud.get_test(db, test_id)
        if test:
            test.dataset_id = dataset_id
            db.commit()

    @staticmethod
    async def run_simulation_job(
        *,
        test_id: str,
        dataset_id: int,
        user_count: int,
        real_world_days: int,
        simulation_minutes: int,
        strategy: str,
        variant_effects: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        return await run_ab_test_simulation(
            test_id=test_id,
            dataset_id=dataset_id,
            user_count=user_count,
            real_world_days=real_world_days,
            simulation_minutes=simulation_minutes,
            strategy=strategy,
            variant_effects=variant_effects,
        )

    @staticmethod
    def run_simulation_sync_wrapper(**kwargs: Any) -> None:
        asyncio.run(ABSimulationOrchestrator.run_simulation_job(**kwargs))
