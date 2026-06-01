# backend/ab_testing/managers.py

import uuid
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from sqlalchemy import func
from .core import ABTestManager, AdaptiveABTest, TestConfig, MetricType, TestResult
from .traffic_splitter import FixedTrafficSplitter, AdaptiveTrafficSplitter, ABVariant
from .decision_engine import ABDecisionEngine
from .statistics import StatisticalAnalyzer

@dataclass
class TestSession:
    test_id: str
    user_id: str
    variant: str
    start_time: datetime
    end_time: Optional[datetime] = None
    metrics: Dict[str, float] = None
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = {}

from backend.microservices.database.session import SessionLocal
from backend.microservices.database import crud
from backend.microservices.database.models import TestSessionORM, ABTestORM

class SessionManager:
    def __init__(self):
        self.active_sessions: Dict[str, TestSession] = {}
        self.session_history: List[TestSession] = []
    
    def start_session(self, test_id: str, user_id: str, variant: str) -> str:
        session_id = str(uuid.uuid4())
        session = TestSession(
            test_id=test_id,
            user_id=user_id,
            variant=variant,
            start_time=datetime.now()
        )
        self.active_sessions[session_id] = session
        with SessionLocal() as db:
            db_session = TestSessionORM(
                session_id=session_id,
                test_id=test_id,
                user_id=user_id,
                variant=variant,
                start_time=session.start_time,
                metrics={}
            )
            db.add(db_session)
            db.commit()
        return session_id
    
    def end_session(self, session_id: str, metrics: Dict[str, float] = None) -> TestSession:
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.active_sessions[session_id]
        session.end_time = datetime.now()
        
        if metrics:
            session.metrics.update(metrics)
        
        self.session_history.append(session)
        del self.active_sessions[session_id]
        with SessionLocal() as db:
            db_session = db.query(TestSessionORM).filter(TestSessionORM.session_id == session_id).first()
            if db_session:
                db_session.end_time = session.end_time
                db_session.metrics = session.metrics
                db.commit()
        return session

    def get_session_metrics(self, test_id: str) -> Dict[str, List[float]]:
        metrics = {}
        for session in self.session_history:
            if session.test_id == test_id:
                for metric_name, value in session.metrics.items():
                    if metric_name not in metrics:
                        metrics[metric_name] = []
                    metrics[metric_name].append(value)
        return metrics
    
class TestRegistry:
    def __init__(self):
        self.tests: Dict[str, Dict[str, Any]] = {}
        self.test_history: List[Dict[str, Any]] = []
    
    def register_test(self, config: TestConfig, created_by: str, description: str = ""):
        test_info = {
            'config': {
                'test_id': config.test_id,
                'variants': config.variants,
                'primary_metric': config.primary_metric,
                'metric_type': config.metric_type.value if hasattr(config.metric_type, 'value') else str(config.metric_type),
                'sample_size': config.sample_size,
                'confidence_level': config.confidence_level,
                'power': config.power,
                'min_effect_size': config.min_effect_size
            },
            'created_by': created_by,
            'created_at': datetime.now(),
            'description': description,
            'status': 'active',
            'total_users': 0,
            'completion_percentage': 0.0
        }
        self.tests[config.test_id] = test_info
    
    def update_test_stats(self, test_id: str, user_count: int, completion_pct: float):
        if test_id in self.tests:
            self.tests[test_id]['total_users'] = user_count
            self.tests[test_id]['completion_percentage'] = completion_pct
        with SessionLocal() as db:
            db_test = db.query(ABTestORM).filter(ABTestORM.test_id == test_id).first()
            if db_test:
                db_test.total_users = user_count
                db_test.completion_percentage = completion_pct
                db.commit()
    
    def archive_test(self, test_id: str, reason: str = ""):
        if test_id in self.tests:
            test_info = self.tests[test_id]
            test_info['status'] = 'archived'
            test_info['archived_at'] = datetime.now()
            test_info['archive_reason'] = reason
            
            self.test_history.append(test_info)
            del self.tests[test_id]
        with SessionLocal() as db:
            db_test = db.query(ABTestORM).filter(ABTestORM.test_id == test_id).first()
            if db_test:
                db_test.status = 'archived'
                db_test.archive_reason = reason
                db.commit()
    
    def get_active_tests(self) -> List[Dict[str, Any]]:
        with SessionLocal() as db:
            db_tests = db.query(ABTestORM).filter(ABTestORM.status == 'active').all()
            return [{
                'test_id': t.test_id,
                'test_name': t.test_name,
                'description': t.description,
                'status': t.status,
                'variants': t.variants,
                'primary_metric': t.primary_metric,
                'metric_type': t.metric_type,
                'sample_size': t.sample_size,
                'confidence_level': t.confidence_level,
                'power': t.power,
                'min_effect_size': t.min_effect_size,
                'created_by_user_id': t.created_by_user_id,
                'created_at': t.created_at,
                'total_users': t.total_users if hasattr(t, 'total_users') else 0,
                'completion_percentage': t.completion_percentage if hasattr(t, 'completion_percentage') else 0.0
            } for t in db_tests]
    
    def get_test_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        with SessionLocal() as db:
            db_tests = db.query(ABTestORM).filter(ABTestORM.status.in_(['archived', 'completed'])).order_by(ABTestORM.updated_at.desc()).limit(limit).all()
            return [{
                'test_id': t.test_id,
                'test_name': t.test_name,
                'description': t.description,
                'status': t.status,
                'variants': t.variants,
                'primary_metric': t.primary_metric,
                'metric_type': t.metric_type,
                'sample_size': t.sample_size,
                'confidence_level': t.confidence_level,
                'power': t.power,
                'min_effect_size': t.min_effect_size,
                'created_by_user_id': t.created_by_user_id,
                'created_at': t.created_at,
                'updated_at': t.updated_at,
                'archive_reason': t.archive_reason if hasattr(t, 'archive_reason') else None,
                'total_users': t.total_users if hasattr(t, 'total_users') else 0,
                'completion_percentage': t.completion_percentage if hasattr(t, 'completion_percentage') else 0.0
            } for t in db_tests]

    def force_update_test_from_database(self, test_id: str):
        """
        Принудительно обновляет статистику теста из базы данных
        """
        with SessionLocal() as db:
            # Получаем статистику из базы данных
            user_count = db.query(func.count(TestSessionORM.id)).filter(TestSessionORM.test_id == test_id).scalar() or 0

            # Получаем тест из базы данных
            db_test = db.query(ABTestORM).filter(ABTestORM.test_id == test_id).first()

            if db_test:
                # Обновляем статистику в реестре
                completion_pct = db_test.completion_percentage if hasattr(db_test, 'completion_percentage') else 0.0
                self.update_test_stats(test_id, user_count, completion_pct)

                # Если тест завершен, обновляем его статус
                if db_test.status == 'completed' and test_id in self.tests:
                    self.tests[test_id]['status'] = 'completed'

class AdaptiveABTestingPlatform:
    def __init__(self):
        self.test_manager = ABTestManager()
        self.session_manager = SessionManager()
        self.test_registry = TestRegistry()
        self.metric_definitions: Dict[str, MetricType] = {}
        self._splitters: Dict[str, Any] = {}

    def _parse_metric_type(self, value: Any) -> MetricType:
        try:
            if isinstance(value, MetricType):
                return value
            if isinstance(value, str):
                return MetricType(value)
        except Exception:
            pass
        return MetricType.CONTINUOUS

    def _load_test_from_db(self, test_id: str) -> bool:
        with SessionLocal() as db:
            db_test = db.query(ABTestORM).filter(ABTestORM.test_id == test_id).first()  # Убрано ограничение на статус "active"
            if db_test is None:
                return False

            config = TestConfig(
                test_id=db_test.test_id,
                variants=db_test.variants or [],
                primary_metric=db_test.primary_metric,
                metric_type=self._parse_metric_type(db_test.metric_type),
                sample_size=db_test.sample_size,
                confidence_level=db_test.confidence_level,
                power=db_test.power,
                min_effect_size=db_test.min_effect_size,
            )

            # Обновляем или создаем тест в менеджере
            if db_test.test_id not in self.test_manager.active_tests:
                self.test_manager.create_test(config)
            else:
                # Если тест уже существует, обновляем его конфигурацию
                self.test_manager.test_configs[config.test_id] = config

            self.test_registry.tests[db_test.test_id] = {
                "config": {
                    "test_id": config.test_id,
                    "variants": config.variants,
                    "primary_metric": config.primary_metric,
                    "metric_type": config.metric_type.value,
                    "sample_size": config.sample_size,
                    "confidence_level": config.confidence_level,
                    "power": config.power,
                    "min_effect_size": config.min_effect_size,
                },
                "created_by": str(db_test.created_by_user_id) if db_test.created_by_user_id is not None else "system",
                "created_at": db_test.created_at or datetime.now(),
                "description": db_test.description or "",
                "status": db_test.status,
                "total_users": db_test.total_users if hasattr(db_test, "total_users") else 0,
                "completion_percentage": db_test.completion_percentage if hasattr(db_test, "completion_percentage") else 0.0,
            }

            self.metric_definitions[config.primary_metric] = config.metric_type
            self._ensure_splitter_loaded(db_test)
            return True

    def _ensure_test_loaded(self, test_id: str) -> bool:
        if test_id in self.test_manager.active_tests:
            return True
        return self._load_test_from_db(test_id)

    def _ensure_splitter_loaded(self, db_test: ABTestORM) -> None:
        if db_test.test_id in self._splitters:
            return

        variants = [ABVariant(name=v, weight=1.0) for v in (db_test.variants or [])]
        if not variants:
            return

        if (db_test.traffic_split_type or "fixed") == "adaptive":
            self._splitters[db_test.test_id] = AdaptiveTrafficSplitter(variants)
        else:
            seed = int(db_test.traffic_split_seed or 42)
            self._splitters[db_test.test_id] = FixedTrafficSplitter(variants, seed=seed)

    def _resolve_test_runtime_meta(self, test_id: str) -> Dict[str, Any]:
        with SessionLocal() as db:
            db_test = db.query(ABTestORM).filter(ABTestORM.test_id == test_id).first()
            if not db_test:
                raise ValueError(f"Test {test_id} not found")

            self._ensure_splitter_loaded(db_test)
            analysis_mode = str(db_test.analysis_mode or "fixed_experiment")
            traffic_split_type = str(db_test.traffic_split_type or "fixed")
            splitter = self._splitters.get(test_id)

            return {
                "db_test": db_test,
                "analysis_mode": analysis_mode,
                "traffic_split_type": traffic_split_type,
                "splitter": splitter,
            }
    
    def _normalize_reward_for_splitter(self, test_id: str, raw_value: float) -> float:
        """
        Нормализует reward в [0, 1] для AdaptiveTrafficSplitter.
        - binary: ожидаем 0/1
        - continuous/ratio: мягкая логистическая нормализация без жёстких клипов по 1000
        """
        config = self.test_manager.test_configs.get(test_id)
        metric_type = config.metric_type if config else MetricType.CONTINUOUS

        val = float(raw_value)
        if metric_type == MetricType.BINARY:
            return float(max(0.0, min(1.0, val)))

        # robust-нормализация для непрерывных метрик
        scale = max(1.0, abs(val) + 1.0)
        normalized = 1.0 / (1.0 + np.exp(-val / scale))
        return float(max(0.0, min(1.0, normalized)))

    def refresh_test_from_database(self, test_id: str):
        """
        Обновляет данные теста из базы данных
        """
        return self._load_test_from_db(test_id)
    
    def force_update_test_statistics(self, test_id: str):
        """
        Принудительно обновляет статистику теста из базы данных.
        Сохраняет прежнее поведение: завершённые тесты выгружаются из in-memory active_tests.
        """
        self.test_registry.force_update_test_from_database(test_id)

        with SessionLocal() as db:
            db_test = db.query(ABTestORM).filter(ABTestORM.test_id == test_id).first()
            if db_test and db_test.status == 'completed' and test_id in self.test_manager.active_tests:
                del self.test_manager.active_tests[test_id]
                del self.test_manager.test_configs[test_id]

    def create_ab_test(self,
                    test_id: str,
                    variants: List[str],
                    primary_metric: str,
                    metric_type: MetricType,
                    created_by: str,
                    description: str = "",
                    sample_size: Optional[int] = None,
                    confidence_level: float = 0.95,
                    power: float = 0.8,
                    min_effect_size: float = 0.1) -> str:
        
        
        config = TestConfig(
            test_id=test_id,
            variants=variants,
            primary_metric=primary_metric,
            metric_type=metric_type,
            sample_size=sample_size,
            confidence_level=confidence_level,
            power=power,
            min_effect_size=min_effect_size
        )
        
        self.test_manager.create_test(config)
        self.test_registry.register_test(config, created_by, description)
        self.metric_definitions[primary_metric] = metric_type
        
        return test_id
    
    def assign_user_to_test(self, test_id: str, user_id: str,
                          user_context: Optional[Dict] = None) -> Dict[str, Any]:
        if not self._ensure_test_loaded(test_id):
            raise ValueError(f"Test {test_id} not found")

        runtime = self._resolve_test_runtime_meta(test_id)
        splitter = runtime["splitter"]
        if splitter is None:
            raise ValueError(f"Splitter for test {test_id} not initialized")

        with SessionLocal() as db:
            persisted_assignment = crud.get_user_assignment(db, test_id=test_id, user_id=user_id)

            if persisted_assignment is not None:
                assignment_meta = {
                    "variant": str(persisted_assignment.variant),
                    "hash_bucket": persisted_assignment.hash_bucket,
                    "hash_space_size": persisted_assignment.hash_space_size,
                    "seed": persisted_assignment.seed,
                    "splitter_type": persisted_assignment.splitter_type,
                    "sticky": True,
                }
            else:
                if hasattr(splitter, "assign_variant_with_metadata"):
                    if runtime["traffic_split_type"] == "fixed":
                        assignment_meta = splitter.assign_variant_with_metadata(user_id=user_id, test_id=test_id)
                    else:
                        assignment_meta = splitter.assign_variant_with_metadata(user_id=user_id)
                else:
                    variant_fallback = splitter.assign_variant(user_id, test_id)
                    assignment_meta = {
                        "variant": variant_fallback,
                        "hash_bucket": None,
                        "hash_space_size": None,
                        "seed": None,
                        "splitter_type": runtime["traffic_split_type"],
                    }

                crud.upsert_user_assignment(
                    db,
                    test_id=test_id,
                    user_id=user_id,
                    variant=str(assignment_meta["variant"]),
                    splitter_type=str(assignment_meta.get("splitter_type") or runtime["traffic_split_type"]),
                    hash_bucket=assignment_meta.get("hash_bucket"),
                    hash_space_size=assignment_meta.get("hash_space_size"),
                    seed=assignment_meta.get("seed"),
                    assignment_metadata={"user_context": user_context or {}},
                    do_commit=True,
                )
                assignment_meta["sticky"] = False

        variant = str(assignment_meta["variant"])
        session_id = self.session_manager.start_session(test_id, user_id, variant)

        with SessionLocal() as db:
            crud.create_assignment_audit(
                db,
                test_id=test_id,
                session_id=session_id,
                user_id=user_id,
                variant=variant,
                splitter_type=str(assignment_meta.get("splitter_type") or runtime["traffic_split_type"]),
                analysis_mode=runtime["analysis_mode"],
                traffic_split_type=runtime["traffic_split_type"],
                hash_bucket=assignment_meta.get("hash_bucket"),
                hash_space_size=assignment_meta.get("hash_space_size"),
                seed=assignment_meta.get("seed"),
                assignment_metadata={
                    "user_context": user_context or {},
                    "sticky_reused": bool(assignment_meta.get("sticky", False)),
                },
            )

        return {
            'session_id': session_id,
            'variant': variant,
            'test_id': test_id,
            'analysis_mode': runtime["analysis_mode"],
            'traffic_split_type': runtime["traffic_split_type"],
            'assignment_audit': {
                'splitter_type': assignment_meta.get("splitter_type"),
                'hash_bucket': assignment_meta.get("hash_bucket"),
                'hash_space_size': assignment_meta.get("hash_space_size"),
                'seed': assignment_meta.get("seed"),
                'sticky_reused': bool(assignment_meta.get("sticky", False)),
            },
        }
    
    def record_user_metric(self, session_id: str, metric_name: str, value: float, event_id: Optional[str] = None) -> Dict[str, Any]:
        session = self.session_manager.active_sessions.get(session_id)
        db_session: Optional[TestSessionORM] = None

        with SessionLocal() as db:
            db_session = db.query(TestSessionORM).filter(TestSessionORM.session_id == session_id).first()
            if not db_session:
                raise ValueError(f"Session {session_id} not found")

            test_id = str(db_session.test_id)
            effective_event_id = event_id or f"{session_id}:{metric_name}:{uuid.uuid4().hex[:12]}"

            if not self._ensure_test_loaded(test_id):
                raise ValueError(f"Test {test_id} not found")

            config = self.test_manager.test_configs.get(test_id)
            primary_metric = self._get_primary_metric(test_id)

            if config:
                metric_type = config.metric_type

                if metric_name == primary_metric:
                    if metric_type == MetricType.BINARY:
                        if float(value) not in (0.0, 1.0):
                            raise ValueError("Binary-метрика должна быть 0 или 1")
                    elif metric_type == MetricType.CONTINUOUS:
                        if not np.isfinite(float(value)):
                            raise ValueError("Continuous-метрика должна быть конечным числом")
                    elif metric_type == MetricType.RATIO:
                        raise ValueError(
                            "Для ratio-метрики нельзя писать primary_metric; используйте *_numerator и *_denominator"
                        )

                if metric_type == MetricType.RATIO:
                    ratio_num_key = f"{primary_metric}_numerator"
                    ratio_den_key = f"{primary_metric}_denominator"
                    if metric_name == ratio_num_key:
                        if not np.isfinite(float(value)):
                            raise ValueError("Ratio numerator должен быть конечным числом")
                    if metric_name == ratio_den_key:
                        if not np.isfinite(float(value)) or float(value) <= 0:
                            raise ValueError("Ratio denominator должен быть положительным конечным числом")

            _, created = crud.create_metric_event_if_absent(
                db,
                event_id=effective_event_id,
                session_id=session_id,
                test_id=test_id,
                metric_name=metric_name,
                value=float(value),
                do_commit=True,
            )

            if not created:
                return {
                    "deduplicated": True,
                    "event_id": effective_event_id,
                    "session_id": session_id,
                    "metric_name": metric_name,
                }

            session_metrics = dict(db_session.metrics or {})
            session_metrics[metric_name] = float(value)
            db_session.metrics = session_metrics
            db.commit()

            if session is not None:
                session.metrics[metric_name] = float(value)

            if metric_name == primary_metric:
                self.test_manager.record_metric(test_id, str(db_session.variant), float(value))

                splitter = self._splitters.get(test_id)
                if isinstance(splitter, AdaptiveTrafficSplitter):
                    reward = self._normalize_reward_for_splitter(test_id, float(value))
                    splitter.update(str(db_session.variant), reward)

        return {
            "deduplicated": False,
            "event_id": effective_event_id,
            "session_id": session_id,
            "metric_name": metric_name,
        }
    def complete_user_session(self, session_id: str, final_metrics: Dict[str, float] = None):
        session = self.session_manager.end_session(session_id, final_metrics)
        self._update_test_progress(session.test_id)

    
    def get_test_results(self, test_id: str) -> Dict[str, Any]:
        with SessionLocal() as db:
            db_test = db.query(ABTestORM).filter(ABTestORM.test_id == test_id).first()
            if not db_test:
                raise ValueError(f"Test {test_id} not found")

            config = TestConfig(
                test_id=db_test.test_id,
                variants=db_test.variants or [],
                primary_metric=db_test.primary_metric,
                metric_type=self._parse_metric_type(db_test.metric_type),
                sample_size=db_test.sample_size,
                confidence_level=db_test.confidence_level,
                power=db_test.power,
                min_effect_size=db_test.min_effect_size,
            )

            temp_test = AdaptiveABTest(config)
            sessions = db.query(TestSessionORM).filter(TestSessionORM.test_id == test_id).all()

            session_metrics: Dict[str, List[float]] = {}
            ratio_components: Dict[str, Dict[str, List[float]]] = {
                v: {"numerators": [], "denominators": []} for v in config.variants
            }

            ratio_num_key = f"{config.primary_metric}_numerator"
            ratio_den_key = f"{config.primary_metric}_denominator"

            for session in sessions:
                metrics = dict(session.metrics or {})
                for metric_name, raw_value in metrics.items():
                    try:
                        val = float(raw_value)
                    except Exception:
                        continue
                    session_metrics.setdefault(metric_name, []).append(val)

                if session.variant not in config.variants:
                    continue

                variant_name = str(session.variant)

                if config.metric_type == MetricType.RATIO:
                    num_raw = metrics.get(ratio_num_key)
                    den_raw = metrics.get(ratio_den_key)

                    ratio_value: Optional[float] = None

                    try:
                        if num_raw is not None and den_raw is not None:
                            num = float(num_raw)
                            den = float(den_raw)
                            if np.isfinite(num) and np.isfinite(den) and den > 0:
                                ratio_components[variant_name]["numerators"].append(num)
                                ratio_components[variant_name]["denominators"].append(den)
                                ratio_value = num / den
                    except Exception:
                        ratio_value = None

                    if ratio_value is not None:
                        try:
                            temp_test.record_observation(variant_name, float(ratio_value))
                        except Exception:
                            pass
                else:
                    if config.primary_metric in metrics:
                        try:
                            temp_test.record_observation(variant_name, float(metrics[config.primary_metric]))
                        except Exception:
                            pass

            results = temp_test.get_results()

            if config.metric_type == MetricType.RATIO and config.variants:
                alpha = 1.0 - float(config.confidence_level)
                if not (0.0 < alpha < 1.0):
                    alpha = 0.05
                analyzer = StatisticalAnalyzer(alpha=alpha)
                control_variant = config.variants[0]
                control_num = np.asarray(ratio_components.get(control_variant, {}).get("numerators", []), dtype=float)
                control_den = np.asarray(ratio_components.get(control_variant, {}).get("denominators", []), dtype=float)

                p_values = {}
                for variant in config.variants[1:]:
                    treat_num = np.asarray(ratio_components.get(variant, {}).get("numerators", []), dtype=float)
                    treat_den = np.asarray(ratio_components.get(variant, {}).get("denominators", []), dtype=float)

                    if len(control_num) < 10 or len(treat_num) < 10:
                        p_values[variant] = 1.0
                        continue

                    try:
                        ratio_result = analyzer.analyze_ratio_metric(
                            control_numerators=control_num,
                            control_denominators=control_den,
                            treatment_numerators=treat_num,
                            treatment_denominators=treat_den,
                            num_comparisons=1,
                        )
                        p_values[variant] = float(ratio_result.p_value)
                    except Exception:
                        p_values[variant] = 1.0
            else:
                p_values = temp_test.calculate_statistical_significance()

        corrected_p = ABDecisionEngine.holm_bonferroni_correction(p_values)

        control_variant = list(results.keys())[0] if results else None
        inferred_means = ABDecisionEngine.infer_variant_means(results)
        winner_pre = control_variant
        winner_uplift_pre = -1e18
        if control_variant and control_variant in inferred_means:
            control_mean = inferred_means[control_variant]
            for variant, mean_val in inferred_means.items():
                if variant == control_variant:
                    continue
                uplift = ((mean_val - control_mean) / control_mean * 100.0) if abs(control_mean) > 1e-12 else 0.0
                if corrected_p.get(variant, 1.0) < 0.05 and uplift > winner_uplift_pre:
                    winner_uplift_pre = uplift
                    winner_pre = variant

        guardrails_status = ABDecisionEngine.evaluate_guardrails(
            guardrails_config=(db_test.guardrails_config if db_test else None),
            variant_means=inferred_means,
            control_variant=control_variant or "",
            winner_variant=winner_pre,
        )

        analysis_validity = ABDecisionEngine.resolve_analysis_validity(
            analysis_mode=(db_test.analysis_mode if db_test else "fixed_experiment"),
            traffic_split_type=(db_test.traffic_split_type if db_test else "fixed"),
            srm_detected=(True if db_test and db_test.srm_check_passed == 0 else False if db_test and db_test.srm_check_passed == 1 else None),
            guardrails_failed=not bool(guardrails_status.get("passed", True)),
        )

        summary = ABDecisionEngine.build_decision_summary(
            results=results,
            p_values_raw=p_values,
            corrected_p_values=corrected_p,
            alpha=0.05,
            analysis_validity=analysis_validity,
            guardrails_status=guardrails_status,
        )

        pm_summary = self._build_pm_summary(results, corrected_p, summary)

        if db_test:
            with SessionLocal() as dbw:
                current = dbw.query(ABTestORM).filter(ABTestORM.test_id == test_id).first()
                if current:
                    current.analysis_validity = analysis_validity
                    current.guardrails_status = guardrails_status
                    dbw.commit()

        quality_gate = self._build_quality_gate(
            db_test=db_test,
            results=results,
            corrected_p=corrected_p,
            guardrails_status=guardrails_status,
            alpha=0.05,
        )

        min_power: Optional[float] = None
        for check in quality_gate.get("checks", []):
            if check.get("id") == "power":
                try:
                    min_power = float(check.get("actual"))
                except Exception:
                    min_power = None
                break

        srm_passed: Optional[bool] = None
        if db_test and db_test.srm_check_passed is not None:
            srm_passed = bool(db_test.srm_check_passed == 1)

        decision_policy = ABDecisionEngine.evaluate_decision_policy(
            analysis_validity=analysis_validity,
            srm_passed=srm_passed,
            guardrails_passed=bool(guardrails_status.get("passed", True)),
            corrected_p_values=corrected_p,
            power=min_power,
            alpha=0.05,
        )

        return {
            'test_id': test_id,
            'results': {k: asdict(v) for k, v in results.items()},
            'statistical_significance': p_values,
            'statistical_significance_corrected': corrected_p,
            'session_metrics': session_metrics,
            'summary': summary,
            'pm_summary': pm_summary,
            'analysis_mode': (db_test.analysis_mode if db_test else "fixed_experiment"),
            'traffic_split_type': (db_test.traffic_split_type if db_test else "fixed"),
            'analysis_validity': analysis_validity,
            'guardrails': guardrails_status,
            'quality_gate': quality_gate,
            'decision_policy': asdict(decision_policy),
        }
    
    def stop_test(self, test_id: str, reason: str = "Manual stop") -> Dict[str, Any]:
        with SessionLocal() as db:
            db_test = db.query(ABTestORM).filter(ABTestORM.test_id == test_id).first()
            if not db_test:
                raise ValueError(f"Test {test_id} not found")

            db_test.status = 'completed'
            db_test.simulation_status = None
            db.commit()

        final_results = self.get_test_results(test_id)

        if test_id in self.test_manager.active_tests:
            del self.test_manager.active_tests[test_id]
        if test_id in self.test_manager.test_configs:
            del self.test_manager.test_configs[test_id]

        if test_id in self.test_registry.tests:
            self.test_registry.tests[test_id]['status'] = 'completed'

        return {
            'test_id': test_id,
            'final_results': final_results,
            'summary': final_results.get('summary', {}),
            'stopped_at': datetime.now(),
            'reason': reason,
            'status': 'completed',
        }
    
    def get_platform_stats(self) -> Dict[str, Any]:
        active_tests = self.test_registry.get_active_tests()
        total_users = sum(test['total_users'] for test in active_tests)
        
        completion_rates = [test['completion_percentage'] for test in active_tests]
        avg_completion = np.mean(completion_rates) if completion_rates else 0
        
        return {
            'active_tests': len(active_tests),
            'total_users': total_users,
            'average_completion': avg_completion,
            'tests_today': len([test for test in active_tests 
                              if test['created_at'].date() == datetime.now().date()])
        }
    
    def _get_primary_metric(self, test_id: str) -> str:
        for test_info in self.test_registry.tests.values():
            if test_info['config']['test_id'] == test_id:
                return test_info['config']['primary_metric']

        with SessionLocal() as db:
            db_test = db.query(ABTestORM).filter(ABTestORM.test_id == test_id).first()
            if db_test:
                return str(db_test.primary_metric)

        return ""
    def _update_test_progress(self, test_id: str):
        from sqlalchemy import func
        with SessionLocal() as db:
            user_count = db.query(func.count(TestSessionORM.id)).filter(TestSessionORM.test_id == test_id).scalar() or 0
        
        config = self.test_manager.test_configs.get(test_id)
        if config and config.sample_size:
            completion_pct = min(100.0, (user_count / config.sample_size) * 100)
        else:
            completion_pct = 0.0
        
        self.test_registry.update_test_stats(test_id, user_count, completion_pct)
    
    def _generate_summary(self, results: Dict[str, TestResult], p_values: Dict[str, float]) -> Dict[str, Any]:
        if not results:
            return {}
        
        control_variant = list(results.keys())[0]
        control_result = results[control_variant]
        
        best_variant = control_variant
        best_improvement = 0.0
        significant_variants = []
        
        for variant, p_value in p_values.items():
            if variant in results:
                variant_result = results[variant]
                improvement = ((variant_result.mean - control_result.mean) /
                             control_result.mean * 100) if control_result.mean != 0 else 0
                
                if p_value < 0.05:
                    significant_variants.append(variant)
                    if improvement > best_improvement:
                        best_variant = variant
                        best_improvement = improvement
        
        if not significant_variants:
            best_variant = control_variant
            best_improvement = 0.0
        
        confidence_level = "high"
        if any(p > 0.05 for p in p_values.values()):
            confidence_level = "medium"
        if any(p > 0.1 for p in p_values.values()):
            confidence_level = "low"
        
        return {
            'best_variant': best_variant,
            'improvement_percentage': best_improvement,
            'recommended_action': f"Switch to {best_variant}" if best_variant != control_variant else "Keep control",
            'confidence_level': confidence_level,
            'significant_variants': significant_variants,
            'control_variant': control_variant,
            'p_values': p_values
        }

    def _build_quality_gate(
        self,
        *,
        db_test: Optional[ABTestORM],
        results: Dict[str, TestResult],
        corrected_p: Dict[str, float],
        guardrails_status: Dict[str, Any],
        alpha: float,
    ) -> Dict[str, Any]:
        variants = list(results.keys())
        control = variants[0] if variants else None
        planned_sample_size = int(db_test.sample_size) if db_test and db_test.sample_size else None
        min_required_per_variant = max(30, int((planned_sample_size / max(1, len(variants))) * 0.9)) if planned_sample_size else 30

        sufficient_n = all(int(results[v].sample_size) >= min_required_per_variant for v in variants)
        any_significant = any(float(p) < alpha for p in corrected_p.values())

        srm_known = bool(db_test and db_test.srm_check_passed is not None)
        srm_pass = bool(db_test and db_test.srm_check_passed == 1)
        guardrails_enabled = bool(guardrails_status.get("enabled", False))
        guardrails_pass = bool(guardrails_status.get("passed", True))

        power_threshold = 0.8
        min_power = 0.0
        power_pass = False
        if control and control in results:
            control_stat = results[control]
            control_std = max(1e-6, float(control_stat.std))
            powers: List[float] = []
            analyzer = StatisticalAnalyzer(alpha=alpha)
            for variant, stat in results.items():
                if variant == control:
                    continue
                effect = float(stat.mean) - float(control_stat.mean)
                n = max(1, min(int(control_stat.sample_size), int(stat.sample_size)))
                pwr = analyzer.calculate_power(
                    observed_effect=float(effect),
                    sample_size_per_variant=int(n),
                    baseline_std=float(control_std),
                    alpha=float(alpha),
                )
                powers.append(max(0.0, min(1.0, float(pwr))))
            if powers:
                min_power = float(min(powers))
                power_pass = min_power >= power_threshold

        checks = [
            {
                "id": "srm_pass",
                "title": "SRM check",
                "passed": srm_pass,
                "actual": (None if not db_test else db_test.srm_p_value),
                "threshold": "srm_check_passed = true",
                "known": srm_known,
            },
            {
                "id": "power",
                "title": "Power >= 0.8",
                "passed": power_pass,
                "actual": min_power,
                "threshold": power_threshold,
                "known": True,
            },
            {
                "id": "corrected_p",
                "title": f"Corrected p-value < {alpha}",
                "passed": any_significant,
                "actual": corrected_p,
                "threshold": alpha,
                "known": True,
            },
            {
                "id": "guardrails",
                "title": "Guardrails pass",
                "passed": guardrails_pass,
                "actual": guardrails_status,
                "threshold": True,
                "known": guardrails_enabled,
            },
            {
                "id": "sufficient_n",
                "title": "Sufficient sample size",
                "passed": sufficient_n,
                "actual": {v: int(results[v].sample_size) for v in variants},
                "threshold": min_required_per_variant,
                "known": True,
            },
        ]

        passed_count = sum(1 for c in checks if c["passed"])
        total = len(checks)
        critical_fail = (srm_known and not srm_pass) or (guardrails_enabled and not guardrails_pass)

        if critical_fail:
            status = "red"
        elif passed_count == total:
            status = "green"
        else:
            status = "yellow"

        return {
            "status": status,
            "passed": passed_count == total and not critical_fail,
            "passed_checks": passed_count,
            "total_checks": total,
            "checks": checks,
        }

    def _build_pm_summary(self, results: Dict[str, TestResult], p_values: Dict[str, float], summary: Dict[str, Any]) -> Dict[str, Any]:
        control_variant = summary.get('control_variant')
        if not control_variant or control_variant not in results:
            return {
                'headline': 'Недостаточно данных для интерпретации',
                'decision': 'Продолжить тест',
                'winner': None,
                'confidence': 'low',
                'insights': [],
                'variant_cards': [],
                'next_steps': ['Соберите больше наблюдений по всем вариантам.'],
            }

        control = results[control_variant]
        alpha = 0.05
        winner = summary.get('best_variant')
        confidence = summary.get('confidence_level', 'low')

        variant_cards: List[Dict[str, Any]] = []
        insights: List[str] = []

        for variant, stat in results.items():
            if variant == control_variant:
                continue

            uplift_pct = ((stat.mean - control.mean) / control.mean * 100) if control.mean != 0 else 0.0
            p_value = p_values.get(variant)
            significant = p_value is not None and p_value < alpha

            direction = 'рост' if uplift_pct >= 0 else 'падение'
            p_value_str = f"{p_value:.4f}" if p_value is not None else "n/a"
            insights.append(
                f"Вариант {variant}: {direction} {abs(uplift_pct):.2f}% к контролю {control_variant}, p-value={p_value_str}"
            )

            variant_cards.append({
                'variant': variant,
                'sample_size': stat.sample_size,
                'mean': stat.mean,
                'uplift_percent_vs_control': uplift_pct,
                'p_value': p_value,
                'significant': significant,
            })

        if winner and winner != control_variant and winner in p_values and p_values[winner] < alpha:
            decision = f"Рекомендуется внедрить вариант {winner}"
            headline = f"Победитель найден: вариант {winner}"
        else:
            decision = "Пока нет статистически значимого победителя"
            headline = "Тест пока без явного победителя"

        next_steps = [
            "Продолжить эксперимент до увеличения выборки по отстающим вариантам.",
            "Проверить сегменты пользователей (новые/возвращающиеся, устройства, источники).",
            "Принять решение о внедрении только при устойчивой значимости (p-value < 0.05).",
        ]

        return {
            'headline': headline,
            'decision': decision,
            'winner': winner if decision.startswith('Рекомендуется') else None,
            'confidence': confidence,
            'control_variant': control_variant,
            'insights': insights,
            'variant_cards': variant_cards,
            'next_steps': next_steps,
        }