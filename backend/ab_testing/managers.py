# backend/ab_testing/managers.py

import uuid
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import json
from .core import ABTestManager, TestConfig, MetricType, TestResult

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

from backend.database.session import SessionLocal
from backend.database import crud
from backend.database.models import TestSessionORM, ABTestORM

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

class AdaptiveABTestingPlatform:
    def __init__(self):
        self.test_manager = ABTestManager()
        self.session_manager = SessionManager()
        self.test_registry = TestRegistry()
        self.metric_definitions: Dict[str, MetricType] = {}

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
            db_test = db.query(ABTestORM).filter(ABTestORM.test_id == test_id, ABTestORM.status == "active").first()
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

            if db_test.test_id not in self.test_manager.active_tests:
                self.test_manager.create_test(config)

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
            return True

    def _ensure_test_loaded(self, test_id: str) -> bool:
        if test_id in self.test_manager.active_tests:
            return True
        return self._load_test_from_db(test_id)

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

        variant = self.test_manager.assign_variant(test_id, user_id, user_context)
        session_id = self.session_manager.start_session(test_id, user_id, variant)
        
        return {
            'session_id': session_id,
            'variant': variant,
            'test_id': test_id
        }
    
    def record_user_metric(self, session_id: str, metric_name: str, value: float):
        session = self.session_manager.active_sessions.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        session.metrics[metric_name] = value
        
        if metric_name == self._get_primary_metric(session.test_id):
            self.test_manager.record_metric(session.test_id, session.variant, value)
    def complete_user_session(self, session_id: str, final_metrics: Dict[str, float] = None):
        session = self.session_manager.end_session(session_id, final_metrics)
        self._update_test_progress(session.test_id)

    
    def get_test_results(self, test_id: str) -> Dict[str, Any]:
        if not self._ensure_test_loaded(test_id):
            raise ValueError(f"Test {test_id} not found")

        results, p_values = self.test_manager.get_test_results(test_id)
        session_metrics = self.session_manager.get_session_metrics(test_id)
        summary = self._generate_summary(results, p_values)
        pm_summary = self._build_pm_summary(results, p_values, summary)
        
        return {
            'test_id': test_id,
            'results': {k: asdict(v) for k, v in results.items()},
            'statistical_significance': p_values,
            'session_metrics': session_metrics,
            'summary': summary,
            'pm_summary': pm_summary,
        }
    
    def stop_test(self, test_id: str, reason: str = "Manual stop") -> Dict[str, Any]:
        if not self._ensure_test_loaded(test_id):
            raise ValueError(f"Test {test_id} not found")

        test_results = self.test_manager.get_test_results(test_id)
        self.test_registry.archive_test(test_id, reason)
        final_summary = self.test_manager.stop_test(test_id)
        
        return {
            'test_id': test_id,
            'final_results': test_results,
            'summary': final_summary,
            'stopped_at': datetime.now(),
            'reason': reason
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