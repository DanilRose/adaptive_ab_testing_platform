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
        return session_id
    
    def end_session(self, session_id: str, metrics: Dict[str, float] = None):
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.active_sessions[session_id]
        session.end_time = datetime.now()
        
        if metrics:
            session.metrics.update(metrics)
        
        self.session_history.append(session)
        del self.active_sessions[session_id]
    
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
            'config': asdict(config),
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
    
    def archive_test(self, test_id: str, reason: str = ""):
        if test_id in self.tests:
            test_info = self.tests[test_id]
            test_info['status'] = 'archived'
            test_info['archived_at'] = datetime.now()
            test_info['archive_reason'] = reason
            
            self.test_history.append(test_info)
            del self.tests[test_id]
    
    def get_active_tests(self) -> List[Dict[str, Any]]:
        return list(self.tests.values())
    
    def get_test_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        return sorted(self.test_history, key=lambda x: x['archived_at'], reverse=True)[:limit]

class AdaptiveABTestingPlatform:
    def __init__(self):
        self.test_manager = ABTestManager()
        self.session_manager = SessionManager()
        self.test_registry = TestRegistry()
        self.metric_definitions: Dict[str, MetricType] = {}
    
    def create_ab_test(self, 
                      test_id: str,
                      variants: List[str],
                      primary_metric: str,
                      metric_type: MetricType,
                      created_by: str,
                      description: str = "",
                      **kwargs) -> str:
        
        config = TestConfig(
            test_id=test_id,
            variants=variants,
            primary_metric=primary_metric,
            metric_type=metric_type,
            **kwargs
        )
        
        self.test_manager.create_test(config)
        self.test_registry.register_test(config, created_by, description)
        self.metric_definitions[primary_metric] = metric_type
        
        return test_id
    
    def assign_user_to_test(self, test_id: str, user_id: str, 
                          user_context: Optional[Dict] = None) -> Dict[str, Any]:
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
        self.session_manager.end_session(session_id, final_metrics)
        self._update_test_progress(session_id)
    
    def get_test_results(self, test_id: str) -> Dict[str, Any]:
        results, p_values = self.test_manager.get_test_results(test_id)
        session_metrics = self.session_manager.get_session_metrics(test_id)
        
        return {
            'test_id': test_id,
            'results': {k: asdict(v) for k, v in results.items()},
            'statistical_significance': p_values,
            'session_metrics': session_metrics,
            'summary': self._generate_summary(results, p_values)
        }
    
    def stop_test(self, test_id: str, reason: str = "Manual stop") -> Dict[str, Any]:
        test_results = self.test_manager.stop_test(test_id)
        self.test_registry.archive_test(test_id, reason)
        
        return {
            'test_id': test_id,
            'final_results': test_results,
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
    
    def _update_test_progress(self, session_id: str):
        session = next((s for s in self.session_manager.session_history 
                       if s.test_id == session_id), None)
        if session:
            test_id = session.test_id
            user_count = len([s for s in self.session_manager.session_history 
                            if s.test_id == test_id])
            
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
        
        for variant, p_value in p_values.items():
            if variant in results:
                variant_result = results[variant]
                improvement = ((variant_result.mean - control_result.mean) / 
                             control_result.mean * 100) if control_result.mean != 0 else 0
                
                if improvement > best_improvement and p_value < 0.05:
                    best_variant = variant
                    best_improvement = improvement
        
        return {
            'best_variant': best_variant,
            'improvement_percentage': best_improvement,
            'recommended_action': f"Switch to {best_variant}" if best_variant != control_variant else "Keep control",
            'confidence_level': "high" if all(p < 0.01 for p in p_values.values()) else "medium"
        }