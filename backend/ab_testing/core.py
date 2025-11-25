# backend/ab_testing/core.py

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import scipy.stats as stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

class MetricType(Enum):
    BINARY = "binary"
    CONTINUOUS = "continuous"
    RATIO = "ratio"

@dataclass
class TestConfig:
    test_id: str
    variants: List[str]
    primary_metric: str
    metric_type: MetricType
    sample_size: Optional[int] = None
    confidence_level: float = 0.95
    power: float = 0.8
    min_effect_size: float = 0.1

@dataclass
class TestResult:
    variant: str
    sample_size: int
    mean: float
    std: float
    confidence_interval: Tuple[float, float]
    p_value: Optional[float] = None

class BanditAlgorithm(ABC):
    @abstractmethod
    def select_variant(self, test_id: str, context: Optional[Dict] = None) -> str:
        pass
    
    @abstractmethod
    def update(self, test_id: str, variant: str, reward: float):
        pass

class ThompsonSamplingBandit(BanditAlgorithm):
    def __init__(self):
        self.successes: Dict[str, Dict[str, float]] = {}
        self.failures: Dict[str, Dict[str, float]] = {}
    
    def select_variant(self, test_id: str, context: Optional[Dict] = None) -> str:
        if test_id not in self.successes:
            raise ValueError(f"Test {test_id} not initialized")
        
        samples = {}
        for variant in self.successes[test_id].keys():
            alpha = self.successes[test_id][variant]
            beta = self.failures[test_id][variant]
            samples[variant] = np.random.beta(alpha, beta)
        
        return max(samples, key=samples.get)
    
    def update(self, test_id: str, variant: str, reward: float):
        if test_id not in self.successes:
            self._initialize_test(test_id, [variant])
        
        if reward > 0:
            self.successes[test_id][variant] += reward
        else:
            self.failures[test_id][variant] += abs(reward)
    
    def _initialize_test(self, test_id: str, variants: List[str]):
        self.successes[test_id] = {v: 1.0 for v in variants}
        self.failures[test_id] = {v: 1.0 for v in variants}

class StatisticalTest:
    @staticmethod
    def calculate_sample_size(metric_type: MetricType, baseline: float, effect_size: float, 
                            alpha: float = 0.05, power: float = 0.8) -> int:
        if metric_type == MetricType.BINARY:
            return StatisticalTest._binary_sample_size(baseline, effect_size, alpha, power)
        elif metric_type == MetricType.CONTINUOUS:
            return StatisticalTest._continuous_sample_size(effect_size, alpha, power)
        else:
            raise NotImplementedError(f"Sample size calculation for {metric_type} not implemented")
    
    @staticmethod
    def _binary_sample_size(baseline: float, effect_size: float, alpha: float, power: float) -> int:
        proportion = baseline + effect_size
        pooled_prop = (baseline + proportion) / 2
        
        z_alpha = stats.norm.ppf(1 - alpha/2)
        z_beta = stats.norm.ppf(power)
        
        numerator = (z_alpha * np.sqrt(2 * pooled_prop * (1 - pooled_prop)) + 
                    z_beta * np.sqrt(baseline * (1 - baseline) + proportion * (1 - proportion)))**2
        denominator = effect_size**2
        
        return int(np.ceil(numerator / denominator))
    
    @staticmethod
    def _continuous_sample_size(effect_size: float, alpha: float, power: float) -> int:
        z_alpha = stats.norm.ppf(1 - alpha/2)
        z_beta = stats.norm.ppf(power)
        
        return int(np.ceil(2 * ((z_alpha + z_beta) / effect_size)**2))
    
    @staticmethod
    def t_test(control_data: np.ndarray, variant_data: np.ndarray) -> Tuple[float, float]:
        t_stat, p_value = stats.ttest_ind(variant_data, control_data, equal_var=False)
        return float(t_stat), float(p_value)
    
    @staticmethod
    def chi_square_test(control_success: int, control_total: int, 
                       variant_success: int, variant_total: int) -> Tuple[float, float]:
        contingency_table = np.array([
            [control_success, control_total - control_success],
            [variant_success, variant_total - variant_success]
        ])
        chi2_stat, p_value, _, _ = stats.chi2_contingency(contingency_table)
        return float(chi2_stat), float(p_value)

class AdaptiveABTest:
    def __init__(self, config: TestConfig):
        self.config = config
        self.bandit = ThompsonSamplingBandit()
        self.data: Dict[str, List[float]] = {v: [] for v in config.variants}
        self._initialize_bandit()
    
    def _initialize_bandit(self):
        for variant in self.config.variants:
            self.bandit.update(self.config.test_id, variant, 0.0)
    
    def assign_variant(self, user_id: str, context: Optional[Dict] = None) -> str:
        return self.bandit.select_variant(self.config.test_id, context)
    
    def record_observation(self, variant: str, value: float):
        self.data[variant].append(value)
        self.bandit.update(self.config.test_id, variant, value)
    
    def get_results(self) -> Dict[str, TestResult]:
        results = {}
        
        for variant in self.config.variants:
            variant_data = np.array(self.data[variant])
            
            if len(variant_data) == 0:
                results[variant] = TestResult(
                    variant=variant,
                    sample_size=0,
                    mean=0.0,
                    std=0.0,
                    confidence_interval=(0.0, 0.0)
                )
                continue
            
            mean = float(np.mean(variant_data))
            std = float(np.std(variant_data, ddof=1))
            sample_size = len(variant_data)
            
            if sample_size > 1:
                sem = std / np.sqrt(sample_size)
                ci_low, ci_high = stats.t.interval(
                    self.config.confidence_level, 
                    sample_size - 1, 
                    loc=mean, 
                    scale=sem
                )
            else:
                ci_low, ci_high = mean, mean
            
            results[variant] = TestResult(
                variant=variant,
                sample_size=sample_size,
                mean=mean,
                std=std,
                confidence_interval=(float(ci_low), float(ci_high))
            )
        
        return results
    
    def calculate_statistical_significance(self) -> Dict[str, float]:
        results = self.get_results()
        control_variant = self.config.variants[0]
        control_data = np.array(self.data[control_variant])
        
        p_values = {}
        
        for variant in self.config.variants[1:]:
            variant_data = np.array(self.data[variant])
            
            if len(control_data) > 0 and len(variant_data) > 0:
                if self.config.metric_type == MetricType.BINARY:
                    control_success = np.sum(control_data)
                    variant_success = np.sum(variant_data)
                    _, p_value = StatisticalTest.chi_square_test(
                        int(control_success), len(control_data),
                        int(variant_success), len(variant_data)
                    )
                else:
                    _, p_value = StatisticalTest.t_test(control_data, variant_data)
                
                p_values[variant] = p_value
        
        return p_values

class ABTestManager:
    def __init__(self):
        self.active_tests: Dict[str, AdaptiveABTest] = {}
        self.test_configs: Dict[str, TestConfig] = {}
    
    def create_test(self, config: TestConfig) -> None:
        if config.test_id in self.active_tests:
            raise ValueError(f"Test {config.test_id} already exists")
        
        self.test_configs[config.test_id] = config
        self.active_tests[config.test_id] = AdaptiveABTest(config)
    
    def assign_variant(self, test_id: str, user_id: str, context: Optional[Dict] = None) -> str:
        if test_id not in self.active_tests:
            raise ValueError(f"Test {test_id} not found")
        
        return self.active_tests[test_id].assign_variant(user_id, context)
    
    def record_metric(self, test_id: str, variant: str, value: float) -> None:
        if test_id not in self.active_tests:
            raise ValueError(f"Test {test_id} not found")
        
        self.active_tests[test_id].record_observation(variant, value)
    
    def get_test_results(self, test_id: str) -> Tuple[Dict[str, TestResult], Dict[str, float]]:
        if test_id not in self.active_tests:
            raise ValueError(f"Test {test_id} not found")
        
        test = self.active_tests[test_id]
        results = test.get_results()
        p_values = test.calculate_statistical_significance()
        
        return results, p_values
    
    def stop_test(self, test_id: str) -> Dict:
        if test_id not in self.active_tests:
            raise ValueError(f"Test {test_id} not found")
        
        results, p_values = self.get_test_results(test_id)
        
        summary = {
            'test_id': test_id,
            'results': results,
            'p_values': p_values,
            'total_observations': sum(len(test.data[v]) for v in self.test_configs[test_id].variants)
        }
        
        del self.active_tests[test_id]
        del self.test_configs[test_id]
        
        return summary