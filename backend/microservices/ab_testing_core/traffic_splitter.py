# backend/ab_testing/traffic_splitter.py
import hashlib
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class ABVariant:
    """Вариант A/B теста"""
    name: str
    weight: float = 1.0  # Относительный вес (для неравного split)
    description: Optional[str] = None


class FixedTrafficSplitter:
    
    def __init__(
        self, 
        variants: List[ABVariant],
        seed: int = 42,
        hash_space_size: int = 10000
    ):
        self.variants = variants
        self.seed = seed
        self.hash_space_size = hash_space_size
        
        # Нормализуем веса
        self.split_ratios = self._normalize_weights(variants)
        
        # Рассчитываем границы в hash space
        self.boundaries = self._calculate_boundaries()
        
        # Статистика
        self.assignment_counts: Dict[str, int] = {v.name: 0 for v in variants}
    
    def _normalize_weights(self, variants: List[ABVariant]) -> List[float]:
        total_weight = sum(v.weight for v in variants)
        return [v.weight / total_weight for v in variants]
    
    def _calculate_boundaries(self) -> List[int]:
        boundaries = [0]
        cumulative = 0
        
        for ratio in self.split_ratios:
            cumulative += ratio
            boundary = int(cumulative * self.hash_space_size)
            boundaries.append(boundary)
        
        return boundaries
    
    def assign_variant(self, user_id: str, test_id: str = "") -> str:
        return self.assign_variant_with_metadata(user_id=user_id, test_id=test_id)["variant"]

    def assign_variant_with_metadata(self, user_id: str, test_id: str = "") -> Dict[str, Optional[float]]:
        hash_input = f"{test_id}:{user_id}:{self.seed}".encode('utf-8')

        hash_digest = hashlib.sha256(hash_input).hexdigest()

        hash_value = int(hash_digest, 16) % self.hash_space_size

        selected_variant = self.variants[-1].name
        for i, variant in enumerate(self.variants):
            if self.boundaries[i] <= hash_value < self.boundaries[i + 1]:
                selected_variant = variant.name
                break

        self.assignment_counts[selected_variant] += 1
        return {
            "variant": selected_variant,
            "hash_bucket": int(hash_value),
            "hash_space_size": int(self.hash_space_size),
            "seed": int(self.seed),
            "splitter_type": "fixed",
        }
    
    def get_assignment_stats(self) -> Dict[str, any]:
        total = sum(self.assignment_counts.values())
        
        if total == 0:
            return {
                'total_assignments': 0,
                'variant_counts': self.assignment_counts,
                'variant_percentages': {},
                'expected_percentages': {v.name: r * 100 for v, r in zip(self.variants, self.split_ratios)},
                'deviation': {}
            }
        
        actual_percentages = {
            name: (count / total) * 100 
            for name, count in self.assignment_counts.items()
        }
        
        expected_percentages = {
            v.name: r * 100 
            for v, r in zip(self.variants, self.split_ratios)
        }
        
        deviation = {
            name: actual_percentages[name] - expected_percentages[name]
            for name in actual_percentages
        }
        
        return {
            'total_assignments': total,
            'variant_counts': self.assignment_counts,
            'variant_percentages': actual_percentages,
            'expected_percentages': expected_percentages,
            'deviation': deviation,
            'max_deviation': max(abs(d) for d in deviation.values()) if deviation else 0.0
        }
    
    def reset_stats(self):
        self.assignment_counts = {v.name: 0 for v in self.variants}


class AdaptiveTrafficSplitter:
    def __init__(self, variants: List[ABVariant]):
        self.variants = variants
        # Thompson Sampling parameters
        self.successes: Dict[str, float] = {v.name: 1.0 for v in variants}
        self.failures: Dict[str, float] = {v.name: 1.0 for v in variants}
        self.assignment_counts: Dict[str, int] = {v.name: 0 for v in variants}
    
    def assign_variant(self, user_id: str) -> str:
        return self.assign_variant_with_metadata(user_id)["variant"]

    def assign_variant_with_metadata(self, user_id: str) -> Dict[str, Optional[float]]:
        samples = {}
        for variant in self.variants:
            alpha = self.successes[variant.name]
            beta = self.failures[variant.name]
            samples[variant.name] = np.random.beta(alpha, beta)

        selected = max(samples, key=samples.get)
        self.assignment_counts[selected] += 1
        return {
            "variant": selected,
            "hash_bucket": None,
            "hash_space_size": None,
            "seed": None,
            "splitter_type": "adaptive",
        }
    
    def update(self, variant: str, reward: float):
        normalized_reward = max(0.0, min(1.0, reward))
        
        self.successes[variant] += normalized_reward
        self.failures[variant] += (1.0 - normalized_reward)
    
    def get_variant_probabilities(self) -> Dict[str, float]:
        samples = 10000
        counts = {v.name: 0 for v in self.variants}
        
        for _ in range(samples):
            sampled = {}
            for variant in self.variants:
                alpha = self.successes[variant.name]
                beta = self.failures[variant.name]
                sampled[variant.name] = np.random.beta(alpha, beta)
            
            winner = max(sampled, key=sampled.get)
            counts[winner] += 1
        
        return {name: count / samples for name, count in counts.items()}


def create_equal_split_variants(variant_names: List[str]) -> List[ABVariant]:
    return [ABVariant(name=name, weight=1.0) for name in variant_names]


def create_weighted_split_variants(variants_config: Dict[str, float]) -> List[ABVariant]:
    return [ABVariant(name=name, weight=weight) for name, weight in variants_config.items()]


if __name__ == "__main__":
    variants = create_equal_split_variants(["control", "treatment"])
    splitter = FixedTrafficSplitter(variants, seed=42)
     
    for i in range(10000):
        user_id = f"user_{i}"
        variant = splitter.assign_variant(user_id, test_id="experiment_123")
    
    stats = splitter.get_assignment_stats()

    variants_90_10 = create_weighted_split_variants({"control": 0.9, "treatment": 0.1})
    splitter_90_10 = FixedTrafficSplitter(variants_90_10, seed=42)
    
    for i in range(10000):
        splitter_90_10.assign_variant(f"user_{i}", "exp_90_10")
    
    stats_90_10 = splitter_90_10.get_assignment_stats()
    
    adaptive = AdaptiveTrafficSplitter(create_equal_split_variants(["A", "B"]))
    
    for i in range(1000):
        variant = adaptive.assign_variant(f"user_{i}")
        
        if variant == "A":
            reward = 0.1  
        else:
            reward = 0.2  
        
        adaptive.update(variant, reward)
    
    probs = adaptive.get_variant_probabilities()

