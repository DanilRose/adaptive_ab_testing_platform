# backend/ab_testing/traffic_splitter.py
"""
Fixed Traffic Splitting для A/B Testing

Google/Meta стандарт:
- Детерминированное распределение на основе hash(user_id)
- НЕ адаптивное (НЕ Thompson Sampling)
- Постоянство: user_id всегда получает один вариант
- Равномерность: близко к заданным пропорциям
"""

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
    """
    Детерминированный traffic splitter
    
    Преимущества:
    ✅ Детерминированность: один user_id → один вариант всегда
    ✅ Равномерность: распределение близко к заданным пропорциям
    ✅ Независимость: от порядка прихода юзеров
    ✅ Статистическая чистота: не вносит bias
    
    Используется в:
    - Google (все продукты)
    - Meta (Facebook, Instagram)
    - Netflix
    - Airbnb
    - Booking.com
    """
    
    def __init__(
        self, 
        variants: List[ABVariant],
        seed: int = 42,
        hash_space_size: int = 10000
    ):
        """
        Args:
            variants: Список вариантов теста
            seed: Seed для воспроизводимости
            hash_space_size: Размер hash space (обычно 10000)
        """
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
        """
        Нормализует веса вариантов в пропорции
        
        Example:
            [A(weight=1), B(weight=1)] → [0.5, 0.5]
            [A(weight=2), B(weight=1)] → [0.67, 0.33]
        """
        total_weight = sum(v.weight for v in variants)
        return [v.weight / total_weight for v in variants]
    
    def _calculate_boundaries(self) -> List[int]:
        """
        Рассчитывает границы в hash space
        
        Example для [0.5, 0.5] и hash_space=10000:
            [0, 5000, 10000]
        """
        boundaries = [0]
        cumulative = 0
        
        for ratio in self.split_ratios:
            cumulative += ratio
            boundary = int(cumulative * self.hash_space_size)
            boundaries.append(boundary)
        
        return boundaries
    
    def assign_variant(self, user_id: str, test_id: str = "") -> str:
        """
        Назначает вариант на основе хеша user_id
        
        Args:
            user_id: Уникальный ID пользователя
            test_id: ID теста (опционально, для изоляции экспериментов)
            
        Returns:
            Название варианта
        """
        # Создаем уникальный ключ для хеширования
        hash_input = f"{test_id}:{user_id}:{self.seed}".encode('utf-8')
        
        # SHA256 хеш
        hash_digest = hashlib.sha256(hash_input).hexdigest()
        
        # Конвертируем в число [0, hash_space_size)
        hash_value = int(hash_digest, 16) % self.hash_space_size
        
        # Находим вариант по границам
        for i, variant in enumerate(self.variants):
            if self.boundaries[i] <= hash_value < self.boundaries[i + 1]:
                self.assignment_counts[variant.name] += 1
                return variant.name
        
        # Fallback (не должно произойти)
        return self.variants[-1].name
    
    def get_assignment_stats(self) -> Dict[str, any]:
        """
        Возвращает статистику распределения
        
        Полезно для проверки равномерности
        """
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
        """Сбрасывает статистику (для новых симуляций)"""
        self.assignment_counts = {v.name: 0 for v in self.variants}


class AdaptiveTrafficSplitter:
    """
    Адаптивный traffic splitter (НЕ РЕКОМЕНДУЕТСЯ для production)
    
    ⚠️ ВНИМАНИЕ: Используйте только для:
    - Exploration (поиск лучших вариантов среди многих)
    - MAB (Multi-Armed Bandit) задачи
    - Быстрое прототипирование
    
    ❌ НЕ используйте для:
    - Финальной валидации (используйте FixedTrafficSplitter)
    - Статистического inference (p-values будут невалидны)
    - Продакшн A/B тестов
    
    Почему НЕ рекомендуется:
    1. Selection bias: лучший вариант получает больше трафика → inflated metrics
    2. Невалидные p-values: assumptions статтестов нарушены
    3. Невоспроизводимость: разные прогоны = разные результаты
    """
    
    def __init__(self, variants: List[ABVariant]):
        self.variants = variants
        # Thompson Sampling parameters
        self.successes: Dict[str, float] = {v.name: 1.0 for v in variants}
        self.failures: Dict[str, float] = {v.name: 1.0 for v in variants}
        self.assignment_counts: Dict[str, int] = {v.name: 0 for v in variants}
    
    def assign_variant(self, user_id: str) -> str:
        """
        Thompson Sampling: sample из Beta распределений
        
        ⚠️ WARNING: Не детерминировано! 
        Тот же user_id может получить разные варианты!
        """
        samples = {}
        for variant in self.variants:
            alpha = self.successes[variant.name]
            beta = self.failures[variant.name]
            samples[variant.name] = np.random.beta(alpha, beta)
        
        selected = max(samples, key=samples.get)
        self.assignment_counts[selected] += 1
        return selected
    
    def update(self, variant: str, reward: float):
        """
        Обновляет Beta распределение на основе результата
        
        Args:
            variant: Название варианта
            reward: Награда [0, 1] (нормализованная метрика)
        """
        # Нормализуем reward в [0, 1]
        normalized_reward = max(0.0, min(1.0, reward))
        
        self.successes[variant] += normalized_reward
        self.failures[variant] += (1.0 - normalized_reward)
    
    def get_variant_probabilities(self) -> Dict[str, float]:
        """
        Возвращает текущие вероятности выбора вариантов
        
        Полезно для мониторинга адаптации
        """
        # Сэмплируем много раз и считаем частоты
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
    """
    Создает варианты с равным весом
    
    Example:
        ["control", "treatment"] → 50/50 split
        ["A", "B", "C"] → 33.3/33.3/33.3 split
    """
    return [ABVariant(name=name, weight=1.0) for name in variant_names]


def create_weighted_split_variants(variants_config: Dict[str, float]) -> List[ABVariant]:
    """
    Создает варианты с кастомными весами
    
    Example:
        {"control": 0.9, "treatment": 0.1} → 90/10 split
    """
    return [ABVariant(name=name, weight=weight) for name, weight in variants_config.items()]


# Пример использования
if __name__ == "__main__":
    # 1. Равный split (Google standard)
    variants = create_equal_split_variants(["control", "treatment"])
    splitter = FixedTrafficSplitter(variants, seed=42)
    
    # Симулируем 10000 пользователей
    for i in range(10000):
        user_id = f"user_{i}"
        variant = splitter.assign_variant(user_id, test_id="experiment_123")
    
    stats = splitter.get_assignment_stats()
    print("✅ Fixed Traffic Split Stats:")
    print(f"  Total assignments: {stats['total_assignments']}")
    print(f"  Control: {stats['variant_counts']['control']} ({stats['variant_percentages']['control']:.2f}%)")
    print(f"  Treatment: {stats['variant_counts']['treatment']} ({stats['variant_percentages']['treatment']:.2f}%)")
    print(f"  Max deviation: {stats['max_deviation']:.2f}%")
    
    # 2. Неравный split (90/10)
    print("\n90/10 Split:")
    variants_90_10 = create_weighted_split_variants({"control": 0.9, "treatment": 0.1})
    splitter_90_10 = FixedTrafficSplitter(variants_90_10, seed=42)
    
    for i in range(10000):
        splitter_90_10.assign_variant(f"user_{i}", "exp_90_10")
    
    stats_90_10 = splitter_90_10.get_assignment_stats()
    print(f"  Control: {stats_90_10['variant_percentages']['control']:.2f}%")
    print(f"  Treatment: {stats_90_10['variant_percentages']['treatment']:.2f}%")
    
    # 3. ⚠️ Adaptive (НЕ рекомендуется)
    print("\n⚠️ Adaptive Traffic Split (NOT RECOMMENDED for production):")
    adaptive = AdaptiveTrafficSplitter(create_equal_split_variants(["A", "B"]))
    
    # Симулируем, что вариант B лучше
    for i in range(1000):
        variant = adaptive.assign_variant(f"user_{i}")
        
        # Фейковые результаты: B в 2 раза лучше
        if variant == "A":
            reward = 0.1  # 10% success
        else:
            reward = 0.2  # 20% success
        
        adaptive.update(variant, reward)
    
    probs = adaptive.get_variant_probabilities()
    print(f"  Variant A probability: {probs['A']:.2f}")
    print(f"  Variant B probability: {probs['B']:.2f}")
    print(f"  ⚠️ Notice: B gets more traffic → selection bias!")
