import asyncio
from datetime import datetime, timedelta
from typing import Dict, Optional, Any, List
from dataclasses import dataclass, asdict

import pandas as pd
import numpy as np

from backend.microservices.ab_testing_core.traffic_splitter import (
    FixedTrafficSplitter,
    AdaptiveTrafficSplitter,
    ABVariant,
    create_equal_split_variants,
)
from backend.microservices.ab_testing_core.statistics import (
    SequentialTesting,
    SRMChecker,
    StatisticalAnalyzer,
    SampleSizeCalculator,
    run_full_ab_analysis,
)
from backend.microservices.database.session import SessionLocal, AsyncSessionLocal
from backend.microservices.database import crud
from backend.microservices.database.models import ABTestORM, ABTestTimeSeriesORM
from backend.microservices.data_gan.service import DatasetPersistenceService


@dataclass
class SimulationConfig:
    """Конфигурация симуляции"""
    test_id: str
    dataset_id: int  
    user_count: int
    real_world_duration_days: int = 14  
    simulation_duration_minutes: Optional[int] = None  
    traffic_split_strategy: str = "fixed" 
    enable_sequential_testing: bool = True
    enable_srm_check: bool = True
    # В прод-сценарии чаще ожидается полный прогон по выбранному synthetic dataset.
    # Early stopping оставляем как отдельный управляемый режим.
    enable_early_stopping: bool = False
    max_sequential_looks: int = 5
    variant_effects: Optional[Dict[str, Dict[str, float]]] = None


@dataclass
class SimulationProgress:
    """Прогресс симуляции в реальном времени"""
    users_processed: int
    total_users: int
    progress_percent: float
    variant_counts: Dict[str, int]
    current_look: int
    srm_status: str
    estimated_completion_time: datetime
    can_stop_early: bool
    early_stop_reason: Optional[str]


class GoogleStandardABTestSimulator:
    """
    A/B Test Simulator по стандартам Google/Meta
    
    Workflow:
    1. Валидация: проверка наличия GAN-датасета
    2. Setup: настройка traffic splitter (fixed/adaptive)
    3. Simulation: запуск с правильной временной шкалой
    4. Sequential Testing: проверка early stopping каждые 20%
    5. SRM Check: проверка рандомизации
    6. Analysis: полная статистика с коррекциями
    """
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.progress: Optional[SimulationProgress] = None

        self._validate_dataset()

        self.traffic_splitter = self._setup_traffic_splitter()

        if config.enable_sequential_testing:
            self.sequential_tester = SequentialTesting(
                alpha=0.05,
                max_looks=config.max_sequential_looks
            )
        else:
            self.sequential_tester = None


        if config.enable_srm_check:
            self.srm_checker = SRMChecker()
        else:
            self.srm_checker = None


        self.analyzer = StatisticalAnalyzer(alpha=0.05)

        self.results_by_variant: Dict[str, List[float]] = {}
        # Для ratio-метрик храним компоненты numerator/denominator по вариантам,
        # чтобы не деградировать в denominator=1 при промежуточном анализе.
        self.ratio_components_by_variant: Dict[str, Dict[str, List[float]]] = {}
        self.session_data: List[Dict[str, Any]] = []

        self.time_series_data: List[Dict[str, Any]] = []
        self.time_series_interval: int = 20  
        

        self._early_stop_triggered = False
        self._early_stop_reason: Optional[str] = None
        self._sequential_look_at_stop: int = 0
    
    def _validate_dataset(self):
        """Валидация наличия синтетического датасета"""
        try:
            with SessionLocal() as db:
                dataset = crud.get_generated_data_by_id(db, self.config.dataset_id)

                if dataset is None:
                    raise ValueError(
                        f"❌ Dataset {self.config.dataset_id} not found! "
                        "You must generate synthetic data first using GAN Manager."
                    )

                if dataset.data_type != "synthetic":
                    raise ValueError(
                        f"❌ Dataset {self.config.dataset_id} is not synthetic! "
                        f"Found type: {dataset.data_type}. Only synthetic data allowed for A/B testing."
                    )

                # Проверка достаточности данных
                if dataset.sample_count < self.config.user_count:
                    raise ValueError(
                        f"❌ Insufficient synthetic data: "
                        f"need {self.config.user_count}, have {dataset.sample_count}. "
                        "Generate more data in GAN Manager."
                    )

                print(f"✅ Dataset validation passed: {dataset.sample_count} synthetic records available")
        except Exception as e:
            # Пробрасываем ValueError как есть, остальные оборачиваем
            if isinstance(e, ValueError):
                raise
            raise ValueError(f"Dataset validation failed: {str(e)}")
    
    def _setup_traffic_splitter(self):
        """Настройка traffic splitter с предупреждениями"""
        with SessionLocal() as db:
            test = db.query(ABTestORM).filter(ABTestORM.test_id == self.config.test_id).first()
            if not test:
                raise ValueError(f"Test {self.config.test_id} not found")
            
            variants = [ABVariant(name=v, weight=1.0) for v in test.variants]
        
        if self.config.traffic_split_strategy == "fixed":
            print("✅ Using Fixed Traffic Split (Google/Meta standard)")
            return FixedTrafficSplitter(variants, seed=42)
        
        elif self.config.traffic_split_strategy == "adaptive":
            print(
                "⚠️  WARNING: Using Adaptive Traffic Split (Thompson Sampling)\n"
                "   This method introduces selection bias and invalidates p-values.\n"
                "   Use only for:\n"
                "   - Exploration phase (finding best of 5+ variants)\n"
                "   - MAB tasks (recommendations, personalization)\n"
                "   - NOT for final validation!\n"
                "   Consider using 'fixed' strategy for production A/B tests.\n"
            )
            return AdaptiveTrafficSplitter(variants)
        
        else:
            raise ValueError(f"Unknown strategy: {self.config.traffic_split_strategy}")
    
    def _load_synthetic_data(self) -> pd.DataFrame:
        """Загрузка синтетических данных из БД"""
        with SessionLocal() as db:
            dataset = crud.get_generated_data_by_id(db, self.config.dataset_id)
            
            records = DatasetPersistenceService.load_dataset_records_for_entity(dataset)

            if not records:
                raise ValueError("No records found in synthetic dataset")

            return pd.DataFrame(records).head(self.config.user_count)
    
    def _get_test_config(self) -> Dict[str, Any]:
        """Получение конфигурации теста"""
        with SessionLocal() as db:
            test = db.query(ABTestORM).filter(ABTestORM.test_id == self.config.test_id).first()
            if not test:
                raise ValueError(f"Test {self.config.test_id} not found")

            extra = dict(test.extra_config or {})
            early_stopping_enabled = bool(extra.get("early_stopping_enabled", self.config.enable_early_stopping))

            return {
                'test_id': test.test_id,
                'primary_metric': test.primary_metric,
                'metric_type': test.metric_type,
                'variants': test.variants,
                'mde_percent': test.mde_percent,
                'sample_size': test.sample_size,
                'confidence_level': test.confidence_level,
                'power': test.power,
                'early_stopping_enabled': early_stopping_enabled,
            }
    
    def _estimate_simulation_duration_minutes(
        self,
        user_count: int,
        sample_size: Optional[int],
        confidence_level: float,
    ) -> int:
        """
        Динамически оценивает длительность симуляции.
        Таргет:
        - ~500 пользователей -> ~5 минут
        - ~50 000 пользователей -> ~60 минут
        """
        user_count = max(1, int(user_count))
        min_users, max_users = 500.0, 50000.0
        min_minutes, max_minutes = 5.0, 60.0

        # Логарифмическая интерполяция по объёму трафика
        log_user = np.log10(float(user_count))
        log_min = np.log10(min_users)
        log_max = np.log10(max_users)
        traffic_factor = (log_user - log_min) / max(1e-9, (log_max - log_min))
        traffic_factor = float(np.clip(traffic_factor, 0.0, 1.0))

        base_minutes = min_minutes + (max_minutes - min_minutes) * traffic_factor

        # Модификатор от статистических параметров теста
        confidence_factor = 1.0 + max(0.0, float(confidence_level) - 0.95) * 2.0
        sample_factor = 1.0
        if sample_size and sample_size > 0:
            sample_factor = float(np.clip(user_count / float(sample_size), 0.7, 1.3))

        estimated = base_minutes * confidence_factor * sample_factor
        return int(np.clip(round(estimated), 5, 120))

    def _calculate_delay_per_user(self) -> float:
        """Рассчитывает задержку между пользователями для сжатой временной шкалы."""
        simulation_minutes = self.config.simulation_duration_minutes or 20
        total_simulation_seconds = simulation_minutes * 60
        return total_simulation_seconds / max(1, self.config.user_count)

    def _build_metric_profile(self, data: pd.DataFrame, primary_metric: str) -> Dict[str, Any]:
        """Строит профиль метрики на основе реального датасета (без хардкода полей)."""
        profile: Dict[str, Any] = {
            "primary_exists": primary_metric in data.columns,
            "is_binary": False,
            "mean": 0.1,
            "std": 0.05,
            "min": 0.0,
            "max": 1.0,
            "numeric_columns": [],
        }

        numeric_df = data.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
        numeric_cols = list(numeric_df.columns)
        profile["numeric_columns"] = numeric_cols

        if primary_metric in data.columns:
            raw = pd.to_numeric(data[primary_metric], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            if len(raw) > 0:
                p_mean = float(raw.mean())
                p_std = float(raw.std(ddof=1)) if len(raw) > 1 else 0.0
                p_min = float(raw.min())
                p_max = float(raw.max())

                unique_values = set(raw.round(8).unique().tolist())
                is_binary = unique_values.issubset({0.0, 1.0}) or (0.0 <= p_min <= 1.0 and 0.0 <= p_max <= 1.0 and p_std < 0.5)

                profile.update({
                    "is_binary": bool(is_binary),
                    "mean": p_mean,
                    "std": max(1e-6, p_std),
                    "min": p_min,
                    "max": p_max,
                })
                return profile

        if len(numeric_cols) > 0:
            values = numeric_df.values.flatten()
            values = values[~np.isnan(values)]
            if len(values) > 0:
                global_mean = float(np.mean(values))
                global_std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
                profile.update({
                    "mean": global_mean,
                    "std": max(1e-6, global_std),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "is_binary": False,
                })

        return profile

    def _generate_metric_value(
        self,
        user: pd.Series,
        variant: str,
        primary_metric: str,
        metric_profile: Dict[str, Any],
        metric_type: str,
    ) -> float:
        """Адаптивно получает метрику из датасета или генерирует по статистике датасета."""
        variant_multiplier = self._get_variant_multiplier(variant, primary_metric)

        primary_exists = metric_profile.get("primary_exists", False)
        mean_val = float(metric_profile.get("mean", 0.1))
        std_val = float(metric_profile.get("std", 0.05))
        min_val = float(metric_profile.get("min", 0.0))
        max_val = float(metric_profile.get("max", 1.0))

        is_binary_metric = metric_type == "binary" or metric_profile.get("is_binary", False) or primary_metric in ["conversion", "click", "click_rate"]

        if primary_exists:
            raw_value = pd.to_numeric(pd.Series([user.get(primary_metric)]), errors="coerce").iloc[0]
            if pd.notna(raw_value):
                if is_binary_metric:
                    # Для бинарных метрик (0/1) нельзя использовать само значение как вероятность:
                    # raw_value = 0 → prob = 0 * multiplier = 0 (нет эффекта!)
                    # raw_value = 1 → prob = 1 * multiplier = 1 (нет вариации!)
                    # Правильный подход: использовать базовую вероятность из датасета (mean)
                    # и применить к ней multiplier варианта.
                    baseline_prob = float(np.clip(mean_val, 0.01, 0.95))
                    prob = float(np.clip(baseline_prob * variant_multiplier, 0.0, 0.999))
                    return 1.0 if np.random.random() < prob else 0.0

                value = float(raw_value)

                # Для revenue избегаем вырожденной массы в нуле,
                # которая резко снижает чувствительность непрерывного теста.
                if primary_metric == "revenue" and value <= 0:
                    value = float(max(1.0, np.random.normal(loc=max(1.0, mean_val * 0.35), scale=max(1.0, std_val * 0.2))))

                value *= variant_multiplier

                # Для непрерывных метрик не ограничиваем сверху историческим max,
                # чтобы не "съедать" эффект варианта.
                if primary_metric == "revenue":
                    return float(max(0.0, value))

                return float(np.clip(value, min_val, max_val if max_val > min_val else value))

        if is_binary_metric:
            baseline_prob = mean_val
            if baseline_prob > 1.0:
                baseline_prob /= 100.0
            baseline_prob = float(np.clip(baseline_prob, 0.01, 0.8))
            prob = float(np.clip(baseline_prob * variant_multiplier, 0.0, 0.999))
            return 1.0 if np.random.random() < prob else 0.0

        generated = float(np.random.normal(loc=mean_val, scale=max(1e-6, std_val)))
        generated *= variant_multiplier
        low = min_val if np.isfinite(min_val) else generated
        high = max_val if np.isfinite(max_val) and max_val > low else generated
        return float(np.clip(generated, low, high))

    def _normalize_reward_for_adaptive(self, raw_value: float, metric_type: str) -> float:
        """
        Нормализация reward для Thompson Sampling в [0, 1].
        - binary: ожидаем 0/1
        - continuous/ratio: мягкая логистическая нормализация
        """
        value = float(raw_value)
        mt = str(metric_type or "continuous").lower()

        if mt == "binary":
            return float(np.clip(value, 0.0, 1.0))

        scale = max(1.0, abs(value) + 1.0)
        normalized = 1.0 / (1.0 + np.exp(-value / scale))
        return float(np.clip(normalized, 0.0, 1.0))
    
    def _calculate_conversion_probability(
        self, 
        user: pd.Series, 
        variant: str,
        primary_metric: str
    ) -> float:
        """Рассчет вероятности конверсии с учетом эффекта варианта"""
        base_prob = 0.1
        
        # Факторы пользователя
        if user.get('user_type') == 'shopper':
            base_prob += 0.2
        if user.get('previous_purchases', 0) > 0:
            base_prob += 0.15
        if user.get('loyalty_score', 0) > 0.7:
            base_prob += 0.1
        if user.get('traffic_source') == 'direct':
            base_prob += 0.05
        
        age = user.get('age', 35)
        income = user.get('income', 0)
        age_factor = max(0, (45 - abs(age - 35)) / 100)
        income_factor = min(0.2, income / 500000)
        
        base_probability = min(0.8, base_prob + age_factor + income_factor)
        
        variant_effect = self._get_variant_multiplier(variant, 'conversion')
        
        return min(0.95, base_probability * variant_effect)
    
    def _calculate_revenue(self, user: pd.Series, variant: str) -> float:
        """Рассчет revenue с учетом эффекта варианта"""
        income = user.get('income', 0)
        base_revenue = income * 0.02
        
        if user.get('user_type') == 'shopper':
            base_revenue *= 1.5
        if user.get('previous_purchases', 0) > 3:
            base_revenue *= 1.3
        if user.get('loyalty_score', 0) > 0.8:
            base_revenue *= 1.2
        
        noise = np.random.normal(1.0, 0.2)
        base_revenue = max(10, base_revenue * noise)
        
        variant_effect = self._get_variant_multiplier(variant, 'revenue')
        
        return base_revenue * variant_effect
    
    def _get_variant_multiplier(self, variant: str, metric: str) -> float:
        """Получает multiplier эффекта для варианта"""
        if not self.config.variant_effects:
            return 1.0
        
        variant_config = self.config.variant_effects.get(variant, {})
        return float(variant_config.get(metric, 1.0))
    
    def _extract_ratio_components(
        self,
        user: pd.Series,
        primary_metric: str,
        ratio_value: float,
    ) -> Optional[tuple[float, float]]:
        """Пытается извлечь корректные numerator/denominator для ratio-метрики из данных пользователя."""
        num_key = f"{primary_metric}_numerator"
        den_key = f"{primary_metric}_denominator"

        num_raw = user.get(num_key)
        den_raw = user.get(den_key)

        try:
            if num_raw is not None and den_raw is not None:
                num = float(num_raw)
                den = float(den_raw)
                if np.isfinite(num) and np.isfinite(den) and den > 0:
                    return float(num), float(den)
        except Exception:
            pass

        # Практический фолбэк для CTR-подобных метрик: clicks / impressions
        # (данные генератора содержат click и pages_per_session).
        pm = str(primary_metric or "").lower()
        if pm in {"ctr", "click_rate", "clickthrough_rate"}:
            try:
                clicks = float(user.get("click", 0.0))
                impressions = float(user.get("pages_per_session", 0.0))
                if np.isfinite(clicks) and np.isfinite(impressions) and impressions > 0:
                    return float(max(0.0, clicks)), float(impressions)
            except Exception:
                pass

        # Если ratio-компоненты не восстановимы, лучше пропустить p-value ratio-сравнения,
        # чем искажать его denominator=1.
        return None

    def _should_check_progress(self, users_processed: int) -> bool:
        """Проверяем прогресс каждые 20% пользователей"""
        if not self.config.enable_sequential_testing:
            return False

        check_points = [
            int(self.config.user_count * 0.2),
            int(self.config.user_count * 0.4),
            int(self.config.user_count * 0.6),
            int(self.config.user_count * 0.8),
            self.config.user_count
        ]

        return users_processed in check_points

    def _save_time_series_snapshot(self, users_processed: int):
        """Сохраняет срез данных для временных рядов (в памяти и в БД)"""
        test_config = self._get_test_config()
        variants = test_config['variants']
        control_variant = variants[0]
        control_data = np.array(self.results_by_variant.get(control_variant, []))

        for variant in variants:
            data = np.array(self.results_by_variant.get(variant, []))
            if len(data) == 0:
                continue

            cumulative_metric = float(np.sum(data))
            mean_metric = float(np.mean(data))

            # Расчет p-value для варианта против контроля
            p_value = None
            ci_lower = None
            ci_upper = None

            if variant != control_variant and len(control_data) >= 30 and len(data) >= 30:
                try:
                    metric_type = str(test_config.get('metric_type', 'continuous'))
                    if metric_type == 'ratio':
                        control_ratio = self.ratio_components_by_variant.get(control_variant, {"numerators": [], "denominators": []})
                        treat_ratio = self.ratio_components_by_variant.get(variant, {"numerators": [], "denominators": []})

                        control_num = np.asarray(control_ratio.get("numerators", []), dtype=float)
                        control_den = np.asarray(control_ratio.get("denominators", []), dtype=float)
                        treat_num = np.asarray(treat_ratio.get("numerators", []), dtype=float)
                        treat_den = np.asarray(treat_ratio.get("denominators", []), dtype=float)

                        if len(control_num) >= 10 and len(treat_num) >= 10:
                            ratio_result = self.analyzer.analyze_ratio_metric(
                                control_numerators=control_num,
                                control_denominators=control_den,
                                treatment_numerators=treat_num,
                                treatment_denominators=treat_den,
                                num_comparisons=max(1, len(variants) - 1),
                            )
                            p_value = float(ratio_result.p_value)
                        else:
                            p_value = None
                    else:
                        from scipy import stats
                        # Welch t-test для консистентности с финальной аналитикой.
                        _, p_raw = stats.ttest_ind(control_data, data, equal_var=False)
                        p_value = float(p_raw)

                    # ДИ для среднего значения МЕТРИКИ самого варианта (а не для разности с контролем),
                    # чтобы график "Доверительные интервалы" отображал корректную сущность.
                    from scipy import stats
                    variant_std = float(np.std(data, ddof=1)) if len(data) > 1 else 0.0
                    variant_se = variant_std / np.sqrt(max(1, len(data)))
                    if variant_se > 0:
                        ci = stats.t.interval(0.95, len(data) - 1, loc=mean_metric, scale=variant_se)
                        ci_lower = float(ci[0])
                        ci_upper = float(ci[1])
                    else:
                        ci_lower = float(mean_metric)
                        ci_upper = float(mean_metric)
                except Exception:
                    pass

            snapshot = {
                'test_id': self.config.test_id,
                'users_processed': users_processed,
                'variant': variant,
                'cumulative_metric': cumulative_metric,
                'mean_metric': mean_metric,
                'sample_size': len(data),
                'p_value': p_value,
                'confidence_interval_lower': ci_lower,
                'confidence_interval_upper': ci_upper,
            }
            self.time_series_data.append(snapshot)
            
            # Сохраняем в БД в реальном времени
            self._save_snapshot_to_db_sync(snapshot)

    def _save_snapshot_to_db_sync(self, snapshot: dict):
        """Синхронное сохранение snapshot в БД (для использования во время симуляции)"""
        try:
            from backend.microservices.database.session import SessionLocal
            from backend.microservices.database import crud

            with SessionLocal() as db:
                crud.create_ab_test_time_series(
                    db,
                    test_id=snapshot['test_id'],
                    users_processed=snapshot['users_processed'],
                    variant=snapshot['variant'],
                    cumulative_metric=snapshot['cumulative_metric'],
                    mean_metric=snapshot['mean_metric'],
                    sample_size=snapshot['sample_size'],
                    p_value=snapshot['p_value'],
                    confidence_interval_lower=snapshot['confidence_interval_lower'],
                    confidence_interval_upper=snapshot['confidence_interval_upper'],
                    do_commit=True
                )
                # Логирование для отладки (каждые 100 пользователей)
                if snapshot['users_processed'] % 100 == 0:
                    print(f"  💾 Saved snapshot: {snapshot['variant']} | users={snapshot['users_processed']} | mean={snapshot['mean_metric']:.4f} | cum={snapshot['cumulative_metric']:.2f}")
        except Exception as e:
            # Тихо игнорируем ошибки сохранения, чтобы не прерывать симуляцию
            print(f"⚠️ Warning: Failed to save time series snapshot to DB: {e}")

    def _update_runtime_progress(
        self,
        users_processed: int,
        sequential_look: Optional[int] = None,
        srm_result: Optional[Any] = None,
    ):
        """Обновляет прогресс теста в БД во время выполнения симуляции."""
        try:
            with SessionLocal() as db:
                test = db.query(ABTestORM).filter(ABTestORM.test_id == self.config.test_id).first()
                if not test:
                    return

                target_users = test.sample_size or self.config.user_count or max(1, users_processed)
                test.total_users = int(users_processed)
                test.completion_percentage = float(min(100.0, (users_processed / max(1, target_users)) * 100.0))

                if sequential_look is not None:
                    test.current_sequential_look = int(sequential_look)

                if srm_result is not None:
                    test.srm_check_passed = 0 if srm_result.srm_detected else 1
                    test.srm_p_value = float(srm_result.p_value)

                db.commit()
        except Exception as e:
            print(f"⚠️ Warning: Failed to update runtime progress: {e}")

    async def _wait_if_paused(self) -> bool:
        """Ожидает, пока тест на паузе. Возвращает False, если тест остановлен/переведен в неподдерживаемый статус."""
        while True:
            with SessionLocal() as db:
                test = db.query(ABTestORM).filter(ABTestORM.test_id == self.config.test_id).first()
                if not test:
                    return False

                is_paused = test.status == "paused" or test.simulation_status == "paused"
                if is_paused:
                    await asyncio.sleep(0.5)
                    continue

                if test.status not in ["active", "paused"]:
                    return False

                return True

    def _perform_interim_analysis(
        self,
        users_processed: int
    ) -> Dict[str, Any]:
        """Промежуточный анализ с Sequential Testing и SRM check."""
        test_config = self._get_test_config()
        variants = test_config['variants']
        control_variant = variants[0]
        metric_type = str(test_config.get('metric_type', 'continuous'))

        # 1. SRM Check
        srm_result = None
        if self.config.enable_srm_check and self.srm_checker:
            variant_counts = {
                v: len(self.results_by_variant.get(v, []))
                for v in variants
            }
            srm_result = self.srm_checker.check_srm_by_variant(variant_counts)

            if srm_result.srm_detected:
                print(f"\n{srm_result.warning}")

        # 2. Statistical Test
        control_data = np.array(self.results_by_variant.get(control_variant, []), dtype=float)

        results = {}
        for variant in variants[1:]:
            treatment_data = np.array(self.results_by_variant.get(variant, []), dtype=float)

            if len(control_data) < 30 or len(treatment_data) < 30:
                continue

            # Основной тест выбирается по типу метрики
            if metric_type == 'binary':
                control_conv = int(np.sum(control_data))
                treatment_conv = int(np.sum(treatment_data))
                test_result = self.analyzer.analyze_binary_metric(
                    control_conversions=control_conv,
                    control_total=len(control_data),
                    treatment_conversions=treatment_conv,
                    treatment_total=len(treatment_data),
                    num_comparisons=max(1, len(variants) - 1),
                )
            elif metric_type == 'ratio':
                control_ratio = self.ratio_components_by_variant.get(control_variant, {"numerators": [], "denominators": []})
                treat_ratio = self.ratio_components_by_variant.get(variant, {"numerators": [], "denominators": []})

                control_num = np.asarray(control_ratio.get("numerators", []), dtype=float)
                control_den = np.asarray(control_ratio.get("denominators", []), dtype=float)
                treat_num = np.asarray(treat_ratio.get("numerators", []), dtype=float)
                treat_den = np.asarray(treat_ratio.get("denominators", []), dtype=float)

                if len(control_num) < 10 or len(treat_num) < 10:
                    continue

                test_result = self.analyzer.analyze_ratio_metric(
                    control_numerators=control_num,
                    control_denominators=control_den,
                    treatment_numerators=treat_num,
                    treatment_denominators=treat_den,
                    num_comparisons=max(1, len(variants) - 1),
                )
            else:
                test_result = self.analyzer.analyze_continuous_metric(
                    control_data,
                    treatment_data,
                    num_comparisons=max(1, len(variants) - 1),
                )

            # Sequential Testing check
            can_stop = False
            stop_reason = None

            if self.sequential_tester:
                # Early stopping разрешаем только при явном включении и после накопления
                # достаточного объёма данных (минимум 50% плановой выборки и минимум 500 юзеров).
                planned_users = int(test_config['sample_size'] or self.config.user_count or 0)
                min_users_for_early_stop = max(500, int(planned_users * 0.5))
                early_stop_allowed = bool(self.config.enable_early_stopping) and users_processed >= min_users_for_early_stop

                if early_stop_allowed:
                    can_stop, stop_reason = self.sequential_tester.should_stop_for_success(
                        test_result.p_value,
                        test_result.relative_uplift_percent,
                    )

                    # Futility check
                    if not can_stop:
                        mde_target = test_config['mde_percent']
                        can_stop_fut, stop_reason_fut = self.sequential_tester.should_stop_for_futility(
                            test_result.relative_uplift_percent / 100.0,
                            mde_target / 100.0,
                            users_processed,
                            test_config['sample_size'] or self.config.user_count,
                        )

                        if can_stop_fut:
                            can_stop = True
                            stop_reason = stop_reason_fut

            results[variant] = {
                'test_result': test_result,
                'can_stop_early': can_stop,
                'stop_reason': stop_reason,
            }

        return {
            'users_processed': users_processed,
            'srm_result': srm_result,
            'variant_results': results,
            'sequential_look': self.sequential_tester.current_look if self.sequential_tester else 0,
        }
    
    async def run_simulation(self) -> Dict[str, Any]:
        """
        Запуск симуляции A/B теста
        
        Returns:
            Полный отчет с результатами
        """
        print(f"\n{'='*80}")
        print(f"🚀 Starting Google-Standard A/B Test Simulation")
        print(f"{'='*80}")
        
        test_config = self._get_test_config()
        synthetic_data = self._load_synthetic_data()

        # Конфиг early stopping подтягиваем из теста/extra_config,
        # чтобы sequential-look в UI соответствовал фактическому режиму.
        self.config.enable_early_stopping = bool(test_config.get('early_stopping_enabled', self.config.enable_early_stopping))

        dynamic_minutes = self._estimate_simulation_duration_minutes(
            user_count=self.config.user_count,
            sample_size=test_config.get('sample_size'),
            confidence_level=float(test_config.get('confidence_level', 0.95)),
        )
        if self.config.simulation_duration_minutes is None:
            self.config.simulation_duration_minutes = dynamic_minutes

        delay_per_user = self._calculate_delay_per_user()
        metric_profile = self._build_metric_profile(synthetic_data, test_config['primary_metric'])
        
        print(f"\n📊 Configuration:")
        print(f"   Test ID: {self.config.test_id}")
        print(f"   Dataset ID: {self.config.dataset_id}")
        print(f"   Strategy: {self.config.traffic_split_strategy}")
        print(f"   Variants: {test_config['variants']}")
        print(f"   Primary Metric: {test_config['primary_metric']}")
        print(f"   Users: {self.config.user_count}")
        print(f"   Real-world Duration: {self.config.real_world_duration_days} days")
        print(f"   Simulation Duration: {self.config.simulation_duration_minutes} min")
        print(f"   Time Compression: {self.config.real_world_duration_days * 1440 / self.config.simulation_duration_minutes:.1f}x")
        print(f"   Delay per User: {delay_per_user:.3f} sec")
        
        for variant in test_config['variants']:
            self.results_by_variant[variant] = []
            self.ratio_components_by_variant[variant] = {
                "numerators": [],
                "denominators": [],
            }
        
        start_time = datetime.utcnow()
        variant_counts = {v: 0 for v in test_config['variants']}
        
        stop_simulation_early = False
        for i, (_, user) in enumerate(synthetic_data.iterrows(), start=1):
            user_id = str(user.get("user_id", i))
            
            if not await self._wait_if_paused():
                print("Simulation interrupted: test status is no longer active")
                break

            variant = self.traffic_splitter.assign_variant(user_id, self.config.test_id)
            variant_counts[variant] += 1
            
            primary_metric = test_config['primary_metric']

            metric_value = self._generate_metric_value(
                user=user,
                variant=variant,
                primary_metric=primary_metric,
                metric_profile=metric_profile,
                metric_type=str(test_config.get('metric_type', 'continuous')),
            )
            
            self.results_by_variant[variant].append(metric_value)

            if str(test_config.get('metric_type', 'continuous')) == 'ratio':
                ratio_components = self._extract_ratio_components(
                    user=user,
                    primary_metric=primary_metric,
                    ratio_value=float(metric_value),
                )
                if ratio_components is not None:
                    num, den = ratio_components
                    self.ratio_components_by_variant[variant]["numerators"].append(float(num))
                    self.ratio_components_by_variant[variant]["denominators"].append(float(den))
            
            if self.config.traffic_split_strategy == "adaptive":
                normalized_reward = self._normalize_reward_for_adaptive(
                    raw_value=metric_value,
                    metric_type=str(test_config.get('metric_type', 'continuous')),
                )
                self.traffic_splitter.update(variant, normalized_reward)
            
            if i % 500 == 0 or i == self.config.user_count:
                progress_pct = (i / self.config.user_count) * 100
                variant_stats_log = {
                    v: f"n={len(self.results_by_variant[v])}, mean={np.mean(self.results_by_variant[v]):.4f}"
                    for v in test_config['variants']
                }
                print(f"\n📈 Progress: {i}/{self.config.user_count} ({progress_pct:.1f}%)")
                print(f"   Variant Distribution: {variant_counts}")
                print(f"   Variant Stats: {variant_stats_log}")

            if i % self.time_series_interval == 0 or i == self.config.user_count:
                self._save_time_series_snapshot(i)

                self._update_runtime_progress(i)

            if self._should_check_progress(i):
                print(f"\n🔍 Performing Interim Analysis at {i} users...")
                interim_results = self._perform_interim_analysis(i)
                self._update_runtime_progress(
                    users_processed=i,
                    sequential_look=interim_results.get('sequential_look'),
                    srm_result=interim_results.get('srm_result'),
                )

                for variant, result in interim_results['variant_results'].items():
                    if result['can_stop_early']:
                        print(f"\n🎯 EARLY STOPPING TRIGGERED!")
                        print(f"   Variant: {variant}")
                        print(f"   Reason: {result['stop_reason']}")

                        self._early_stop_triggered = True
                        self._early_stop_reason = result['stop_reason']
                        self._sequential_look_at_stop = interim_results['sequential_look']

                        stop_simulation_early = True
                        break
                if stop_simulation_early:
                    break
            
            if delay_per_user > 0.001: 
                await asyncio.sleep(delay_per_user)
        
        print(f"\n{'='*80}")
        print(f"📊 FINAL ANALYSIS")
        print(f"{'='*80}")

        final_results = self._generate_final_report(test_config, start_time)
        final_results['users_processed'] = sum(len(v) for v in self.results_by_variant.values())

        await self._save_results_to_db(final_results)

        return final_results
    
    def _generate_final_report(
        self, 
        test_config: Dict[str, Any],
        start_time: datetime
    ) -> Dict[str, Any]:
        """Генерация финального отчета"""
        control_variant = test_config['variants'][0]
        control_data = np.array(self.results_by_variant[control_variant])
        
        variant_stats = {}
        for variant in test_config['variants']:
            data = np.array(self.results_by_variant[variant])
            variant_stats[variant] = {
                'sample_size': len(data),
                'mean': float(np.mean(data)) if len(data) > 0 else 0.0,
                'std': float(np.std(data, ddof=1)) if len(data) > 1 else 0.0,
                'median': float(np.median(data)) if len(data) > 0 else 0.0,
            }
        
        comparisons = {}
        for variant in test_config['variants'][1:]:
            treatment_data = np.array(self.results_by_variant[variant])
            
            if len(control_data) >= 30 and len(treatment_data) >= 30:
                full_analysis = run_full_ab_analysis(
                    control_data,
                    treatment_data,
                    baseline_std=variant_stats[control_variant]['std'],
                    mde_target=test_config['mde_percent'],
                    metric_type=str(test_config.get('metric_type', 'continuous')),
                    alpha=0.05
                )
                
                comparisons[variant] = {
                    'test_result': asdict(full_analysis['test_result']),
                    'power': full_analysis['power'],
                    'decision': full_analysis['decision'],
                    'confidence': full_analysis['confidence']
                }
        
        variant_counts = {v: len(self.results_by_variant[v]) for v in test_config['variants']}
        srm_final = self.srm_checker.check_srm_by_variant(variant_counts) if self.srm_checker else None
        
        traffic_stats = self.traffic_splitter.get_assignment_stats()
        
        end_time = datetime.utcnow()
        duration = (end_time - start_time).total_seconds() / 60
        
        return {
            'test_id': self.config.test_id,
            'dataset_id': self.config.dataset_id,
            'strategy': self.config.traffic_split_strategy,
            'completed_at': end_time.isoformat(),
            'duration_minutes': duration,
            'real_world_equivalent_days': self.config.real_world_duration_days,
            
            'variant_stats': variant_stats,
            'comparisons': comparisons,
            'srm_check': asdict(srm_final) if srm_final else None,
            'traffic_stats': traffic_stats,
            
            'sequential_testing': {
                'enabled': self.config.enable_sequential_testing,
                'looks_performed': self.sequential_tester.current_look if self.sequential_tester else 0,
                'max_looks': self.config.max_sequential_looks
            },
            
            'recommendation': self._generate_recommendation(comparisons, srm_final)
        }
    
    def _generate_recommendation(
        self,
        comparisons: Dict[str, Any],
        srm_result: Optional[Any]
    ) -> Dict[str, str]:
        """Генерирует рекомендацию на основе результатов"""
        if srm_result and srm_result.srm_detected:
            return {
                'decision': 'INVALID - SRM DETECTED',
                'reasoning': 'Sample Ratio Mismatch detected. Possible randomization bug. Do not trust results.',
                'confidence': 'NONE'
            }
        
        # Находим лучший вариант
        best_variant = None
        best_uplift = 0.0
        high_confidence = False
        
        for variant, comp in comparisons.items():
            test_result = comp['test_result']
            if test_result['significant'] and comp['power'] >= 0.8:
                uplift = test_result['relative_uplift_percent']
                if uplift > best_uplift:
                    best_variant = variant
                    best_uplift = uplift
                    high_confidence = True
        
        if best_variant:
            return {
                'decision': f'LAUNCH {best_variant}',
                'reasoning': f'Significant uplift: +{best_uplift:.1f}%, adequate power, SRM passed',
                'confidence': 'HIGH' if high_confidence else 'MEDIUM'
            }
        
        return {
            'decision': 'CONTINUE or STOP',
            'reasoning': 'No significant winner found. Consider extending test or stopping.',
            'confidence': 'LOW'
        }
    
    async def _save_results_to_db(self, results: Dict[str, Any]):
        """Асинхронное сохранение результатов в БД"""
        async with AsyncSessionLocal() as db:
            from sqlalchemy import select
            result = await db.execute(
                select(ABTestORM).filter(ABTestORM.test_id == self.config.test_id)
            )
            test = result.scalar_one_or_none()

            if test:
                # Обновляем статус
                test.status = 'completed'
                test.simulation_status = None  # Сбрасываем статус симуляции
                test.total_users = sum(
                    stats['sample_size']
                    for stats in results['variant_stats'].values()
                )
                processed_users = int(results.get('users_processed', 0))
                target_users = test.sample_size or self.config.user_count or processed_users or 1
                # После завершения симуляции тест должен отображаться как завершённый (100%),
                # даже если завершение было досрочным. Факт ранней остановки хранится отдельно.
                _ = target_users  # сохраняем вычисление для обратной совместимости логики
                test.completion_percentage = 100.0 if processed_users > 0 else 0.0


                if results['srm_check']:
                    test.srm_check_passed = 0 if results['srm_check']['srm_detected'] else 1
                    test.srm_p_value = results['srm_check']['p_value']

                if results['sequential_testing']['enabled']:
                    test.current_sequential_look = results['sequential_testing']['looks_performed']

                    if self._early_stop_triggered:
                        test.stopped_early = 1
                        test.early_stop_reason = self._early_stop_reason

                await db.commit()
                print(f"\n Results saved to database")

            print(f" Time series data already saved in real-time: {len(self.time_series_data)} snapshots")


async def run_ab_test_simulation(
    test_id: str,
    dataset_id: int,
    user_count: int = 1000,
    real_world_days: int = 14,
    simulation_minutes: Optional[int] = None,
    strategy: str = "fixed",
    variant_effects: Optional[Dict[str, Dict[str, float]]] = None
) -> Dict[str, Any]:
    config = SimulationConfig(
        test_id=test_id,
        dataset_id=dataset_id,
        user_count=user_count,
        real_world_duration_days=real_world_days,
        simulation_duration_minutes=simulation_minutes,
        traffic_split_strategy=strategy,
        enable_sequential_testing=True,
        enable_srm_check=True,
        enable_early_stopping=False,
        variant_effects=variant_effects
    )
    
    simulator = GoogleStandardABTestSimulator(config)
    results = await simulator.run_simulation()
    
    return results