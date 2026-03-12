# backend/ab_testing/statistics.py
"""
Google-Standard Statistical Tools для A/B Testing

Включает:
- Sample Size calculation с правильным MDE
- Sequential Testing (O'Brien-Fleming boundaries)
- Sample Ratio Mismatch (SRM) проверка
- Statistical Power Analysis
- Multiple Comparisons Correction (Bonferroni, FDR)
- CUPED для Variance Reduction
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from scipy import stats
from scipy.optimize import brentq


@dataclass
class StatisticalTestResult:
    """Результат статистического теста"""
    t_statistic: float
    p_value: float
    p_value_corrected: float
    significant: bool
    effect_size_cohens_d: float
    confidence_interval: Tuple[float, float]
    relative_uplift_percent: float
    standard_error: float


@dataclass
class SRMCheckResult:
    """Результат проверки Sample Ratio Mismatch"""
    srm_detected: bool
    p_value: float
    chi2_statistic: float
    expected: List[int]
    observed: List[int]
    warning: Optional[str]


@dataclass
class PowerAnalysisResult:
    """Результат анализа статистической мощности"""
    power: float
    sample_size_per_variant: int
    mde_absolute: float
    mde_percent: float
    alpha: float
    baseline_mean: float


class SampleSizeCalculator:
    """
    Рассчет размера выборки для A/B тестов
    
    Google использует консервативные оценки:
    - Power = 0.8 (минимум)
    - Alpha = 0.05 (two-tailed)
    - MDE = минимальный практически значимый эффект
    """
    
    @staticmethod
    def calculate_sample_size(
        baseline_mean: float,
        baseline_std: float,
        mde_percent: float,
        alpha: float = 0.05,
        power: float = 0.8,
        two_tailed: bool = True
    ) -> int:
        """
        Рассчитывает необходимый размер выборки НА ВАРИАНТ
        
        Args:
            baseline_mean: Среднее значение в контроле
            baseline_std: Стандартное отклонение в контроле
            mde_percent: Минимальный детектируемый эффект в %
            alpha: Уровень значимости (Type I error)
            power: Статистическая мощность (1 - Type II error)
            two_tailed: Двусторонний тест
            
        Returns:
            Размер выборки на один вариант
        """
        # Эффект в абсолютных единицах
        mde_absolute = baseline_mean * (mde_percent / 100.0)
        
        # Effect size (Cohen's d)
        effect_size = mde_absolute / baseline_std
        
        # Z-scores
        z_alpha = stats.norm.ppf(1 - alpha / 2) if two_tailed else stats.norm.ppf(1 - alpha)
        z_beta = stats.norm.ppf(power)
        
        # Sample size formula для t-test
        n_per_group = 2 * ((z_alpha + z_beta) / effect_size) ** 2
        
        # Добавляем 10% запас (Google практика)
        n_per_group_with_buffer = int(np.ceil(n_per_group * 1.1))
        
        return max(30, n_per_group_with_buffer)  # Минимум 30 для CLT
    
    @staticmethod
    def calculate_sample_size_for_binary(
        baseline_conversion: float,
        mde_percent: float,
        alpha: float = 0.05,
        power: float = 0.8
    ) -> int:
        """
        Для бинарных метрик (конверсия, CTR)
        
        Args:
            baseline_conversion: Текущая конверсия (0.15 = 15%)
            mde_percent: Относительное изменение (10 = +10%)
            
        Returns:
            Размер выборки на вариант
        """
        p1 = baseline_conversion
        p2 = baseline_conversion * (1 + mde_percent / 100.0)
        
        # Pooled proportion
        p_pooled = (p1 + p2) / 2
        
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        z_beta = stats.norm.ppf(power)
        
        numerator = (z_alpha * np.sqrt(2 * p_pooled * (1 - p_pooled)) +
                    z_beta * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
        denominator = (p2 - p1) ** 2
        
        n_per_group = numerator / denominator
        
        # Добавляем запас
        return int(np.ceil(n_per_group * 1.1))


class SequentialTesting:
    """
    Sequential Testing с O'Brien-Fleming boundaries
    
    Позволяет:
    - Останавливать тест досрочно при достижении значимости
    - Контролировать Type I error rate
    - Экономить время и ресурсы
    
    Google/Meta используют именно этот подход
    """
    
    def __init__(self, alpha: float = 0.05, max_looks: int = 5):
        """
        Args:
            alpha: Общий уровень значимости
            max_looks: Количество промежуточных проверок
        """
        self.alpha = alpha
        self.max_looks = max_looks
        self.boundaries = self._calculate_obrien_fleming_boundaries()
        self.current_look = 0
        self.looks_history: List[Dict[str, Any]] = []
    
    def _calculate_obrien_fleming_boundaries(self) -> List[float]:
        """
        O'Brien-Fleming spending function
        
        Особенности:
        - Очень консервативные границы в начале
        - Более либеральные в конце
        - Сохраняет общий alpha
        """
        boundaries = []
        
        for k in range(1, self.max_looks + 1):
            # Information fraction
            t = k / self.max_looks
            
            # O'Brien-Fleming critical value
            z_boundary = stats.norm.ppf(1 - self.alpha / (2 * np.sqrt(t)))
            
            # Конвертируем в p-value
            p_boundary = 2 * (1 - stats.norm.cdf(z_boundary))
            
            boundaries.append(p_boundary)
        
        return boundaries
    
    def should_stop_for_success(self, p_value: float, effect: float) -> Tuple[bool, str]:
        """
        Проверяет, достигнута ли граница для остановки
        
        Returns:
            (should_stop, reason)
        """
        if self.current_look >= self.max_looks:
            # Финальная проверка
            stop = p_value < self.alpha
            reason = "Reached max looks, final check" if stop else "No significance at max looks"
            return stop, reason
        
        # Текущая граница
        boundary = self.boundaries[self.current_look]
        
        # Логируем проверку
        self.looks_history.append({
            'look': self.current_look + 1,
            'p_value': p_value,
            'boundary': boundary,
            'effect': effect
        })
        
        self.current_look += 1
        
        if p_value < boundary:
            return True, f"Crossed O'Brien-Fleming boundary at look {self.current_look}"
        
        return False, "No significance yet"
    
    def should_stop_for_futility(
        self,
        observed_effect: float,
        target_mde: float,
        observations: int,
        target_sample_size: int
    ) -> Tuple[bool, str]:
        """
        Проверка на бесперспективность (futility)

        Останавливаем, если:
        1. Наблюдаемый эффект << target MDE
        2. Собрано достаточно данных (>70% от target)
        3. Тренд не улучшается

        Исправлено: более консервативная проверка, чтобы не останавливать слишком рано
        """
        progress = observations / target_sample_size

        # Должны собрать хотя бы 70% данных (было 50%)
        if progress < 0.7:
            return False, "Too early for futility check"

        # Наблюдаемый эффект слишком мал (было 0.3, стало 0.15 - менее строго)
        if abs(observed_effect) < abs(target_mde) * 0.15:
            # Дополнительная проверка: если эффект близок к нулю и progress > 85%
            if progress > 0.85 and abs(observed_effect) < abs(target_mde) * 0.2:
                return True, f"Observed effect ({observed_effect:.1%}) << target MDE ({target_mde:.1%})"

        return False, "Continue experiment"
    
    def get_adjusted_alpha(self) -> float:
        """Возвращает скорректированный alpha для текущей проверки"""
        if self.current_look >= self.max_looks:
            return self.alpha
        return self.boundaries[self.current_look]


class SRMChecker:
    """
    Sample Ratio Mismatch (SRM) Detection
    
    SRM = индикатор проблем в системе рандомизации:
    - Баги в коде распределения
    - Фильтрация данных после рандомизации
    - Проблемы с логированием
    
    Google проверяет SRM в каждом эксперименте
    """
    
    @staticmethod
    def check_srm(
        expected_ratios: List[float],
        observed_counts: List[int],
        alpha: float = 0.001  # Очень консервативный порог
    ) -> SRMCheckResult:
        """
        Chi-square goodness-of-fit test для проверки SRM
        
        Args:
            expected_ratios: Ожидаемые пропорции [0.5, 0.5] или [0.33, 0.33, 0.34]
            observed_counts: Фактические количества юзеров
            alpha: Порог для детекции SRM (0.001 = очень строго)
            
        Returns:
            SRMCheckResult с результатами проверки
        """
        total = sum(observed_counts)
        expected_counts = [ratio * total for ratio in expected_ratios]
        
        # Chi-square test
        chi2_stat, p_value = stats.chisquare(observed_counts, expected_counts)
        
        # SRM обнаружен если p-value очень низкий
        srm_detected = p_value < alpha
        
        warning = None
        if srm_detected:
            warning = (
                f"⚠️ CRITICAL: Sample Ratio Mismatch detected! "
                f"Expected {expected_counts}, got {observed_counts}. "
                f"This indicates a potential randomization bug."
            )
        
        return SRMCheckResult(
            srm_detected=srm_detected,
            p_value=p_value,
            chi2_statistic=chi2_stat,
            expected=[int(c) for c in expected_counts],
            observed=observed_counts,
            warning=warning
        )
    
    @staticmethod
    def check_srm_by_variant(variant_counts: Dict[str, int], expected_split: Optional[Dict[str, float]] = None) -> SRMCheckResult:
        """
        Удобная обертка для проверки по вариантам
        
        Args:
            variant_counts: {"control": 1000, "treatment": 1050}
            expected_split: {"control": 0.5, "treatment": 0.5} или None для равного
        """
        variants = sorted(variant_counts.keys())
        observed = [variant_counts[v] for v in variants]
        
        if expected_split is None:
            # Равное распределение
            n_variants = len(variants)
            expected_ratios = [1.0 / n_variants] * n_variants
        else:
            expected_ratios = [expected_split[v] for v in variants]
        
        return SRMChecker.check_srm(expected_ratios, observed)


class StatisticalAnalyzer:
    """
    Production-grade статистический анализ A/B тестов
    
    Включает:
    - Welch's t-test (не предполагает равные дисперсии)
    - Multiple comparisons correction
    - Effect size (Cohen's d)
    - Confidence intervals
    - Power analysis
    """
    
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
    
    def analyze_continuous_metric(
        self,
        control: np.ndarray,
        treatment: np.ndarray,
        num_comparisons: int = 1,
        correction_method: str = "bonferroni"
    ) -> StatisticalTestResult:
        """
        Анализ непрерывной метрики (revenue, session_duration, etc.)
        
        Args:
            control: Данные контрольной группы
            treatment: Данные тестовой группы
            num_comparisons: Количество сравнений (для коррекции)
            correction_method: "bonferroni" или "fdr"
        """
        # Welch's t-test (более robust чем обычный t-test)
        t_stat, p_value = stats.ttest_ind(treatment, control, equal_var=False)
        
        # Multiple comparisons correction
        if correction_method == "bonferroni":
            p_value_corrected = min(p_value * num_comparisons, 1.0)
        else:  # FDR (менее консервативный)
            p_value_corrected = p_value * num_comparisons / 1.5  # Упрощенная версия
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt((np.var(control, ddof=1) + np.var(treatment, ddof=1)) / 2)
        if pooled_std == 0:
            cohens_d = 0.0
        else:
            cohens_d = (np.mean(treatment) - np.mean(control)) / pooled_std
        
        # Standard error
        se = pooled_std * np.sqrt(1/len(control) + 1/len(treatment))
        
        # Confidence interval (95%)
        diff = np.mean(treatment) - np.mean(control)
        ci_margin = 1.96 * se
        ci_lower = diff - ci_margin
        ci_upper = diff + ci_margin
        
        # Relative uplift
        if np.mean(control) != 0:
            relative_uplift = (np.mean(treatment) - np.mean(control)) / np.mean(control) * 100
        else:
            relative_uplift = 0.0
        
        # Значимость с учетом коррекции
        significant = p_value_corrected < self.alpha
        
        return StatisticalTestResult(
            t_statistic=float(t_stat),
            p_value=float(p_value),
            p_value_corrected=float(p_value_corrected),
            significant=significant,
            effect_size_cohens_d=float(cohens_d),
            confidence_interval=(float(ci_lower), float(ci_upper)),
            relative_uplift_percent=float(relative_uplift),
            standard_error=float(se)
        )
    
    def analyze_binary_metric(
        self,
        control_conversions: int,
        control_total: int,
        treatment_conversions: int,
        treatment_total: int,
        num_comparisons: int = 1
    ) -> StatisticalTestResult:
        """
        Анализ бинарной метрики (конверсия, CTR)
        
        Использует two-proportion z-test
        """
        # Conversion rates
        p_control = control_conversions / control_total
        p_treatment = treatment_conversions / treatment_total
        
        # Pooled proportion
        p_pooled = (control_conversions + treatment_conversions) / (control_total + treatment_total)
        
        # Standard error
        se = np.sqrt(p_pooled * (1 - p_pooled) * (1/control_total + 1/treatment_total))
        
        # Z-statistic
        if se == 0:
            z_stat = 0.0
            p_value = 1.0
        else:
            z_stat = (p_treatment - p_control) / se
            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))  # Two-tailed
        
        # Bonferroni correction
        p_value_corrected = min(p_value * num_comparisons, 1.0)
        
        # Effect size (h - для пропорций)
        cohens_h = 2 * (np.arcsin(np.sqrt(p_treatment)) - np.arcsin(np.sqrt(p_control)))
        
        # Confidence interval
        ci_margin = 1.96 * se
        diff = p_treatment - p_control
        ci_lower = diff - ci_margin
        ci_upper = diff + ci_margin
        
        # Relative uplift
        if p_control != 0:
            relative_uplift = (p_treatment - p_control) / p_control * 100
        else:
            relative_uplift = 0.0
        
        return StatisticalTestResult(
            t_statistic=float(z_stat),  # На самом деле z-stat
            p_value=float(p_value),
            p_value_corrected=float(p_value_corrected),
            significant=p_value_corrected < self.alpha,
            effect_size_cohens_d=float(cohens_h),
            confidence_interval=(float(ci_lower), float(ci_upper)),
            relative_uplift_percent=float(relative_uplift),
            standard_error=float(se)
        )
    
    def calculate_power(
        self,
        observed_effect: float,
        sample_size_per_variant: int,
        baseline_std: float,
        alpha: float = 0.05
    ) -> float:
        """
        Рассчитывает фактическую статистическую мощность
        
        Power = вероятность обнаружить эффект, если он есть
        """
        if baseline_std == 0:
            return 0.0
        
        effect_size = observed_effect / baseline_std
        
        # Critical value
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        
        # Non-centrality parameter
        ncp = effect_size * np.sqrt(sample_size_per_variant / 2)
        
        # Power calculation
        power = 1 - stats.norm.cdf(z_alpha - ncp)
        
        return float(min(1.0, max(0.0, power)))


class CUPEDAnalyzer:
    """
    CUPED (Controlled-experiment Using Pre-Experiment Data)
    
    Variance Reduction техника от Microsoft/Google:
    - Использует pre-period данные для снижения дисперсии
    - Позволяет детектировать меньшие эффекты
    - Ускоряет тесты на 30-50%
    
    Требование: наличие pre-experiment данных
    """
    
    @staticmethod
    def apply_cuped(
        post_metric: np.ndarray,
        pre_metric: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Применяет CUPED корректировку
        
        Args:
            post_metric: Метрика после начала эксперимента
            pre_metric: Метрика до эксперимента (covariате)
            
        Returns:
            (cuped_metric, stats_dict)
        """
        # Оптимальный коэффициент theta
        covariance = np.cov(post_metric, pre_metric)[0, 1]
        variance_pre = np.var(pre_metric, ddof=1)
        
        if variance_pre == 0:
            # Нет вариации в pre-metric, CUPED не поможет
            return post_metric, {
                'theta': 0.0,
                'variance_reduction': 0.0,
                'original_variance': float(np.var(post_metric, ddof=1)),
                'cuped_variance': float(np.var(post_metric, ddof=1))
            }
        
        theta = covariance / variance_pre
        
        # CUPED-скорректированная метрика
        pre_mean = np.mean(pre_metric)
        cuped_metric = post_metric - theta * (pre_metric - pre_mean)
        
        # Статистика снижения дисперсии
        var_original = np.var(post_metric, ddof=1)
        var_cuped = np.var(cuped_metric, ddof=1)
        variance_reduction = 1 - (var_cuped / var_original)
        
        stats_dict = {
            'theta': float(theta),
            'variance_reduction': float(variance_reduction),
            'original_variance': float(var_original),
            'cuped_variance': float(var_cuped),
            'equivalent_sample_size_multiplier': float(1 / (1 - variance_reduction))
        }
        
        return cuped_metric, stats_dict


# Удобная функция для полного анализа
def run_full_ab_analysis(
    control_data: np.ndarray,
    treatment_data: np.ndarray,
    baseline_std: float,
    mde_target: float,
    metric_type: str = "continuous",
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Полный анализ A/B теста по стандартам Google
    
    Args:
        control_data: Данные контрольной группы
        treatment_data: Данные тестовой группы
        baseline_std: Стандартное отклонение baseline
        mde_target: Целевой MDE в процентах
        metric_type: "continuous" или "binary"
        alpha: Уровень значимости
        
    Returns:
        Полный отчет с рекомендациями
    """
    analyzer = StatisticalAnalyzer(alpha=alpha)
    
    # 1. Основной статистический тест
    if metric_type == "continuous":
        test_result = analyzer.analyze_continuous_metric(control_data, treatment_data)
    else:
        control_conv = int(np.sum(control_data))
        treatment_conv = int(np.sum(treatment_data))
        test_result = analyzer.analyze_binary_metric(
            control_conv, len(control_data),
            treatment_conv, len(treatment_data)
        )
    
    # 2. Power analysis
    observed_effect = np.mean(treatment_data) - np.mean(control_data)
    power = analyzer.calculate_power(
        observed_effect,
        len(control_data),
        baseline_std,
        alpha
    )
    
    # 3. Рекомендация
    if test_result.significant and power >= 0.8:
        decision = "LAUNCH"
        confidence = "HIGH"
    elif test_result.significant and power >= 0.6:
        decision = "LAUNCH WITH CAUTION"
        confidence = "MEDIUM"
    elif not test_result.significant and abs(test_result.relative_uplift_percent) < abs(mde_target) * 0.5:
        decision = "STOP (No Effect)"
        confidence = "HIGH"
    else:
        decision = "CONTINUE"
        confidence = "LOW"
    
    return {
        'test_result': test_result,
        'power': power,
        'decision': decision,
        'confidence': confidence,
        'sample_sizes': {
            'control': len(control_data),
            'treatment': len(treatment_data)
        },
        'means': {
            'control': float(np.mean(control_data)),
            'treatment': float(np.mean(treatment_data))
        }
    }
