# backend/ab_testing/statistics.py
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from scipy import stats
from scipy.optimize import brentq


@dataclass
class StatisticalTestResult:
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
    srm_detected: bool
    p_value: float
    chi2_statistic: float
    expected: List[int]
    observed: List[int]
    warning: Optional[str]


@dataclass
class PowerAnalysisResult:
    power: float
    sample_size_per_variant: int
    mde_absolute: float
    mde_percent: float
    alpha: float
    baseline_mean: float


class SampleSizeCalculator:
    
    @staticmethod
    def calculate_sample_size(
        baseline_mean: float,
        baseline_std: float,
        mde_percent: float,
        alpha: float = 0.05,
        power: float = 0.8,
        two_tailed: bool = True
    ) -> int:

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
    
    def __init__(self, alpha: float = 0.05, max_looks: int = 5):
        self.alpha = alpha
        self.max_looks = max_looks
        self.boundaries = self._calculate_obrien_fleming_boundaries()
        self.current_look = 0
        self.looks_history: List[Dict[str, Any]] = []
    
    def _calculate_obrien_fleming_boundaries(self) -> List[float]:
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
        target_sample_size = max(1, int(target_sample_size))
        progress = observations / target_sample_size

        # До 85% выборки не останавливаемся по бесперспективности,
        # иначе высок риск преждевременной остановки на шуме.
        if progress < 0.85:
            return False, "Too early for futility check"

        abs_mde = abs(float(target_mde))
        abs_effect = abs(float(observed_effect))

        # Если пользователь поставил нулевой MDE, фьютили-ти критерий неприменим.
        if abs_mde <= 1e-12:
            return False, "Continue experiment"

        # Останавливаем только когда наблюдаемый эффект существенно ниже MDE
        # на высокой доле собранных данных.
        if abs_effect < abs_mde * 0.25:
            return True, f"Observed effect ({observed_effect:.1%}) << target MDE ({target_mde:.1%})"

        return False, "Continue experiment"
    
    def get_adjusted_alpha(self) -> float:
        """Возвращает скорректированный alpha для текущей проверки"""
        if self.current_look >= self.max_looks:
            return self.alpha
        return self.boundaries[self.current_look]


class SRMChecker:

    
    @staticmethod
    def check_srm(
        expected_ratios: List[float],
        observed_counts: List[int],
        alpha: float = 0.001  # Очень консервативный порог
    ) -> SRMCheckResult:
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

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha

    def _bootstrap_mean_diff_ci(
        self,
        control: np.ndarray,
        treatment: np.ndarray,
        iterations: int = 4000,
        ci_level: float = 0.95,
    ) -> Tuple[float, float]:
        if len(control) == 0 or len(treatment) == 0:
            return 0.0, 0.0

        rng = np.random.default_rng(42)
        diffs = np.empty(iterations, dtype=float)

        for i in range(iterations):
            c = rng.choice(control, size=len(control), replace=True)
            t = rng.choice(treatment, size=len(treatment), replace=True)
            diffs[i] = float(np.mean(t) - np.mean(c))

        alpha_tail = (1.0 - ci_level) / 2.0
        low = float(np.quantile(diffs, alpha_tail))
        high = float(np.quantile(diffs, 1.0 - alpha_tail))
        return low, high

    def analyze_continuous_metric(
        self,
        control: np.ndarray,
        treatment: np.ndarray,
        num_comparisons: int = 1,
        correction_method: str = "bonferroni"
    ) -> StatisticalTestResult:
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
    
    def analyze_ratio_metric(
        self,
        control_numerators: np.ndarray,
        control_denominators: np.ndarray,
        treatment_numerators: np.ndarray,
        treatment_denominators: np.ndarray,
        num_comparisons: int = 1,
    ) -> StatisticalTestResult:
        control_denominators = np.where(control_denominators <= 0, np.nan, control_denominators)
        treatment_denominators = np.where(treatment_denominators <= 0, np.nan, treatment_denominators)

        control_ratio = control_numerators / control_denominators
        treatment_ratio = treatment_numerators / treatment_denominators

        control_ratio = control_ratio[np.isfinite(control_ratio)]
        treatment_ratio = treatment_ratio[np.isfinite(treatment_ratio)]

        if len(control_ratio) < 10 or len(treatment_ratio) < 10:
            return StatisticalTestResult(
                t_statistic=0.0,
                p_value=1.0,
                p_value_corrected=1.0,
                significant=False,
                effect_size_cohens_d=0.0,
                confidence_interval=(0.0, 0.0),
                relative_uplift_percent=0.0,
                standard_error=0.0,
            )

        t_stat, p_value = stats.ttest_ind(treatment_ratio, control_ratio, equal_var=False)
        p_value_corrected = min(float(p_value) * max(1, int(num_comparisons)), 1.0)

        control_mean = float(np.mean(control_ratio))
        treatment_mean = float(np.mean(treatment_ratio))
        diff = treatment_mean - control_mean

        pooled_std = np.sqrt((np.var(control_ratio, ddof=1) + np.var(treatment_ratio, ddof=1)) / 2.0)
        cohens_d = float(diff / pooled_std) if pooled_std > 1e-12 else 0.0

        ci_low, ci_high = self._bootstrap_mean_diff_ci(control_ratio, treatment_ratio)

        se = np.sqrt(
            np.var(control_ratio, ddof=1) / len(control_ratio)
            + np.var(treatment_ratio, ddof=1) / len(treatment_ratio)
        )

        relative_uplift = (diff / control_mean * 100.0) if abs(control_mean) > 1e-12 else 0.0

        return StatisticalTestResult(
            t_statistic=float(t_stat),
            p_value=float(p_value),
            p_value_corrected=float(p_value_corrected),
            significant=p_value_corrected < self.alpha,
            effect_size_cohens_d=cohens_d,
            confidence_interval=(float(ci_low), float(ci_high)),
            relative_uplift_percent=float(relative_uplift),
            standard_error=float(se),
        )

    def calculate_power(
        self,
        observed_effect: float,
        sample_size_per_variant: int,
        baseline_std: float,
        alpha: float = 0.05
    ) -> float:
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
    
    @staticmethod
    def apply_cuped(
        post_metric: np.ndarray,
        pre_metric: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, float]]:
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
    analyzer = StatisticalAnalyzer(alpha=alpha)
    
    # 1. Основной статистический тест
    if metric_type == "continuous":
        test_result = analyzer.analyze_continuous_metric(control_data, treatment_data)
    elif metric_type == "ratio":
        # Для ratio-метрик ожидаем, что вход уже является отношением по пользователям.
        # Используем ratio-анализ с unit-denominator=1 как безопасный фолбэк.
        test_result = analyzer.analyze_ratio_metric(
            control_numerators=control_data,
            control_denominators=np.ones_like(control_data),
            treatment_numerators=treatment_data,
            treatment_denominators=np.ones_like(treatment_data),
        )
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
