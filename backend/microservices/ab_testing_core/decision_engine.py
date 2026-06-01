from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class VariantDecision:
    variant: str
    p_value_raw: Optional[float]
    p_value_corrected: Optional[float]
    significant: bool
    uplift_percent_vs_control: float
    sample_size: int


@dataclass
class GuardrailCheck:
    metric: str
    threshold: float
    direction: str
    observed: float
    passed: bool


@dataclass
class DecisionPolicyResult:
    allowed: bool
    status: str
    reasons: List[str]
    checks: Dict[str, Any]


class ABDecisionEngine:
    """Единый движок принятия решений для A/B тестов."""

    @staticmethod
    def holm_bonferroni_correction(p_values: Dict[str, float]) -> Dict[str, float]:
        items = sorted([(k, float(v)) for k, v in p_values.items()], key=lambda kv: kv[1])
        m = len(items)
        corrected: Dict[str, float] = {}

        prev = 0.0
        for idx, (variant, p) in enumerate(items):
            multiplier = max(1, m - idx)
            adjusted = min(1.0, p * multiplier)
            adjusted = max(prev, adjusted)  # monotonicity
            corrected[variant] = adjusted
            prev = adjusted

        return corrected

    @staticmethod
    def resolve_analysis_validity(
        *,
        analysis_mode: str,
        traffic_split_type: str,
        srm_detected: Optional[bool],
        guardrails_failed: bool,
    ) -> str:
        if srm_detected is True:
            return "invalid_srm"
        if guardrails_failed:
            return "invalid_guardrails"
        if analysis_mode == "adaptive_bandit" or traffic_split_type == "adaptive":
            return "exploration_only"
        return "valid_for_inference"

    @staticmethod
    def evaluate_guardrails(
        *,
        guardrails_config: Optional[Dict[str, Any]],
        variant_means: Dict[str, float],
        control_variant: str,
        winner_variant: Optional[str],
    ) -> Dict[str, Any]:
        if not guardrails_config or not winner_variant or winner_variant == control_variant:
            return {
                "enabled": bool(guardrails_config),
                "checks": [],
                "passed": True,
                "failed_metrics": [],
            }

        checks: List[GuardrailCheck] = []
        failed: List[str] = []

        for metric_name, cfg in guardrails_config.items():
            threshold = float(cfg.get("threshold", 0.0))
            direction = str(cfg.get("direction", "max_increase"))

            control_value = float(cfg.get("control_value", variant_means.get(control_variant, 0.0)))
            winner_value = float(cfg.get("winner_value", variant_means.get(winner_variant, 0.0)))

            if abs(control_value) <= 1e-12:
                observed_change = 0.0
            else:
                observed_change = (winner_value - control_value) / abs(control_value) * 100.0

            passed = True
            if direction == "max_increase":
                passed = observed_change <= threshold
            elif direction == "max_decrease":
                passed = observed_change >= -threshold
            elif direction == "min_increase":
                passed = observed_change >= threshold
            elif direction == "min_decrease":
                passed = observed_change <= -threshold

            if not passed:
                failed.append(metric_name)

            checks.append(
                GuardrailCheck(
                    metric=metric_name,
                    threshold=threshold,
                    direction=direction,
                    observed=observed_change,
                    passed=passed,
                )
            )

        return {
            "enabled": True,
            "checks": [asdict(c) for c in checks],
            "passed": len(failed) == 0,
            "failed_metrics": failed,
        }

    @staticmethod
    def evaluate_decision_policy(
        *,
        analysis_validity: str,
        srm_passed: Optional[bool],
        guardrails_passed: bool,
        corrected_p_values: Dict[str, float],
        power: Optional[float],
        alpha: float,
    ) -> DecisionPolicyResult:
        reasons: List[str] = []
        checks: Dict[str, Any] = {}

        valid_design = analysis_validity == "valid_for_inference"
        checks["valid_design"] = valid_design
        if not valid_design:
            reasons.append("Невалидный дизайн эксперимента для итогового вывода")

        srm_ok = (srm_passed is True) if srm_passed is not None else False
        checks["srm_passed"] = srm_ok
        if srm_passed is None:
            reasons.append("SRM не проверен")
        elif not srm_ok:
            reasons.append("SRM не пройден")

        checks["guardrails_passed"] = bool(guardrails_passed)
        if not guardrails_passed:
            reasons.append("Guardrails нарушены")

        p_ok = any(float(p) < float(alpha) for p in corrected_p_values.values()) if corrected_p_values else False
        checks["corrected_p_below_alpha"] = p_ok
        if not p_ok:
            reasons.append("Нет статистически значимых различий по скорректированным p-value")

        power_ok = power is not None and float(power) >= 0.8
        checks["power_ok"] = power_ok
        if power is None:
            reasons.append("Power не рассчитан")
        elif not power_ok:
            reasons.append("Недостаточная статистическая мощность (< 0.8)")

        allowed = all([valid_design, srm_ok, guardrails_passed, p_ok, power_ok])
        status = "deploy" if allowed else "hold"
        return DecisionPolicyResult(allowed=allowed, status=status, reasons=reasons, checks=checks)

    @staticmethod
    def build_decision_summary(
        *,
        results: Dict[str, Any],
        p_values_raw: Dict[str, float],
        corrected_p_values: Dict[str, float],
        alpha: float,
        analysis_validity: str,
        guardrails_status: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not results:
            return {
                "best_variant": None,
                "improvement_percentage": 0.0,
                "recommended_action": "Нет данных",
                "confidence_level": "low",
                "significant_variants": [],
                "control_variant": None,
                "p_values": {},
                "p_values_corrected": {},
                "analysis_validity": analysis_validity,
                "guardrails": guardrails_status,
                "variant_decisions": [],
            }

        control_variant = list(results.keys())[0]
        control_result = results[control_variant]
        control_mean = float(control_result.mean)

        significant_variants: List[str] = []
        best_variant: Optional[str] = control_variant
        best_uplift = 0.0
        variant_cards: List[VariantDecision] = []

        for variant, stat in results.items():
            if variant == control_variant:
                continue

            uplift = ((float(stat.mean) - control_mean) / control_mean * 100.0) if abs(control_mean) > 1e-12 else 0.0
            p_raw = p_values_raw.get(variant)
            p_corr = corrected_p_values.get(variant)
            significant = p_corr is not None and p_corr < alpha

            if significant:
                significant_variants.append(variant)
                if uplift > best_uplift:
                    best_uplift = uplift
                    best_variant = variant

            variant_cards.append(
                VariantDecision(
                    variant=variant,
                    p_value_raw=p_raw,
                    p_value_corrected=p_corr,
                    significant=significant,
                    uplift_percent_vs_control=uplift,
                    sample_size=int(stat.sample_size),
                )
            )

        if analysis_validity != "valid_for_inference":
            best_variant = None
            best_uplift = 0.0
            significant_variants = []
            recommended_action = "Использовать только как исследовательский сигнал; финальный fixed A/B обязателен"
            confidence = "low"
        elif guardrails_status.get("enabled") and not guardrails_status.get("passed", True):
            recommended_action = "Победитель заблокирован guardrails — не внедрять"
            confidence = "high"
        elif best_variant != control_variant and best_uplift > 0:
            recommended_action = f"Внедрить вариант {best_variant}"
            confidence = "high"
        else:
            recommended_action = "Оставить контроль / продолжить сбор данных"
            confidence = "medium" if significant_variants else "low"

        return {
            "best_variant": best_variant,
            "improvement_percentage": float(best_uplift),
            "recommended_action": recommended_action,
            "confidence_level": confidence,
            "significant_variants": significant_variants,
            "control_variant": control_variant,
            "p_values": p_values_raw,
            "p_values_corrected": corrected_p_values,
            "analysis_validity": analysis_validity,
            "guardrails": guardrails_status,
            "variant_decisions": [asdict(v) for v in variant_cards],
        }

    @staticmethod
    def infer_variant_means(results: Dict[str, Any]) -> Dict[str, float]:
        return {variant: float(stat.mean) for variant, stat in results.items()}

    @staticmethod
    def estimate_guardrail_from_series(values: List[float]) -> float:
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return 0.0
        return float(arr.mean())
