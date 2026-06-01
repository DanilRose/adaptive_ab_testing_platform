from __future__ import annotations

from typing import Any, Dict, List, Optional
from datetime import datetime
import math

import numpy as np
import scipy.stats as stats
from fastapi import HTTPException

from backend.microservices.ab_testing_core.decision_engine import ABDecisionEngine
from backend.microservices.ab_testing_core.statistics import StatisticalAnalyzer
from backend.microservices.database.session import SessionLocal
from backend.microservices.database import crud
from backend.microservices.database.models import ABTestORM, TestSessionORM


class ResultsAnalyticsService:
    @staticmethod
    def get_statistical_significance(*, test_id: str, alpha: float, platform: Any) -> Dict[str, Any]:
        try:
            db_results = platform.get_test_results(test_id)
            results = db_results.get("results", {})
            p_values = db_results.get("statistical_significance", {})

            if not results:
                return {
                    "test_id": test_id,
                    "alpha_level": alpha,
                    "significance_analysis": {},
                    "interpretation": ResultsAnalyticsService._interpret_significance({}, alpha),
                }

            significance_analysis = {}
            control_variant = list(results.keys())[0]
            control_data = ResultsAnalyticsService._to_stat_object(results[control_variant])

            for variant, p_value in p_values.items():
                variant_payload = results.get(variant)
                if not variant_payload:
                    continue

                variant_data = ResultsAnalyticsService._to_stat_object(variant_payload)

                power = ResultsAnalyticsService._calculate_power(control_data, variant_data, alpha)
                effect_size = float(variant_data.mean - control_data.mean)
                effect_ci = ResultsAnalyticsService._calculate_effect_confidence_interval(control_data, variant_data)

                significance_analysis[variant] = {
                    "p_value": float(p_value),
                    "statistically_significant": float(p_value) < alpha,
                    "power": power,
                    "effect_size": effect_size,
                    "effect_confidence_interval": effect_ci,
                    "required_sample_size": ResultsAnalyticsService._calculate_required_sample_size(
                        control_data, variant_data, alpha, 0.8
                    ),
                }

            return {
                "test_id": test_id,
                "alpha_level": alpha,
                "significance_analysis": significance_analysis,
                "interpretation": ResultsAnalyticsService._interpret_significance(significance_analysis, alpha),
            }
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @staticmethod
    def get_time_series_chart_data(*, test_id: str, platform: Any) -> Dict[str, Any]:
        try:
            with SessionLocal() as db:
                time_series_records = crud.get_ab_test_time_series(db, test_id, limit=5000)
                test = db.query(ABTestORM).filter(ABTestORM.test_id == test_id).first()

            if not test:
                raise HTTPException(status_code=404, detail="Тест не найден")

            try:
                db_centric_results = platform.get_test_results(test_id)
                quality_gate = db_centric_results.get("quality_gate") or {
                    "status": "yellow",
                    "passed": False,
                    "passed_checks": 0,
                    "total_checks": 5,
                    "checks": [],
                }
                decision_policy = db_centric_results.get("decision_policy") or None
            except Exception:
                quality_gate = {
                    "status": "yellow",
                    "passed": False,
                    "passed_checks": 0,
                    "total_checks": 5,
                    "checks": [],
                }
                decision_policy = None

            if not time_series_records:
                return {
                    "test_id": test_id,
                    "message": "No time series data available. Run simulation first.",
                    "data": [],
                    "variants": test.variants or [],
                    "completion_percentage": float(test.completion_percentage or 0.0),
                    "stopped_early": bool(test.stopped_early),
                    "early_stop_reason": test.early_stop_reason,
                    "current_sequential_look": int(test.current_sequential_look or 0),
                    "max_sequential_looks": int(test.max_sequential_looks or 5),
                    "srm_check_passed": (None if test.srm_check_passed is None else bool(test.srm_check_passed)),
                    "srm_p_value": test.srm_p_value,
                    "traffic_split": {"variant_counts": {}, "variant_percentages": {}},
                    "winner": None,
                    "winner_confidence": "low",
                    "analysis_mode": test.analysis_mode,
                    "analysis_validity": test.analysis_validity,
                    "guardrails": test.guardrails_status,
                    "p_values_corrected_latest": {},
                    "quality_gate": quality_gate,
                    "recommendation_status": "need_more_data",
                    "recommendation_reason": [
                        "Недостаточно данных временных рядов для итогового вывода",
                        "Эксперимент необходимо продолжить до накопления репрезентативной выборки",
                        "Решение о внедрении будет сформировано после появления winner и проверок валидности",
                    ],
                    "rollout_hint": "0% (продолжить эксперимент)",
                    "decision_policy": decision_policy,
                }

            normalized_rows: List[Dict[str, Any]] = []
            for record in time_series_records:
                variant = str(record.variant).strip() if record.variant is not None else ""
                users_processed = ResultsAnalyticsService._safe_int(record.users_processed)
                sample_size = ResultsAnalyticsService._safe_int(record.sample_size)

                if not variant or users_processed is None:
                    continue

                normalized_rows.append({
                    "users_processed": users_processed,
                    "variant": variant,
                    "cumulative_metric": ResultsAnalyticsService._safe_float(record.cumulative_metric),
                    "mean_metric": ResultsAnalyticsService._safe_float(record.mean_metric) or 0.0,
                    "sample_size": sample_size or 0,
                    "p_value": ResultsAnalyticsService._safe_float(record.p_value),
                    "confidence_interval_lower": ResultsAnalyticsService._safe_float(record.confidence_interval_lower),
                    "confidence_interval_upper": ResultsAnalyticsService._safe_float(record.confidence_interval_upper),
                })

            if not normalized_rows:
                return {
                    "test_id": test_id,
                    "message": "Time series rows are malformed or empty.",
                    "data": [],
                    "variants": test.variants or [],
                    "completion_percentage": float(test.completion_percentage or 0.0),
                    "stopped_early": bool(test.stopped_early),
                    "early_stop_reason": test.early_stop_reason,
                    "current_sequential_look": int(test.current_sequential_look or 0),
                    "max_sequential_looks": int(test.max_sequential_looks or 5),
                    "srm_check_passed": (None if test.srm_check_passed is None else bool(test.srm_check_passed)),
                    "srm_p_value": test.srm_p_value,
                    "traffic_split": {"variant_counts": {}, "variant_percentages": {}},
                    "winner": None,
                    "winner_confidence": "low",
                    "analysis_mode": test.analysis_mode,
                    "analysis_validity": test.analysis_validity,
                    "guardrails": test.guardrails_status,
                    "p_values_corrected_latest": {},
                    "quality_gate": quality_gate,
                    "recommendation_status": "need_more_data",
                    "recommendation_reason": [
                        "Данные временных рядов повреждены или отсутствуют",
                        "Невозможно корректно проверить p-value, SRM и uplift",
                        "Требуется повторный запуск/продолжение эксперимента",
                    ],
                    "rollout_hint": "0% (продолжить эксперимент)",
                    "decision_policy": decision_policy,
                }

            chart_data: List[Dict[str, Any]] = normalized_rows

            variants = sorted(list({row["variant"] for row in normalized_rows}))
            latest_users = max(row["users_processed"] for row in normalized_rows)
            latest_rows = [row for row in normalized_rows if row["users_processed"] == latest_users]

            control_variant = variants[0] if variants else None
            control_mean = 0.0
            latest_by_variant: Dict[str, Dict[str, Any]] = {}
            for row in latest_rows:
                latest_by_variant[row["variant"]] = row
                if row["variant"] == control_variant:
                    control_mean = float(row.get("mean_metric") or 0.0)

            raw_latest_p_values: Dict[str, float] = {}
            for v in variants:
                if v == control_variant:
                    continue
                row = latest_by_variant.get(v)
                if not row:
                    continue
                raw_latest_p_values[v] = float(row["p_value"]) if row.get("p_value") is not None else 1.0

            corrected_latest_p_values = ABDecisionEngine.holm_bonferroni_correction(raw_latest_p_values)

            winner = None
            best_uplift = -1e18
            winner_confidence = "low"
            for v in variants:
                if v == control_variant:
                    continue
                row = latest_by_variant.get(v)
                if not row:
                    continue
                row_mean = float(row.get("mean_metric") or 0.0)
                uplift = ((row_mean - control_mean) / control_mean * 100.0) if control_mean != 0 else 0.0
                p_val_corrected = corrected_latest_p_values.get(v, 1.0)

                if p_val_corrected < 0.05 and uplift > best_uplift:
                    best_uplift = uplift
                    winner = v
                    winner_confidence = "high"

            if winner is None:
                observed_best = -1e18
                for v in variants:
                    if v == control_variant:
                        continue
                    row = latest_by_variant.get(v)
                    if not row:
                        continue
                    row_mean = float(row.get("mean_metric") or 0.0)
                    uplift = ((row_mean - control_mean) / control_mean * 100.0) if control_mean != 0 else 0.0
                    if uplift > observed_best:
                        observed_best = uplift
                best_uplift = observed_best if observed_best != -1e18 else 0.0

            variant_counts = {
                v: int(latest_by_variant[v].get("sample_size") or 0) if v in latest_by_variant else 0
                for v in variants
            }
            total_count = sum(variant_counts.values())
            variant_percentages = {
                v: (variant_counts[v] / total_count * 100.0) if total_count > 0 else 0.0
                for v in variants
            }

            analyzer = StatisticalAnalyzer(alpha=0.05)
            grouped: Dict[int, Dict[str, Any]] = {}
            for row in chart_data:
                grouped.setdefault(row["users_processed"], {})[row["variant"]] = row

            power_over_time: List[Dict[str, Any]] = []
            uplift_over_time: List[Dict[str, Any]] = []

            for users_processed in sorted(grouped.keys()):
                point = grouped[users_processed]
                control = point.get(control_variant) if control_variant else None
                if not control:
                    continue

                power_point: Dict[str, Any] = {"users_processed": users_processed}
                uplift_point: Dict[str, Any] = {"users_processed": users_processed}

                for v in variants:
                    if v == control_variant:
                        continue
                    treatment = point.get(v)
                    if not treatment:
                        continue

                    observed_effect = float(treatment["mean_metric"] - control["mean_metric"])
                    sample_size_per_variant = max(1, min(int(control["sample_size"]), int(treatment["sample_size"])))

                    control_ci_low = control.get("confidence_interval_lower")
                    control_ci_high = control.get("confidence_interval_upper")
                    if control_ci_low is not None and control_ci_high is not None and sample_size_per_variant > 1:
                        se_est = abs(float(control_ci_high) - float(control_ci_low)) / (2.0 * 1.96)
                        std_proxy = max(1e-6, se_est * np.sqrt(sample_size_per_variant))
                    else:
                        std_proxy = max(1e-6, abs(float(control["mean_metric"])) * 0.5)

                    power_point[v] = analyzer.calculate_power(
                        observed_effect=observed_effect,
                        sample_size_per_variant=sample_size_per_variant,
                        baseline_std=std_proxy,
                        alpha=0.05,
                    )

                    uplift_point[v] = (
                        (float(treatment["mean_metric"]) - float(control["mean_metric"])) / float(control["mean_metric"]) * 100.0
                    ) if float(control["mean_metric"]) != 0 else 0.0

                power_over_time.append(power_point)
                uplift_over_time.append(uplift_point)

            srm_passed = None if test.srm_check_passed is None else bool(test.srm_check_passed)
            guardrails_status = ResultsAnalyticsService._as_dict(test.guardrails_status)
            guardrails_enabled = bool(guardrails_status.get("enabled", False))
            guardrails_passed = bool(guardrails_status.get("passed", True)) if guardrails_enabled else True
            analysis_valid = str(test.analysis_validity or "") == "valid_for_inference"
            if not analysis_valid:
                winner = None
                winner_confidence = "low"
                best_uplift = 0.0

            winner_present = winner is not None
            winner_corrected_p = corrected_latest_p_values.get(str(winner), 1.0) if winner_present else 1.0
            winner_p_ok = bool(winner_present and winner_corrected_p < 0.05)
            uplift_positive = bool(winner_present and float(best_uplift) > 0.0)

            deploy_allowed = all([
                winner_present,
                winner_p_ok,
                analysis_valid,
                srm_passed is True,
                guardrails_passed,
                uplift_positive,
            ])

            recommendation_reason: List[str] = [
                f"Победитель: {'определён (' + str(winner) + ')' if winner_present else 'не определён'}",
                f"Скорректированное p-значение победителя: {winner_corrected_p:.4f} ({'норма' if winner_p_ok else 'не норма'})",
                f"Валидность анализа: {ResultsAnalyticsService._translate_analysis_validity(test.analysis_validity)} ({'норма' if analysis_valid else 'не норма'})",
                f"Проверка равномерности трафика (SRM): {'пройдена' if srm_passed is True else ('не пройдена' if srm_passed is False else 'нет данных')}",
                f"Защитные ограничения (guardrails): {'соблюдены' if guardrails_passed else 'нарушены'}",
                f"Прирост победителя: {float(best_uplift):.2f}% ({'норма' if uplift_positive else 'не норма'})",
            ]

            hard_blockers = [
                not analysis_valid,
                srm_passed is False,
                not guardrails_passed,
                winner_present and not uplift_positive,
            ]

            if decision_policy and decision_policy.get("allowed") is True:
                recommendation_status = "deploy"
                rollout_hint = "100%" if winner_confidence == "high" and quality_gate.get("status") == "green" else "50%"
            elif decision_policy and decision_policy.get("allowed") is False:
                reasons = decision_policy.get("reasons") or []
                if any(
                    key in " ".join(reasons)
                    for key in ["SRM", "Guardrails", "Невалидный дизайн"]
                ):
                    recommendation_status = "do_not_deploy"
                    rollout_hint = "0%"
                else:
                    recommendation_status = "need_more_data"
                    rollout_hint = "0% (продолжить эксперимент)"
            elif deploy_allowed:
                recommendation_status = "deploy"
                rollout_hint = "100%" if winner_confidence == "high" and quality_gate.get("status") == "green" else "50%"
            elif any(hard_blockers):
                recommendation_status = "do_not_deploy"
                rollout_hint = "0%"
            else:
                recommendation_status = "need_more_data"
                rollout_hint = "0% (продолжить эксперимент)"

            response_payload = {
                "test_id": test_id,
                "variants": variants,
                "data": chart_data,
                "total_snapshots": len(chart_data),
                "snapshots_per_variant": len(chart_data) // len(variants) if variants else 0,
                "completion_percentage": float(test.completion_percentage or 0.0),
                "stopped_early": bool(test.stopped_early),
                "early_stop_reason": test.early_stop_reason,
                "early_stopping_enabled": bool(ResultsAnalyticsService._as_dict(test.extra_config).get("early_stopping_enabled", False)),
                "current_sequential_look": int(test.current_sequential_look or 0),
                "max_sequential_looks": int(test.max_sequential_looks or 5),
                "srm_check_passed": srm_passed,
                "srm_p_value": test.srm_p_value,
                "traffic_split": {
                    "variant_counts": variant_counts,
                    "variant_percentages": variant_percentages,
                },
                "winner": winner,
                "winner_uplift_percent": float(best_uplift) if winner else 0.0,
                "winner_confidence": winner_confidence,
                "power_over_time": power_over_time,
                "uplift_over_time": uplift_over_time,
                "analysis_mode": test.analysis_mode,
                "analysis_validity": test.analysis_validity,
                "guardrails": test.guardrails_status,
                "p_values_corrected_latest": corrected_latest_p_values,
                "quality_gate": quality_gate,
                "recommendation_status": recommendation_status,
                "recommendation_reason": recommendation_reason,
                "rollout_hint": rollout_hint,
                "decision_policy": decision_policy,
            }
            return ResultsAnalyticsService._sanitize_for_json(response_payload)

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @staticmethod
    def _as_dict(value: Any) -> Dict[str, Any]:
        return value if isinstance(value, dict) else {}

    @staticmethod
    def _translate_analysis_validity(validity: Any) -> str:
        mapping = {
            "valid_for_inference": "валиден для итогового вывода",
            "exploration_only": "только исследовательский режим",
            "invalid_srm": "невалиден: перекос трафика",
            "invalid_guardrails": "невалиден: нарушены защитные метрики",
        }
        return mapping.get(str(validity or ""), str(validity or "нет данных"))

    @staticmethod
    def _safe_float(value: Any) -> Optional[float]:
        try:
            if value is None:
                return None
            f = float(value)
            return f if math.isfinite(f) else None
        except Exception:
            return None

    @staticmethod
    def _safe_int(value: Any) -> Optional[int]:
        try:
            if value is None:
                return None
            f = float(value)
            if not math.isfinite(f):
                return None
            return int(f)
        except Exception:
            return None

    @staticmethod
    def _sanitize_for_json(value: Any) -> Any:
        if isinstance(value, dict):
            return {k: ResultsAnalyticsService._sanitize_for_json(v) for k, v in value.items()}
        if isinstance(value, list):
            return [ResultsAnalyticsService._sanitize_for_json(v) for v in value]
        if isinstance(value, float):
            return value if math.isfinite(value) else None
        if isinstance(value, np.floating):
            f = float(value)
            return f if math.isfinite(f) else None
        return value

    @staticmethod
    def _to_stat_object(stat: Any) -> Any:
        if isinstance(stat, dict):
            return type(
                "StatObj",
                (),
                {
                    "mean": float(stat.get("mean", 0.0)),
                    "std": float(stat.get("std", 0.0)),
                    "sample_size": int(stat.get("sample_size", 0)),
                },
            )()
        return stat

    @staticmethod
    def _calculate_power(control_data: Any, variant_data: Any, alpha: float) -> float:
        try:
            effect_size = abs(variant_data.mean - control_data.mean)
            pooled_std = np.sqrt((control_data.std ** 2 + variant_data.std ** 2) / 2)

            if pooled_std == 0:
                return 0.0

            standardized_effect = effect_size / pooled_std
            power = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(0.8)
            return min(1.0, max(0.0, stats.norm.cdf(standardized_effect * np.sqrt(variant_data.sample_size / 2) - power)))
        except Exception:
            return 0.0

    @staticmethod
    def _calculate_effect_confidence_interval(control_data: Any, variant_data: Any) -> List[float]:
        try:
            mean_diff = variant_data.mean - control_data.mean
            se_diff = np.sqrt(
                (control_data.std ** 2 / control_data.sample_size) +
                (variant_data.std ** 2 / variant_data.sample_size)
            )

            t_critical = stats.t.ppf(0.975, min(control_data.sample_size, variant_data.sample_size) - 1)
            margin = t_critical * se_diff

            return [float(mean_diff - margin), float(mean_diff + margin)]
        except Exception:
            return [0.0, 0.0]

    @staticmethod
    def _calculate_required_sample_size(control_data: Any, variant_data: Any, alpha: float, power: float) -> int:
        try:
            effect_size = abs(variant_data.mean - control_data.mean)
            pooled_std = np.sqrt((control_data.std ** 2 + variant_data.std ** 2) / 2)

            if pooled_std == 0:
                return 0

            standardized_effect = effect_size / pooled_std
            z_alpha = stats.norm.ppf(1 - alpha / 2)
            z_beta = stats.norm.ppf(power)

            required_n = 2 * ((z_alpha + z_beta) / standardized_effect) ** 2
            return int(np.ceil(required_n))
        except Exception:
            return 0

    @staticmethod
    def _interpret_significance(significance_analysis: Dict[str, Any], alpha: float) -> Dict[str, str]:
        significant_variants = [
            variant for variant, analysis in significance_analysis.items()
            if analysis["statistically_significant"]
        ]

        if not significant_variants:
            return {
                "conclusion": "Нет статистически значимых различий",
                "recommendation": "Продолжить сбор данных или рассмотреть другие метрики"
            }

        best_variant = max(significant_variants, key=lambda v: significance_analysis[v]["effect_size"])

        return {
            "conclusion": f"Статистически значимые различия обнаружены для вариантов: {', '.join(significant_variants)}",
            "best_performing": best_variant,
            "recommendation": f"Рекомендуется внедрить вариант {best_variant}",
        }

    @staticmethod
    def get_time_series_data(*, test_id: str, time_window: str) -> Dict[str, Any]:
        with SessionLocal() as db:
            records = crud.get_ab_test_time_series(db, test_id, limit=5000)

        variants = sorted(list(set(r.variant for r in records)))
        grouped: Dict[int, Dict[str, Any]] = {}
        for r in records:
            grouped.setdefault(r.users_processed, {"users_processed": r.users_processed})[r.variant] = {
                "mean_metric": r.mean_metric,
                "cumulative_metric": r.cumulative_metric,
                "p_value": r.p_value,
                "sample_size": r.sample_size,
            }

        return {
            "variants": variants,
            "points": [grouped[k] for k in sorted(grouped.keys())],
            "time_window": time_window,
        }

    @staticmethod
    def analyze_trends(*, time_series_data: Dict[str, Any]) -> Dict[str, Any]:
        points = time_series_data.get("points", [])
        if len(points) < 2:
            return {"trend": "insufficient_data"}

        variants = time_series_data.get("variants", [])
        if not variants:
            return {"trend": "insufficient_data"}

        control = variants[0]
        first = points[0]
        last = points[-1]

        trend = {}
        for v in variants:
            if v not in first or v not in last:
                continue
            start = first[v].get("mean_metric", 0.0)
            end = last[v].get("mean_metric", 0.0)
            trend[v] = {
                "start_mean": start,
                "end_mean": end,
                "delta": end - start,
            }

        control_delta = trend.get(control, {}).get("delta", 0.0)
        best = None
        best_delta = -1e18
        for v, t in trend.items():
            rel = t["delta"] - control_delta
            if rel > best_delta:
                best_delta = rel
                best = v

        return {
            "trend": "ok",
            "by_variant": trend,
            "best_trending_variant": best,
        }

    @staticmethod
    def get_segmentation_analysis(*, test_id: str, segment_by: str) -> Dict[str, Any]:
        with SessionLocal() as db:
            test = db.query(ABTestORM).filter(ABTestORM.test_id == test_id).first()
            if not test or not test.dataset_id:
                return {"segments": {}, "message": "No dataset bound to this test"}

            dataset = crud.get_generated_data_by_id(db, test.dataset_id)
            if not dataset:
                return {"segments": {}, "message": "Dataset not found"}

            records = dataset.load_records() if hasattr(dataset, "load_records") else None
            if records is None:
                from backend.microservices.data_gan.service import DatasetPersistenceService
                records = DatasetPersistenceService.load_dataset_records_for_entity(dataset)
            if not records:
                return {"segments": {}, "message": "No records in dataset"}

            sessions = db.query(TestSessionORM).filter(TestSessionORM.test_id == test_id).all()

        record_by_user_id: Dict[str, Dict[str, Any]] = {}
        for row in records:
            uid = row.get("user_id")
            if uid is not None:
                record_by_user_id[str(uid)] = row

        primary_metric = str(test.primary_metric or "")
        ratio_num_key = f"{primary_metric}_numerator"
        ratio_den_key = f"{primary_metric}_denominator"
        metric_type = str(test.metric_type or "continuous")

        segments: Dict[str, Dict[str, Any]] = {}
        matched_sessions = 0

        for session in sessions:
            user_row = record_by_user_id.get(str(session.user_id))
            if user_row is None:
                continue

            segment_value = str(user_row.get(segment_by, "unknown"))
            variant = str(session.variant)
            metrics = dict(session.metrics or {})

            metric_value: Optional[float] = None
            if metric_type == "ratio":
                num_raw = metrics.get(ratio_num_key)
                den_raw = metrics.get(ratio_den_key)
                try:
                    if num_raw is not None and den_raw is not None:
                        num = float(num_raw)
                        den = float(den_raw)
                        if np.isfinite(num) and np.isfinite(den) and den > 0:
                            metric_value = float(num / den)
                except Exception:
                    metric_value = None
            else:
                try:
                    raw = metrics.get(primary_metric)
                    if raw is not None:
                        value = float(raw)
                        if np.isfinite(value):
                            metric_value = value
                except Exception:
                    metric_value = None

            if segment_value not in segments:
                segments[segment_value] = {
                    "users": 0,
                    "by_variant": {},
                }

            seg = segments[segment_value]
            seg["users"] += 1

            if variant not in seg["by_variant"]:
                seg["by_variant"][variant] = {
                    "sample_size": 0,
                    "mean_metric": 0.0,
                    "metric_sum": 0.0,
                    "metric_count": 0,
                }

            card = seg["by_variant"][variant]
            card["sample_size"] += 1
            if metric_value is not None:
                card["metric_sum"] += float(metric_value)
                card["metric_count"] += 1

            matched_sessions += 1

        total = max(1, sum(seg["users"] for seg in segments.values()))

        for seg_data in segments.values():
            for card in seg_data["by_variant"].values():
                cnt = int(card.get("metric_count", 0))
                card["mean_metric"] = float(card["metric_sum"] / cnt) if cnt > 0 else 0.0
                card.pop("metric_sum", None)
                card.pop("metric_count", None)
            seg_data["share_percent"] = float(seg_data["users"]) / total * 100.0

        return {
            "segments": segments,
            "total_users": total,
            "matched_sessions": matched_sessions,
            "metric": primary_metric,
            "metric_type": metric_type,
        }

    @staticmethod
    def compare_segments(*, segments_analysis: Dict[str, Any]) -> Dict[str, Any]:
        segments = segments_analysis.get("segments", {})
        if not segments:
            return {"message": "No segments"}

        sorted_segments = sorted(segments.items(), key=lambda kv: kv[1].get("users", 0), reverse=True)
        return {
            "largest_segment": sorted_segments[0][0],
            "smallest_segment": sorted_segments[-1][0],
            "segment_count": len(sorted_segments),
        }

    @staticmethod
    def calculate_performance_metrics(*, test_history: List[Dict[str, Any]], days: int) -> Dict[str, Any]:
        completed = [t for t in test_history if t.get("status") in ["completed", "archived"]]
        completion_rates = [float(t.get("completion_percentage", 0.0)) for t in completed]
        return {
            "tests_analyzed": len(test_history),
            "completed_or_archived": len(completed),
            "avg_completion_percentage": float(np.mean(completion_rates)) if completion_rates else 0.0,
            "window_days": days,
        }

    @staticmethod
    def generate_performance_recommendations(*, performance_metrics: Dict[str, Any]) -> List[str]:
        recs: List[str] = []
        avg_completion = float(performance_metrics.get("avg_completion_percentage", 0.0))
        if avg_completion < 70:
            recs.append("Увеличить целевую длительность симуляций или объём трафика для достижения стабильных результатов")
        recs.append("Проверять SRM и sequential-look перед финальным решением по победителю")
        recs.append("Использовать fixed traffic split для финальной валидации гипотез")
        return recs

    @staticmethod
    def get_detailed_results(*, test_id: str, platform: Any) -> Dict[str, Any]:
        results = platform.get_test_results(test_id)
        detailed_analysis = ResultsAnalyticsService._perform_detailed_analysis(results)
        return {
            "test_id": test_id,
            "basic_results": results,
            "detailed_analysis": detailed_analysis,
            "generated_at": datetime.now().isoformat(),
        }

    @staticmethod
    def _perform_detailed_analysis(results: Dict[str, Any]) -> Dict[str, Any]:
        basic_results = results.get("results", {})
        control_variant = list(basic_results.keys())[0] if basic_results else None

        analysis = {}
        for variant, result_data in basic_results.items():
            if variant == control_variant:
                continue

            analysis[variant] = {
                "relative_improvement": ResultsAnalyticsService._calculate_relative_improvement(
                    basic_results[control_variant]["mean"],
                    result_data["mean"],
                ),
                "confidence_interval_width": result_data["confidence_interval"][1] - result_data["confidence_interval"][0],
                "coefficient_of_variation": result_data["std"] / result_data["mean"] if result_data["mean"] != 0 else 0,
                "sample_efficiency": result_data["sample_size"] / basic_results[control_variant]["sample_size"] if basic_results[control_variant]["sample_size"] != 0 else 0,
            }

        return analysis

    @staticmethod
    def _calculate_relative_improvement(control_mean: float, variant_mean: float) -> float:
        if control_mean == 0:
            return 0.0
        return ((variant_mean - control_mean) / control_mean) * 100
