# backend/api/routes/results.py

from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import numpy as np
import scipy.stats as stats
from datetime import datetime
import math

from backend.microservices.ab_testing.service import ABPlatformProvider
from backend.microservices.auth_core.models import User
from backend.microservices.auth_core.service import get_current_user
from backend.microservices.database.session import SessionLocal
from backend.microservices.database import crud
from backend.microservices.database.models import ABTestORM, TestSessionORM
from backend.microservices.ab_testing_core.statistics import StatisticalAnalyzer
from backend.microservices.data_gan.service import DatasetPersistenceService
from backend.microservices.ab_testing_core.decision_engine import ABDecisionEngine

router = APIRouter(prefix="/api/v1/results", tags=["Results & Analytics"])
platform = ABPlatformProvider.get()


class StatisticalSummary(BaseModel):
    variant: str
    sample_size: int
    mean: float
    std: float
    confidence_interval: List[float]
    relative_improvement: Optional[float] = None


class TestAnalysis(BaseModel):
    test_id: str
    best_variant: str
    improvement_percentage: float
    confidence_level: str
    recommended_action: str
    statistical_significance: Dict[str, float]
    summary: Dict[str, StatisticalSummary]


@router.get("/{test_id}/detailed", summary="Детальные результаты теста")
async def get_detailed_results(test_id: str, current_user: User = Depends(get_current_user)):
    try:
        results = platform.get_test_results(test_id)
        detailed_analysis = await _perform_detailed_analysis(results)

        return {
            "test_id": test_id,
            "basic_results": results,
            "detailed_analysis": detailed_analysis,
            "generated_at": datetime.now().isoformat()
        }

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{test_id}/statistical-significance", summary="Статистическая значимость")
async def get_statistical_significance(test_id: str, alpha: float = 0.05, current_user: User = Depends(get_current_user)):
    try:
        db_results = platform.get_test_results(test_id)
        results = db_results.get("results", {})
        p_values = db_results.get("statistical_significance", {})

        if not results:
            return {
                "test_id": test_id,
                "alpha_level": alpha,
                "significance_analysis": {},
                "interpretation": _interpret_significance({}, alpha),
            }

        significance_analysis = {}
        control_variant = list(results.keys())[0]
        control_data = _to_stat_object(results[control_variant])

        for variant, p_value in p_values.items():
            variant_payload = results.get(variant)
            if not variant_payload:
                continue

            variant_data = _to_stat_object(variant_payload)

            power = await _calculate_power(control_data, variant_data, alpha)
            effect_size = float(variant_data.mean - control_data.mean)
            effect_ci = await _calculate_effect_confidence_interval(control_data, variant_data)

            significance_analysis[variant] = {
                "p_value": float(p_value),
                "statistically_significant": float(p_value) < alpha,
                "power": power,
                "effect_size": effect_size,
                "effect_confidence_interval": effect_ci,
                "required_sample_size": await _calculate_required_sample_size(
                    control_data, variant_data, alpha, 0.8
                )
            }

        return {
            "test_id": test_id,
            "alpha_level": alpha,
            "significance_analysis": significance_analysis,
            "interpretation": _interpret_significance(significance_analysis, alpha)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{test_id}/time-series", summary="Временные ряды метрик")
async def get_time_series_data(
    test_id: str,
    time_window: str = Query("7d", pattern="^(1d|7d|30d|all)$"),
    current_user: User = Depends(get_current_user)
):
    try:
        time_series_data = await _generate_time_series_data(test_id, time_window)

        return {
            "test_id": test_id,
            "time_window": time_window,
            "time_series": time_series_data,
            "trend_analysis": await _analyze_trends(time_series_data)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{test_id}/segmentation", summary="Анализ по сегментам")
async def get_segmentation_analysis(test_id: str, segment_by: str = "user_type", current_user: User = Depends(get_current_user)):
    try:
        segments_analysis = await _generate_segmentation_analysis(test_id, segment_by)

        return {
            "test_id": test_id,
            "segment_by": segment_by,
            "segments_analysis": segments_analysis,
            "segment_comparison": await _compare_segments(segments_analysis)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{test_id}/financial-impact", summary="Финансовый анализ")
async def get_financial_impact(test_id: str, arpu: float = 100.0, current_user: User = Depends(get_current_user)):
    try:
        db_results = platform.get_test_results(test_id)
        results = db_results.get("results", {})
        corrected_p = db_results.get("statistical_significance_corrected", {}) or {}
        financial_analysis = await _calculate_financial_impact(results, corrected_p, arpu)

        return {
            "test_id": test_id,
            "assumed_arpu": arpu,
            "financial_analysis": financial_analysis,
            "roi_calculation": await _calculate_roi(financial_analysis)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/platform/performance", summary="Производительность платформы")
async def get_platform_performance(days: int = 30, current_user: User = Depends(get_current_user)):
    try:
        platform_stats = platform.get_platform_stats()
        test_history = platform.test_registry.get_test_history(limit=100)

        performance_metrics = await _calculate_performance_metrics(test_history, days)

        return {
            "time_period_days": days,
            "platform_stats": platform_stats,
            "performance_metrics": performance_metrics,
            "recommendations": await _generate_performance_recommendations(performance_metrics)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{test_id}/export", summary="Экспорт результатов")
async def export_test_results(
    test_id: str,
    format: str = Query("json", pattern="^(json|csv|excel)$"),
    include_raw_data: bool = False,
    current_user: User = Depends(get_current_user),
):
    try:
        db_results = platform.get_test_results(test_id)

        export_data = {
            "test_id": test_id,
            "exported_at": datetime.now().isoformat(),
            "summary": db_results.get("summary", {}),
            "detailed_results": db_results.get("results", {}),
            "statistical_significance": db_results.get("statistical_significance", {}),
            "statistical_significance_corrected": db_results.get("statistical_significance_corrected", {}),
            "quality_gate": db_results.get("quality_gate", {}),
        }

        if include_raw_data:
            export_data["session_metrics"] = db_results.get("session_metrics", {})

        if format == "json":
            return export_data
        if format == "csv":
            return {"csv_data": await _convert_to_csv(export_data)}
        return {"excel_data": "Base64 encoded Excel file"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{test_id}/time-series-data", summary="Данные временных рядов для графиков")
async def get_time_series_chart_data(
    test_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Возвращает расширенные данные для графиков:
    - cumulative / mean / p-value / CI
    - early stopping
    - sequential progress
    - SRM
    - traffic split
    - winner snapshot
    """
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
        except Exception:
            # Не блокируем отрисовку графиков из-за вторичных ошибок агрегированного summary.
            quality_gate = {
                "status": "yellow",
                "passed": False,
                "passed_checks": 0,
                "total_checks": 5,
                "checks": [],
            }

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
                "srm_check_passed": test.srm_check_passed,
                "srm_p_value": test.srm_p_value,
                "traffic_split": {"variant_counts": {}, "variant_percentages": {}},
                "winner": None,
                "winner_confidence": "low",
                "analysis_mode": test.analysis_mode,
                "analysis_validity": test.analysis_validity,
                "guardrails": test.guardrails_status,
                "p_values_corrected_latest": {},
                "quality_gate": quality_gate,
            }

        normalized_rows: List[Dict[str, Any]] = []
        for record in time_series_records:
            variant = str(record.variant).strip() if record.variant is not None else ""
            users_processed = _safe_int(record.users_processed)
            sample_size = _safe_int(record.sample_size)

            # Пропускаем битые строки, чтобы не ронять весь endpoint.
            if not variant or users_processed is None:
                continue

            normalized_rows.append({
                "users_processed": users_processed,
                "variant": variant,
                "cumulative_metric": _safe_float(record.cumulative_metric),
                "mean_metric": _safe_float(record.mean_metric) or 0.0,
                "sample_size": sample_size or 0,
                "p_value": _safe_float(record.p_value),
                "confidence_interval_lower": _safe_float(record.confidence_interval_lower),
                "confidence_interval_upper": _safe_float(record.confidence_interval_upper),
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
                "srm_check_passed": test.srm_check_passed,
                "srm_p_value": test.srm_p_value,
                "traffic_split": {"variant_counts": {}, "variant_percentages": {}},
                "winner": None,
                "winner_confidence": "low",
                "analysis_mode": test.analysis_mode,
                "analysis_validity": test.analysis_validity,
                "guardrails": test.guardrails_status,
                "p_values_corrected_latest": {},
                "quality_gate": quality_gate,
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

            # Победителя определяем только среди статистически значимых вариантов
            # с учетом коррекции множественных сравнений.
            if p_val_corrected < 0.05 and uplift > best_uplift:
                best_uplift = uplift
                winner = v
                winner_confidence = "high"

        # Если значимого победителя нет, возвращаем отсутствие победителя,
        # но сохраняем наилучший наблюдаемый uplift для информативности UI.
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

        # оценка power over time (для каждого среза, treatment vs control)
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

                # baseline_std должен быть оценкой дисперсии метрики, а не средним значением.
                # Для быстрых time-series оценим std по ДИ контроля: CI ~ mean ± 1.96 * SE.
                control_ci_low = control.get("confidence_interval_lower")
                control_ci_high = control.get("confidence_interval_upper")
                if control_ci_low is not None and control_ci_high is not None and sample_size_per_variant > 1:
                    se_est = abs(float(control_ci_high) - float(control_ci_low)) / (2.0 * 1.96)
                    std_proxy = max(1e-6, se_est * np.sqrt(sample_size_per_variant))
                else:
                    # Фолбэк на относительную оценку только если ДИ ещё недоступен.
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

        response_payload = {
            "test_id": test_id,
            "variants": variants,
            "data": chart_data,
            "total_snapshots": len(chart_data),
            "snapshots_per_variant": len(chart_data) // len(variants) if variants else 0,
            "completion_percentage": float(test.completion_percentage or 0.0),
            "stopped_early": bool(test.stopped_early),
            "early_stop_reason": test.early_stop_reason,
            "early_stopping_enabled": bool(_as_dict(test.extra_config).get("early_stopping_enabled", False)),
            "current_sequential_look": int(test.current_sequential_look or 0),
            "max_sequential_looks": int(test.max_sequential_looks or 5),
            "srm_check_passed": test.srm_check_passed,
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
        }
        return _sanitize_for_json(response_payload)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        f = float(value)
        return f if math.isfinite(f) else None
    except Exception:
        return None


def _safe_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        # Защита от nan/inf и строковых артефактов.
        f = float(value)
        if not math.isfinite(f):
            return None
        return int(f)
    except Exception:
        return None


def _sanitize_for_json(value: Any) -> Any:
    """
    Рекурсивно очищает payload от NaN/Inf значений,
    которые приводят к 500 при JSON-сериализации в FastAPI/Starlette.
    """
    if isinstance(value, dict):
        return {k: _sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_for_json(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.floating):
        f = float(value)
        return f if math.isfinite(f) else None
    return value


async def _perform_detailed_analysis(results: Dict[str, Any]) -> Dict[str, Any]:
    """Выполняет детальный статистический анализ результатов"""
    basic_results = results.get('results', {})
    control_variant = list(basic_results.keys())[0] if basic_results else None

    analysis = {}
    for variant, result_data in basic_results.items():
        if variant == control_variant:
            continue

        analysis[variant] = {
            "relative_improvement": _calculate_relative_improvement(
                basic_results[control_variant]['mean'],
                result_data['mean']
            ),
            "confidence_interval_width": result_data['confidence_interval'][1] - result_data['confidence_interval'][0],
            "coefficient_of_variation": result_data['std'] / result_data['mean'] if result_data['mean'] != 0 else 0,
            "sample_efficiency": result_data['sample_size'] / basic_results[control_variant]['sample_size'] if basic_results[control_variant]['sample_size'] != 0 else 0
        }

    return analysis


async def _calculate_power(control_data: Any, variant_data: Any, alpha: float) -> float:
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


async def _calculate_effect_confidence_interval(control_data: Any, variant_data: Any) -> List[float]:
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


async def _calculate_required_sample_size(control_data: Any, variant_data: Any, alpha: float, power: float) -> int:
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


def _interpret_significance(significance_analysis: Dict[str, Any], alpha: float) -> Dict[str, str]:
    significant_variants = [
        variant for variant, analysis in significance_analysis.items()
        if analysis['statistically_significant']
    ]

    if not significant_variants:
        return {
            "conclusion": "Нет статистически значимых различий",
            "recommendation": "Продолжить сбор данных или рассмотреть другие метрики"
        }

    best_variant = max(significant_variants, key=lambda v: significance_analysis[v]['effect_size'])

    return {
        "conclusion": f"Статистически значимые различия обнаружены для вариантов: {', '.join(significant_variants)}",
        "best_performing": best_variant,
        "recommendation": f"Рекомендуется внедрить вариант {best_variant}"
    }


async def _generate_time_series_data(test_id: str, time_window: str) -> Dict[str, Any]:
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


async def _generate_segmentation_analysis(test_id: str, segment_by: str) -> Dict[str, Any]:
    with SessionLocal() as db:
        test = db.query(ABTestORM).filter(ABTestORM.test_id == test_id).first()
        if not test or not test.dataset_id:
            return {"segments": {}, "message": "No dataset bound to this test"}

        dataset = crud.get_generated_data_by_id(db, test.dataset_id)
        if not dataset:
            return {"segments": {}, "message": "Dataset not found"}

        records = DatasetPersistenceService.load_dataset_records_for_entity(dataset)
        if not records:
            return {"segments": {}, "message": "No records in dataset"}

        sessions = db.query(TestSessionORM).filter(TestSessionORM.test_id == test_id).all()

    # Индексируем датасет по user_id, чтобы анализировать фактические outcome по сегментам и вариантам.
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

            if metric_value is None and primary_metric in metrics:
                try:
                    raw_ratio = float(metrics.get(primary_metric))
                    if np.isfinite(raw_ratio):
                        metric_value = raw_ratio
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

    # Финализация mean
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


async def _calculate_financial_impact(
    results: Dict[str, Any],
    corrected_p_values: Dict[str, float],
    arpu: float,
) -> Dict[str, Any]:
    variants = list(results.keys())
    if not variants:
        return {"incremental_revenue": 0.0, "best_variant": None}

    control_variant = variants[0]
    control = _to_stat_object(results[control_variant])
    control_mean = float(control.mean)
    control_n = int(control.sample_size)

    by_variant = {}
    best_significant_variant: Optional[str] = None
    best_significant_incremental = 0.0

    best_observed_variant: Optional[str] = None
    best_observed_incremental = -1e18

    for v in variants:
        stat = _to_stat_object(results[v])
        mean_val = float(stat.mean)
        n_val = int(stat.sample_size)
        uplift = ((mean_val - control_mean) / control_mean * 100.0) if control_mean != 0 else 0.0

        # Оценка инкремента: uplift метрики * ARPU * размер когорты варианта.
        incremental = (mean_val - control_mean) * n_val * float(arpu)

        p_corr = None if v == control_variant else corrected_p_values.get(v)
        is_significant = bool(p_corr is not None and float(p_corr) < 0.05)

        by_variant[v] = {
            "uplift_percent": float(uplift),
            "sample_size": int(n_val),
            "incremental_revenue": float(incremental),
            "p_value_corrected": (None if p_corr is None else float(p_corr)),
            "significant": is_significant,
        }

        if v != control_variant and incremental > best_observed_incremental:
            best_observed_incremental = float(incremental)
            best_observed_variant = v

        if v != control_variant and is_significant and incremental > best_significant_incremental:
            best_significant_incremental = float(incremental)
            best_significant_variant = v

    deployed_variant = best_significant_variant
    deployed_incremental = best_significant_incremental if best_significant_variant else 0.0

    return {
        "control_variant": control_variant,
        "best_variant": deployed_variant,
        "best_observed_variant": best_observed_variant,
        "incremental_revenue": float(deployed_incremental),
        "best_observed_incremental_revenue": float(max(0.0, best_observed_incremental if best_observed_incremental != -1e18 else 0.0)),
        "by_variant": by_variant,
        "control_users": int(control_n),
        "assumptions": {
            "arpu": float(arpu),
            "uses_significance_gate": True,
            "significance_threshold": 0.05,
        },
    }


async def _convert_to_csv(data: Dict[str, Any]) -> str:
    return "test_id,exported_at\n" + f"{data.get('test_id')},{data.get('exported_at')}\n"


async def _convert_to_excel(data: Dict[str, Any]) -> str:
    return "Excel export placeholder"


async def _analyze_trends(time_series_data: Dict[str, Any]) -> Dict[str, Any]:
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


async def _compare_segments(segments_analysis: Dict[str, Any]) -> Dict[str, Any]:
    segments = segments_analysis.get("segments", {})
    if not segments:
        return {"message": "No segments"}

    sorted_segments = sorted(segments.items(), key=lambda kv: kv[1].get("users", 0), reverse=True)
    return {
        "largest_segment": sorted_segments[0][0],
        "smallest_segment": sorted_segments[-1][0],
        "segment_count": len(sorted_segments),
    }


async def _calculate_roi(financial_analysis: Dict[str, Any]) -> Dict[str, Any]:
    incremental_revenue = float(financial_analysis.get("incremental_revenue", 0.0))

    cost_scenarios = [500.0, 1000.0, 3000.0]
    rollout_shares = [0.25, 0.5, 1.0]

    matrix: List[Dict[str, Any]] = []
    for cost in cost_scenarios:
        for share in rollout_shares:
            effective_incremental = incremental_revenue * float(share)
            roi = ((effective_incremental - cost) / cost) * 100.0 if cost > 0 else 0.0
            matrix.append(
                {
                    "estimated_cost": float(cost),
                    "rollout_share": float(share),
                    "incremental_revenue": float(effective_incremental),
                    "roi_percent": float(roi),
                }
            )

    return {
        "base_incremental_revenue": float(incremental_revenue),
        "scenarios": matrix,
        "note": "ROI рассчитан по сценариям стоимости и доли rollout вместо фиксированной заглушки",
    }


async def _calculate_performance_metrics(test_history: List[Dict[str, Any]], days: int) -> Dict[str, Any]:
    completed = [t for t in test_history if t.get("status") in ["completed", "archived"]]
    completion_rates = [float(t.get("completion_percentage", 0.0)) for t in completed]
    return {
        "tests_analyzed": len(test_history),
        "completed_or_archived": len(completed),
        "avg_completion_percentage": float(np.mean(completion_rates)) if completion_rates else 0.0,
        "window_days": days,
    }


async def _generate_performance_recommendations(performance_metrics: Dict[str, Any]) -> List[str]:
    recs: List[str] = []
    avg_completion = float(performance_metrics.get("avg_completion_percentage", 0.0))
    if avg_completion < 70:
        recs.append("Увеличить целевую длительность симуляций или объём трафика для достижения стабильных результатов")
    recs.append("Проверять SRM и sequential-look перед финальным решением по победителю")
    recs.append("Использовать fixed traffic split для финальной валидации гипотез")
    return recs


def _calculate_relative_improvement(control_mean: float, variant_mean: float) -> float:
    """Расчет относительного улучшения"""
    if control_mean == 0:
        return 0.0
    return ((variant_mean - control_mean) / control_mean) * 100


def _to_stat_object(stat: Any) -> Any:
    """
    Приводит статистику варианта к объекту с полями .mean/.std/.sample_size.
    Поддерживает как dict (DB-centric ответ), так и dataclass-объекты.
    """
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
