# backend/api/routes/results.py

from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import numpy as np
import scipy.stats as stats
from datetime import datetime

from backend.api.platform_instance import get_platform
from backend.auth.models import User
from backend.auth.service import get_current_user
from backend.database.session import SessionLocal
from backend.database import crud
from backend.database.models import ABTestORM
from backend.ab_testing.statistics import StatisticalAnalyzer
from backend.microservices.data_gan.service import DatasetPersistenceService

router = APIRouter(prefix="/api/v1/results", tags=["Results & Analytics"])
platform = get_platform()


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
        results, p_values = platform.test_manager.get_test_results(test_id)

        significance_analysis = {}
        control_variant = list(results.keys())[0]
        control_data = results[control_variant]

        for variant, p_value in p_values.items():
            variant_data = results[variant]

            power = await _calculate_power(control_data, variant_data, alpha)
            effect_size = variant_data.mean - control_data.mean
            effect_ci = await _calculate_effect_confidence_interval(control_data, variant_data)

            significance_analysis[variant] = {
                "p_value": p_value,
                "statistically_significant": p_value < alpha,
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
        results, _ = platform.test_manager.get_test_results(test_id)
        financial_analysis = await _calculate_financial_impact(results, arpu)

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
        results, p_values = platform.test_manager.get_test_results(test_id)

        export_data = {
            "test_id": test_id,
            "exported_at": datetime.now().isoformat(),
            "summary": platform._generate_summary(results, p_values),
            "detailed_results": {k: vars(v) for k, v in results.items()},
            "statistical_significance": p_values
        }

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
            }

        chart_data: List[Dict[str, Any]] = []
        for record in time_series_records:
            chart_data.append({
                "users_processed": record.users_processed,
                "variant": record.variant,
                "cumulative_metric": record.cumulative_metric,
                "mean_metric": record.mean_metric,
                "sample_size": record.sample_size,
                "p_value": record.p_value,
                "confidence_interval_lower": record.confidence_interval_lower,
                "confidence_interval_upper": record.confidence_interval_upper,
            })

        variants = sorted(list(set(record.variant for record in time_series_records)))
        latest_users = max(r.users_processed for r in time_series_records)
        latest_rows = [r for r in time_series_records if r.users_processed == latest_users]

        control_variant = variants[0] if variants else None
        control_mean = 0.0
        latest_by_variant: Dict[str, Any] = {}
        for row in latest_rows:
            latest_by_variant[row.variant] = row
            if row.variant == control_variant:
                control_mean = float(row.mean_metric)

        winner = None
        best_uplift = -1e18
        winner_confidence = "low"
        for v in variants:
            if v == control_variant:
                continue
            row = latest_by_variant.get(v)
            if not row:
                continue
            uplift = ((row.mean_metric - control_mean) / control_mean * 100.0) if control_mean != 0 else 0.0
            p_val = row.p_value if row.p_value is not None else 1.0

            # Победителя определяем только среди статистически значимых вариантов.
            if p_val < 0.05 and uplift > best_uplift:
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
                uplift = ((row.mean_metric - control_mean) / control_mean * 100.0) if control_mean != 0 else 0.0
                if uplift > observed_best:
                    observed_best = uplift
            best_uplift = observed_best if observed_best != -1e18 else 0.0

        variant_counts = {
            v: int(latest_by_variant[v].sample_size) if v in latest_by_variant else 0
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

        return {
            "test_id": test_id,
            "variants": variants,
            "data": chart_data,
            "total_snapshots": len(chart_data),
            "snapshots_per_variant": len(chart_data) // len(variants) if variants else 0,
            "completion_percentage": float(test.completion_percentage or 0.0),
            "stopped_early": bool(test.stopped_early),
            "early_stop_reason": test.early_stop_reason,
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
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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

    # без pandas: группируем в памяти
    segments: Dict[str, int] = {}
    for row in records:
        key = str(row.get(segment_by, "unknown"))
        segments[key] = segments.get(key, 0) + 1

    total = max(1, sum(segments.values()))
    return {
        "segments": {
            k: {
                "users": v,
                "share_percent": v / total * 100.0,
            }
            for k, v in segments.items()
        },
        "total_users": total,
    }


async def _calculate_financial_impact(results: Dict[str, Any], arpu: float) -> Dict[str, Any]:
    variants = list(results.keys())
    if not variants:
        return {"incremental_revenue": 0.0, "best_variant": None}

    control = results[variants[0]]
    control_mean = float(control.mean)
    control_n = int(control.sample_size)

    best_variant = variants[0]
    best_incremental = 0.0

    by_variant = {}
    for v in variants:
        stat = results[v]
        mean_val = float(stat.mean)
        n_val = int(stat.sample_size)
        uplift = ((mean_val - control_mean) / control_mean * 100.0) if control_mean != 0 else 0.0
        incremental = (mean_val - control_mean) * n_val * float(arpu)
        by_variant[v] = {
            "uplift_percent": uplift,
            "sample_size": n_val,
            "incremental_revenue": incremental,
        }
        if incremental > best_incremental:
            best_incremental = incremental
            best_variant = v

    return {
        "control_variant": variants[0],
        "best_variant": best_variant,
        "incremental_revenue": float(best_incremental),
        "by_variant": by_variant,
        "control_users": control_n,
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
    estimated_cost = 1000.0
    roi = ((incremental_revenue - estimated_cost) / estimated_cost) * 100.0 if estimated_cost else 0.0
    return {
        "estimated_cost": estimated_cost,
        "incremental_revenue": incremental_revenue,
        "roi_percent": roi,
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
