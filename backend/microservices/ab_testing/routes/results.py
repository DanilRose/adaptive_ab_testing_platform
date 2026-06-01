# backend/api/routes/results.py

from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
from datetime import datetime

from backend.microservices.ab_testing.service import ABPlatformProvider
from backend.microservices.auth_core.models import User
from backend.microservices.auth_core.service import get_current_user
from backend.microservices.ab_testing.services.results_analytics import ResultsAnalyticsService

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
        return ResultsAnalyticsService.get_detailed_results(test_id=test_id, platform=platform)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{test_id}/statistical-significance", summary="Статистическая значимость")
async def get_statistical_significance(test_id: str, alpha: float = 0.05, current_user: User = Depends(get_current_user)):
    return ResultsAnalyticsService.get_statistical_significance(
        test_id=test_id,
        alpha=alpha,
        platform=platform,
    )


@router.get("/{test_id}/time-series", summary="Временные ряды метрик")
async def get_time_series_data(
    test_id: str,
    time_window: str = Query("7d", pattern="^(1d|7d|30d|all)$"),
    current_user: User = Depends(get_current_user)
):
    try:
        time_series_data = ResultsAnalyticsService.get_time_series_data(test_id=test_id, time_window=time_window)
        return {
            "test_id": test_id,
            "time_window": time_window,
            "time_series": time_series_data,
            "trend_analysis": ResultsAnalyticsService.analyze_trends(time_series_data=time_series_data),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{test_id}/segmentation", summary="Анализ по сегментам")
async def get_segmentation_analysis(test_id: str, segment_by: str = "user_type", current_user: User = Depends(get_current_user)):
    try:
        segments_analysis = ResultsAnalyticsService.get_segmentation_analysis(test_id=test_id, segment_by=segment_by)
        return {
            "test_id": test_id,
            "segment_by": segment_by,
            "segments_analysis": segments_analysis,
            "segment_comparison": ResultsAnalyticsService.compare_segments(segments_analysis=segments_analysis),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




@router.get("/platform/performance", summary="Производительность платформы")
async def get_platform_performance(days: int = 30, current_user: User = Depends(get_current_user)):
    try:
        platform_stats = platform.get_platform_stats()
        test_history = platform.test_registry.get_test_history(limit=100)

        performance_metrics = ResultsAnalyticsService.calculate_performance_metrics(
            test_history=test_history,
            days=days,
        )

        return {
            "time_period_days": days,
            "platform_stats": platform_stats,
            "performance_metrics": performance_metrics,
            "recommendations": ResultsAnalyticsService.generate_performance_recommendations(
                performance_metrics=performance_metrics,
            ),
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
    return ResultsAnalyticsService.get_time_series_chart_data(test_id=test_id, platform=platform)


def _convert_to_csv(data: Dict[str, Any]) -> str:
    return "test_id,exported_at\n" + f"{data.get('test_id')},{data.get('exported_at')}\n"


async def _convert_to_excel(data: Dict[str, Any]) -> str:
    return "Excel export placeholder"
