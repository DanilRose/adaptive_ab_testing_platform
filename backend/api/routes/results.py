# backend/api/routes/results.py

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
import scipy.stats as stats
from datetime import datetime, timedelta
import json

from backend.ab_testing.managers import AdaptiveABTestingPlatform

router = APIRouter(prefix="/api/v1/results", tags=["Results & Analytics"])

platform = AdaptiveABTestingPlatform()

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

class TimeSeriesRequest(BaseModel):
    test_id: str
    time_window: str = Field("7d", pattern="^(1d|7d|30d|all)$")
    metric: Optional[str] = None

@router.get("/{test_id}/detailed", summary="Детальные результаты теста")
async def get_detailed_results(test_id: str):
    try:
        results = platform.get_test_results(test_id)
        
        # Дополнительный статистический анализ
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
async def get_statistical_significance(test_id: str, alpha: float = 0.05):
    try:
        results, p_values = platform.test_manager.get_test_results(test_id)
        
        significance_analysis = {}
        control_variant = list(results.keys())[0]
        control_data = results[control_variant]
        
        for variant, p_value in p_values.items():
            variant_data = results[variant]
            
            # Расчет мощности теста
            power = await _calculate_power(control_data, variant_data, alpha)
            
            # Расчет доверительных интервалов для разницы
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
async def get_time_series_data(request: TimeSeriesRequest):
    try:
        # В реальном приложении здесь была бы логика получения временных рядов из БД
        # Сейчас генерируем синтетические данные для демонстрации
        
        time_series_data = await _generate_time_series_data(request.test_id, request.time_window)
        
        return {
            "test_id": request.test_id,
            "time_window": request.time_window,
            "time_series": time_series_data,
            "trend_analysis": await _analyze_trends(time_series_data)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{test_id}/segmentation", summary="Анализ по сегментам")
async def get_segmentation_analysis(test_id: str, segment_by: str = "user_type"):
    try:
        # В реальном приложении здесь была бы логика сегментации из БД
        # Сейчас генерируем синтетические сегменты
        
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
async def get_financial_impact(test_id: str, arpu: float = 100.0):
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
async def get_platform_performance(days: int = 30):
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
    format: str = Query("json", regex="^(json|csv|excel)$"),
    include_raw_data: bool = False
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
        elif format == "csv":
            # Конвертация в CSV формат
            csv_data = await _convert_to_csv(export_data)
            return {"csv_data": csv_data}
        elif format == "excel":
            # Логика для Excel экспорта
            excel_data = await _convert_to_excel(export_data)
            return {"excel_data": "Base64 encoded Excel file"}  # Упрощенно
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Вспомогательные методы
async def _perform_detailed_analysis(results: Dict[str, Any]) -> Dict[str, Any]:
    """Выполняет детальный статистический анализ результатов"""
    basic_results = results.get('results', {})
    control_variant = list(basic_results.keys())[0] if basic_results else None
    
    analysis = {}
    for variant, result_data in basic_results.items():
        if variant == control_variant:
            continue
            
        # Расчет различных метрик
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
    """Расчет мощности теста"""
    try:
        # Упрощенный расчет мощности
        effect_size = abs(variant_data.mean - control_data.mean)
        pooled_std = np.sqrt((control_data.std**2 + variant_data.std**2) / 2)
        
        if pooled_std == 0:
            return 0.0
            
        standardized_effect = effect_size / pooled_std
        power = stats.norm.ppf(1 - alpha/2) + stats.norm.ppf(0.8)
        
        return min(1.0, max(0.0, stats.norm.cdf(standardized_effect * np.sqrt(variant_data.sample_size/2) - power)))
    except:
        return 0.0

async def _calculate_effect_confidence_interval(control_data: Any, variant_data: Any) -> List[float]:
    """Расчет доверительного интервала для размера эффекта"""
    try:
        mean_diff = variant_data.mean - control_data.mean
        se_diff = np.sqrt(
            (control_data.std**2 / control_data.sample_size) + 
            (variant_data.std**2 / variant_data.sample_size)
        )
        
        t_critical = stats.t.ppf(0.975, min(control_data.sample_size, variant_data.sample_size) - 1)
        margin = t_critical * se_diff
        
        return [float(mean_diff - margin), float(mean_diff + margin)]
    except:
        return [0.0, 0.0]

async def _calculate_required_sample_size(control_data: Any, variant_data: Any, alpha: float, power: float) -> int:
    """Расчет требуемого размера выборки"""
    try:
        effect_size = abs(variant_data.mean - control_data.mean)
        pooled_std = np.sqrt((control_data.std**2 + variant_data.std**2) / 2)
        
        if pooled_std == 0:
            return 0
            
        standardized_effect = effect_size / pooled_std
        z_alpha = stats.norm.ppf(1 - alpha/2)
        z_beta = stats.norm.ppf(power)
        
        required_n = 2 * ((z_alpha + z_beta) / standardized_effect) ** 2
        return int(np.ceil(required_n))
    except:
        return 0

def _interpret_significance(significance_analysis: Dict[str, Any], alpha: float) -> Dict[str, str]:
    """Интерпретация статистической значимости"""
    significant_variants = [
        variant for variant, analysis in significance_analysis.items() 
        if analysis['statistically_significant']
    ]
    
    if not significant_variants:
        return {
            "conclusion": "Нет статистически значимых различий",
            "recommendation": "Продолжить сбор данных или рассмотреть другие метрики"
        }
    
    best_variant = max(significant_variants, 
                      key=lambda v: significance_analysis[v]['effect_size'])
    
    return {
        "conclusion": f"Статистически значимые различия обнаружены для вариантов: {', '.join(significant_variants)}",
        "best_performing": best_variant,
        "recommendation": f"Рекомендуется внедрить вариант {best_variant}"
    }

# Заглушки для методов, которые требуют интеграции с БД
async def _generate_time_series_data(test_id: str, time_window: str) -> Dict[str, Any]:
    return {"synthetic": True, "message": "Time series data would come from database"}

async def _generate_segmentation_analysis(test_id: str, segment_by: str) -> Dict[str, Any]:
    return {"synthetic": True, "message": "Segmentation analysis would come from database"}

async def _calculate_financial_impact(results: Dict[str, Any], arpu: float) -> Dict[str, Any]:
    return {"synthetic": True, "message": "Financial impact calculation"}

async def _convert_to_csv(data: Dict[str, Any]) -> str:
    return "CSV data would be generated here"

async def _convert_to_excel(data: Dict[str, Any]) -> str:
    return "Excel data would be generated here"

async def _analyze_trends(time_series_data: Dict[str, Any]) -> Dict[str, Any]:
    return {"trend_analysis": "Would analyze trends in time series data"}

async def _compare_segments(segments_analysis: Dict[str, Any]) -> Dict[str, Any]:
    return {"segment_comparison": "Would compare segments"}

async def _calculate_roi(financial_analysis: Dict[str, Any]) -> Dict[str, Any]:
    return {"roi_calculation": "Would calculate ROI"}

async def _calculate_performance_metrics(test_history: List[Dict[str, Any]], days: int) -> Dict[str, Any]:
    return {"performance_metrics": "Would calculate performance metrics"}

async def _generate_performance_recommendations(performance_metrics: Dict[str, Any]) -> List[str]:
    return ["Recommendation 1", "Recommendation 2"]

def _calculate_relative_improvement(control_mean: float, variant_mean: float) -> float:
    """Расчет относительного улучшения"""
    if control_mean == 0:
        return 0.0
    return ((variant_mean - control_mean) / control_mean) * 100