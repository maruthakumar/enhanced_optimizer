"""
Advanced Analytics Module for Heavy Optimizer Platform

Provides comprehensive analytics for portfolio optimization results including:
- Performance attribution analysis
- Sensitivity analysis 
- Scenario modeling
- Diversification analysis
- Production risk metrics
- Advanced visualizations
"""

from .performance_attribution import PerformanceAttributionAnalyzer, AttributionConfig
from .sensitivity_analysis import SensitivityAnalyzer, SensitivityConfig
from .scenario_modeling import ScenarioModeler, ScenarioConfig
from .diversification_analysis import DiversificationAnalyzer, DiversificationConfig
from .risk_metrics import RiskMetricsCalculator, RiskMetricsConfig
from .analytics_visualizer import AnalyticsVisualizer
from .analytics_exporter import AnalyticsExporter
from .advanced_analytics_engine import AdvancedAnalyticsEngine

__all__ = [
    'PerformanceAttributionAnalyzer', 'AttributionConfig',
    'SensitivityAnalyzer', 'SensitivityConfig',
    'ScenarioModeler', 'ScenarioConfig', 
    'DiversificationAnalyzer', 'DiversificationConfig',
    'RiskMetricsCalculator', 'RiskMetricsConfig',
    'AnalyticsVisualizer',
    'AnalyticsExporter',
    'AdvancedAnalyticsEngine'
]

__version__ = '1.0.0'