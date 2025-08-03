"""
cuDF Engine Module
GPU-accelerated calculations using cuDF DataFrames
"""

from .gpu_calculator import (
    calculate_correlations_cudf,
    calculate_fitness_cudf,
    calculate_roi_cudf,
    calculate_drawdown_cudf,
    calculate_win_rate_cudf,
    calculate_profit_factor_cudf,
    calculate_sharpe_ratio_cudf,
    calculate_sortino_ratio_cudf,
    calculate_calmar_ratio_cudf
)

__all__ = [
    'calculate_correlations_cudf',
    'calculate_fitness_cudf',
    'calculate_roi_cudf',
    'calculate_drawdown_cudf',
    'calculate_win_rate_cudf',
    'calculate_profit_factor_cudf',
    'calculate_sharpe_ratio_cudf',
    'calculate_sortino_ratio_cudf',
    'calculate_calmar_ratio_cudf'
]