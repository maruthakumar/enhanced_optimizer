"""
Correlation Analysis Module for Heavy Optimizer Platform

This module provides GPU-accelerated correlation matrix calculations
and preserves the legacy correlation penalty logic for portfolio diversification.
"""

from .correlation_matrix_calculator import (
    CorrelationMatrixCalculator,
    CorrelationConfig
)

__all__ = [
    'CorrelationMatrixCalculator',
    'CorrelationConfig'
]

__version__ = '1.0.0'