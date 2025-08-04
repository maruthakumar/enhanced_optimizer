"""
Risk management package for VaR/CVaR calculations.
"""
from .var_cvar_calculator import VaRCVaRCalculator, create_risk_config

__all__ = [
    'VaRCVaRCalculator',
    'create_risk_config'
]