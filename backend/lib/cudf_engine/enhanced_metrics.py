"""
Enhanced Financial Metrics using cuDF
Kelly Criterion, Market Regime, and Risk-Adjusted Returns
"""

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np

# Try to import cuDF
try:
    import cudf
    import cupy as cp
    CUDF_AVAILABLE = True
except ImportError:
    CUDF_AVAILABLE = False
    logging.warning("cuDF/cuPy not available for enhanced metrics")

logger = logging.getLogger(__name__)

def kelly_position_sizing(win_rate: float, 
                         avg_win: float, 
                         avg_loss: float,
                         max_allocation: float = 0.25) -> float:
    """
    Calculate optimal position size using Kelly Criterion
    
    Args:
        win_rate: Probability of winning (0-1)
        avg_win: Average winning return
        avg_loss: Average losing return (positive value)
        max_allocation: Maximum allowed allocation (safety cap)
        
    Returns:
        Optimal position size as fraction of capital
    """
    if avg_win <= 0 or avg_loss <= 0:
        return 0.0
    
    # Kelly formula: f = (p*b - q) / b
    # where p = win probability, q = loss probability, b = win/loss ratio
    loss_rate = 1 - win_rate
    win_loss_ratio = avg_win / avg_loss
    
    kelly_fraction = (win_rate * win_loss_ratio - loss_rate) / win_loss_ratio
    
    # Apply safety cap
    return max(0, min(kelly_fraction, max_allocation))

def calculate_kelly_allocation_cudf(returns: 'cudf.Series',
                                   capital: float = 1000000) -> Dict[str, float]:
    """
    Calculate Kelly Criterion allocation for a strategy
    
    Args:
        returns: cuDF Series of returns
        capital: Total capital available
        
    Returns:
        Dictionary with allocation details
    """
    if not CUDF_AVAILABLE:
        raise RuntimeError("cuDF not available")
    
    # Calculate win/loss statistics
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    
    if len(wins) == 0 or len(losses) == 0:
        return {
            'kelly_fraction': 0.0,
            'allocated_capital': 0.0,
            'reason': 'insufficient_win_loss_data'
        }
    
    win_rate = float(len(wins)) / float(len(returns))
    avg_win = float(wins.mean())
    avg_loss = float(-losses.mean())  # Convert to positive
    
    # Calculate Kelly fraction
    kelly_frac = kelly_position_sizing(win_rate, avg_win, avg_loss)
    
    return {
        'kelly_fraction': kelly_frac,
        'allocated_capital': capital * kelly_frac,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'expected_growth': win_rate * avg_win - (1 - win_rate) * avg_loss
    }

def regime_weighted_selection_cudf(df: 'cudf.DataFrame',
                                  confidence_threshold: float = 0.7,
                                  regime_column: str = 'Regime_Confidence_%') -> 'cudf.DataFrame':
    """
    Filter and weight strategies based on market regime confidence
    
    Args:
        df: cuDF DataFrame with regime data
        confidence_threshold: Minimum confidence for inclusion
        regime_column: Name of confidence column
        
    Returns:
        Filtered and weighted DataFrame
    """
    if not CUDF_AVAILABLE:
        raise RuntimeError("cuDF not available")
    
    if regime_column not in df.columns:
        logger.warning(f"Regime confidence column '{regime_column}' not found")
        return df
    
    # Filter by confidence threshold
    valid_mask = df[regime_column] >= confidence_threshold
    filtered_df = df[valid_mask].copy()
    
    # Add regime weight
    filtered_df['regime_weight'] = filtered_df[regime_column] / 100.0
    
    logger.info(f"Filtered {len(df)} to {len(filtered_df)} rows based on regime confidence")
    
    return filtered_df

def calculate_var_cvar_cudf(returns: 'cudf.Series',
                           confidence_levels: List[float] = [0.95, 0.99]) -> Dict[str, Dict[str, float]]:
    """
    Calculate Value at Risk (VaR) and Conditional VaR (CVaR)
    
    Args:
        returns: cuDF Series of returns
        confidence_levels: List of confidence levels (e.g., 0.95 for 95%)
        
    Returns:
        Dictionary with VaR and CVaR for each confidence level
    """
    if not CUDF_AVAILABLE:
        raise RuntimeError("cuDF not available")
    
    results = {}
    
    for confidence in confidence_levels:
        # Calculate quantile for VaR
        quantile = 1 - confidence
        var = float(returns.quantile(quantile))
        
        # Calculate CVaR (expected value below VaR)
        below_var = returns[returns <= var]
        cvar = float(below_var.mean()) if len(below_var) > 0 else var
        
        results[f'{int(confidence*100)}%'] = {
            'var': var,
            'cvar': cvar,
            'var_ratio': abs(var) / float(returns.std()) if returns.std() > 0 else 0
        }
    
    return results

def calculate_information_ratio_cudf(returns: 'cudf.Series',
                                    benchmark_returns: 'cudf.Series',
                                    periods_per_year: int = 252) -> float:
    """
    Calculate Information Ratio (active return / tracking error)
    
    Args:
        returns: Strategy returns
        benchmark_returns: Benchmark returns
        periods_per_year: Number of periods per year
        
    Returns:
        Information ratio
    """
    if not CUDF_AVAILABLE:
        raise RuntimeError("cuDF not available")
    
    if len(returns) != len(benchmark_returns):
        raise ValueError("Returns and benchmark must have same length")
    
    # Calculate active returns
    active_returns = returns - benchmark_returns
    
    # Calculate tracking error (std of active returns)
    tracking_error = float(active_returns.std())
    
    if tracking_error == 0:
        return 0.0
    
    # Annualize
    active_return_annual = float(active_returns.mean()) * periods_per_year
    tracking_error_annual = tracking_error * np.sqrt(periods_per_year)
    
    return active_return_annual / tracking_error_annual

def calculate_omega_ratio_cudf(returns: 'cudf.Series',
                              threshold: float = 0.0) -> float:
    """
    Calculate Omega Ratio (probability weighted ratio of gains vs losses)
    
    Args:
        returns: cuDF Series of returns
        threshold: Minimum acceptable return threshold
        
    Returns:
        Omega ratio
    """
    if not CUDF_AVAILABLE:
        raise RuntimeError("cuDF not available")
    
    # Calculate gains and losses relative to threshold
    excess_returns = returns - threshold
    gains = excess_returns[excess_returns > 0]
    losses = -excess_returns[excess_returns < 0]
    
    if len(losses) == 0 or losses.sum() == 0:
        return float('inf')
    
    if len(gains) == 0:
        return 0.0
    
    return float(gains.sum()) / float(losses.sum())

def zone_based_optimization_cudf(df: 'cudf.DataFrame',
                                zone_column: str = 'zone',
                                strategy_columns: List[str] = None) -> Dict[str, 'cudf.DataFrame']:
    """
    Optimize strategies based on intraday zones
    
    Args:
        df: cuDF DataFrame with zone data
        zone_column: Name of zone column
        strategy_columns: List of strategy columns
        
    Returns:
        Dictionary of DataFrames by zone
    """
    if not CUDF_AVAILABLE:
        raise RuntimeError("cuDF not available")
    
    if zone_column not in df.columns:
        logger.warning(f"Zone column '{zone_column}' not found")
        return {'all': df}
    
    # Get unique zones
    zones = df[zone_column].unique().to_pandas()
    
    zone_data = {}
    for zone in zones:
        zone_df = df[df[zone_column] == zone]
        
        # Calculate zone-specific metrics
        if strategy_columns:
            zone_returns = zone_df[strategy_columns].sum(axis=1)
            
            zone_data[zone] = {
                'data': zone_df,
                'count': len(zone_df),
                'avg_return': float(zone_returns.mean()),
                'volatility': float(zone_returns.std()),
                'sharpe': calculate_sharpe_ratio_cudf(zone_returns) if len(zone_returns) > 1 else 0
            }
        else:
            zone_data[zone] = {
                'data': zone_df,
                'count': len(zone_df)
            }
    
    logger.info(f"Split data into {len(zones)} zones")
    return zone_data

def calculate_enhanced_fitness_cudf(df: 'cudf.DataFrame',
                                   portfolio: List[str],
                                   config: Dict) -> Dict[str, float]:
    """
    Calculate enhanced fitness metrics including Sharpe, Sortino, etc.
    
    Args:
        df: cuDF DataFrame with strategy data
        portfolio: List of strategy columns
        config: Configuration with parameters
        
    Returns:
        Dictionary of all metrics
    """
    if not CUDF_AVAILABLE:
        raise RuntimeError("cuDF not available")
    
    # Import base calculations
    from .gpu_calculator import (
        calculate_fitness_cudf,
        calculate_sharpe_ratio_cudf,
        calculate_sortino_ratio_cudf,
        calculate_calmar_ratio_cudf
    )
    
    # Get base metrics
    base_metrics = calculate_fitness_cudf(df, portfolio, config.get('fitness_weights', {}))
    
    # Calculate portfolio returns
    portfolio_returns = df[portfolio].sum(axis=1)
    
    # Add enhanced metrics
    if config.get('calculate_sharpe', True):
        base_metrics['sharpe_ratio'] = calculate_sharpe_ratio_cudf(
            portfolio_returns,
            risk_free_rate=config.get('risk_free_rate', 0.02),
            periods_per_year=config.get('periods_per_year', 252)
        )
    
    if config.get('calculate_sortino', True):
        base_metrics['sortino_ratio'] = calculate_sortino_ratio_cudf(
            portfolio_returns,
            risk_free_rate=config.get('risk_free_rate', 0.02),
            periods_per_year=config.get('periods_per_year', 252)
        )
    
    if config.get('calculate_calmar', True):
        base_metrics['calmar_ratio'] = calculate_calmar_ratio_cudf(
            portfolio_returns,
            periods_per_year=config.get('periods_per_year', 252)
        )
    
    # Calculate Kelly allocation
    if config.get('calculate_kelly', True):
        kelly_allocation = calculate_kelly_allocation_cudf(
            portfolio_returns,
            capital=config.get('capital', 1000000)
        )
        base_metrics['kelly_fraction'] = kelly_allocation['kelly_fraction']
        base_metrics['kelly_allocated_capital'] = kelly_allocation['allocated_capital']
    
    # Calculate VaR/CVaR
    if config.get('calculate_var', True):
        var_cvar = calculate_var_cvar_cudf(portfolio_returns)
        base_metrics['var_95'] = var_cvar['95%']['var']
        base_metrics['cvar_95'] = var_cvar['95%']['cvar']
        base_metrics['var_99'] = var_cvar['99%']['var']
        base_metrics['cvar_99'] = var_cvar['99%']['cvar']
    
    # Calculate Omega ratio
    if config.get('calculate_omega', True):
        base_metrics['omega_ratio'] = calculate_omega_ratio_cudf(portfolio_returns)
    
    return base_metrics