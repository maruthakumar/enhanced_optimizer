"""
GPU-Accelerated Calculations using cuDF
High-performance GPU calculations for portfolio optimization metrics
"""

import logging
from typing import List, Dict, Optional, Tuple
import numpy as np

# Import GPU utilities for centralized cuDF handling
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from gpu_utils import get_cudf_safe, get_cupy_safe, CUDF_AVAILABLE

# Get cuDF and cuPy safely
cudf = get_cudf_safe()
cp = get_cupy_safe()

logger = logging.getLogger(__name__)

def calculate_correlations_cudf(df: 'cudf.DataFrame', 
                               strategy_cols: List[str],
                               method: str = 'pearson') -> 'cudf.DataFrame':
    """
    Calculate correlation matrix using cuDF
    
    Args:
        df: cuDF DataFrame with strategy data
        strategy_cols: List of strategy column names
        method: Correlation method (pearson, spearman)
        
    Returns:
        Correlation matrix as cuDF DataFrame
    """
    if not CUDF_AVAILABLE:
        raise RuntimeError("cuDF is not available")
    
    try:
        logger.info(f"Calculating {method} correlations for {len(strategy_cols)} strategies")
        
        # Extract strategy data
        strategy_df = df[strategy_cols]
        
        if method == 'pearson':
            # Use cuDF's built-in correlation
            corr_matrix = strategy_df.corr(method='pearson')
        elif method == 'spearman':
            # For Spearman, rank the data first
            ranked_df = strategy_df.rank(method='average')
            corr_matrix = ranked_df.corr(method='pearson')
        else:
            raise ValueError(f"Unknown correlation method: {method}")
        
        logger.info(f"Correlation matrix computed: {corr_matrix.shape}")
        return corr_matrix
        
    except Exception as e:
        logger.error(f"Error calculating correlations: {str(e)}")
        raise

def calculate_fitness_cudf(df: 'cudf.DataFrame',
                          portfolio: List[str],
                          metrics_config: Dict[str, float]) -> Dict[str, float]:
    """
    Calculate fitness metrics for a portfolio using cuDF
    
    Args:
        df: cuDF DataFrame with strategy data
        portfolio: List of strategy names in portfolio
        metrics_config: Dictionary with metric weights
        
    Returns:
        Dictionary of calculated metrics
    """
    if not CUDF_AVAILABLE:
        raise RuntimeError("cuDF is not available")
    
    try:
        # Calculate portfolio returns (sum of selected strategies)
        portfolio_returns = df[portfolio].sum(axis=1)
        
        # Calculate individual metrics
        metrics = {}
        
        # Total ROI
        total_roi = portfolio_returns.sum()
        metrics['total_roi'] = float(total_roi)
        
        # Maximum Drawdown
        cumulative_returns = portfolio_returns.cumsum()
        running_max = cumulative_returns.cummax()
        drawdown = cumulative_returns - running_max
        max_drawdown = float(drawdown.min())
        metrics['max_drawdown'] = max_drawdown
        
        # Win Rate
        wins = (portfolio_returns > 0).sum()
        total_days = len(portfolio_returns)
        win_rate = float(wins) / total_days if total_days > 0 else 0
        metrics['win_rate'] = win_rate
        
        # Profit Factor
        gains = portfolio_returns[portfolio_returns > 0].sum()
        losses = -portfolio_returns[portfolio_returns < 0].sum()
        profit_factor = float(gains) / float(losses) if losses > 0 else float('inf')
        metrics['profit_factor'] = profit_factor
        
        # ROI/Drawdown Ratio
        roi_dd_ratio = total_roi / abs(max_drawdown) if max_drawdown != 0 else float('inf')
        metrics['roi_dd_ratio'] = roi_dd_ratio
        
        # Calculate weighted fitness score
        fitness = 0
        for metric, value in metrics.items():
            weight = metrics_config.get(f'{metric}_weight', 0)
            # Handle special cases
            if metric == 'max_drawdown':
                # For drawdown, less negative is better
                fitness += weight * (1 / (1 + abs(value)))
            elif metric == 'roi_dd_ratio' and value == float('inf'):
                fitness += weight * 1000  # Cap infinite values
            else:
                fitness += weight * value
        
        metrics['fitness_score'] = fitness
        
        logger.info(f"Calculated fitness for portfolio of {len(portfolio)} strategies: {fitness:.4f}")
        return metrics
        
    except Exception as e:
        logger.error(f"Error calculating fitness: {str(e)}")
        raise

def calculate_roi_cudf(returns: 'cudf.Series') -> float:
    """Calculate total ROI from returns series"""
    return float(returns.sum())

def calculate_drawdown_cudf(returns: 'cudf.Series') -> Tuple[float, 'cudf.Series']:
    """
    Calculate maximum drawdown and drawdown series
    
    Returns:
        Tuple of (max_drawdown, drawdown_series)
    """
    cumulative = returns.cumsum()
    running_max = cumulative.cummax()
    drawdown = cumulative - running_max
    max_drawdown = float(drawdown.min())
    return max_drawdown, drawdown

def calculate_win_rate_cudf(returns: 'cudf.Series') -> float:
    """Calculate win rate (percentage of positive returns)"""
    wins = (returns > 0).sum()
    total = len(returns)
    return float(wins) / total if total > 0 else 0

def calculate_profit_factor_cudf(returns: 'cudf.Series') -> float:
    """Calculate profit factor (gross profits / gross losses)"""
    gains = returns[returns > 0].sum()
    losses = -returns[returns < 0].sum()
    return float(gains) / float(losses) if losses > 0 else float('inf')

def calculate_sharpe_ratio_cudf(returns: 'cudf.Series', 
                               risk_free_rate: float = 0.02,
                               periods_per_year: int = 252) -> float:
    """
    Calculate Sharpe ratio
    
    Args:
        returns: Daily returns series
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of trading periods per year
        
    Returns:
        Sharpe ratio
    """
    if len(returns) < 2:
        return 0.0
    
    # Convert to daily risk-free rate
    daily_rf = risk_free_rate / periods_per_year
    
    # Calculate excess returns
    excess_returns = returns - daily_rf
    
    # Calculate Sharpe ratio
    mean_excess = float(excess_returns.mean())
    std_excess = float(excess_returns.std())
    
    if std_excess == 0:
        return 0.0
    
    # Annualize
    sharpe = (mean_excess / std_excess) * np.sqrt(periods_per_year)
    return sharpe

def calculate_sortino_ratio_cudf(returns: 'cudf.Series',
                                risk_free_rate: float = 0.02,
                                periods_per_year: int = 252) -> float:
    """
    Calculate Sortino ratio (uses downside deviation)
    
    Args:
        returns: Daily returns series
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of trading periods per year
        
    Returns:
        Sortino ratio
    """
    if len(returns) < 2:
        return 0.0
    
    # Convert to daily risk-free rate
    daily_rf = risk_free_rate / periods_per_year
    
    # Calculate excess returns
    excess_returns = returns - daily_rf
    
    # Calculate downside returns only
    downside_returns = excess_returns[excess_returns < 0]
    
    if len(downside_returns) == 0:
        return float('inf')  # No downside risk
    
    # Calculate downside deviation
    downside_std = float(downside_returns.std())
    
    if downside_std == 0:
        return float('inf')
    
    # Calculate Sortino ratio
    mean_excess = float(excess_returns.mean())
    sortino = (mean_excess / downside_std) * np.sqrt(periods_per_year)
    
    return sortino

def calculate_calmar_ratio_cudf(returns: 'cudf.Series',
                               periods_per_year: int = 252) -> float:
    """
    Calculate Calmar ratio (annualized return / max drawdown)
    
    Args:
        returns: Daily returns series
        periods_per_year: Number of trading periods per year
        
    Returns:
        Calmar ratio
    """
    if len(returns) < periods_per_year:
        logger.warning("Insufficient data for annual Calmar ratio calculation")
        periods_per_year = len(returns)
    
    # Calculate annualized return
    total_return = float(returns.sum())
    num_periods = len(returns)
    annualized_return = (total_return / num_periods) * periods_per_year
    
    # Calculate max drawdown
    max_dd, _ = calculate_drawdown_cudf(returns)
    
    if max_dd == 0:
        return float('inf')
    
    # Calmar ratio
    calmar = annualized_return / abs(max_dd)
    return calmar

def load_strategy_data_cudf(df: 'cudf.DataFrame',
                           strategy_columns: List[str],
                           date_range: Optional[Tuple[str, str]] = None) -> 'cudf.DataFrame':
    """
    Load and prepare strategy data for optimization
    
    Args:
        df: Full cuDF DataFrame
        strategy_columns: List of strategy columns to use
        date_range: Optional date range filter
        
    Returns:
        Filtered cuDF DataFrame
    """
    try:
        # Filter by date range if specified
        if date_range and 'Date' in df.columns:
            start_date, end_date = date_range
            mask = (df['Date'] >= start_date) & (df['Date'] <= end_date)
            df = df[mask]
            logger.info(f"Filtered to date range {start_date} to {end_date}: {len(df)} rows")
        
        # Select only required columns
        required_cols = ['Date'] + strategy_columns
        available_cols = [col for col in required_cols if col in df.columns]
        
        if len(available_cols) < len(required_cols):
            missing = set(required_cols) - set(available_cols)
            logger.warning(f"Missing columns: {missing}")
        
        result_df = df[available_cols]
        logger.info(f"Loaded strategy data: {len(result_df)} rows, {len(result_df.columns)} columns")
        
        return result_df
        
    except Exception as e:
        logger.error(f"Error loading strategy data: {str(e)}")
        raise