"""
Enhanced financial metrics including Sharpe, Sortino, and Calmar ratios.
GPU-accelerated calculations using cuDF.
"""
import numpy as np
import logging
from typing import Dict, Union, Optional, Tuple

# Try to import GPU libraries
try:
    import cudf
    import cupy as cp
    GPU_AVAILABLE = True
except (ImportError, RuntimeError):
    GPU_AVAILABLE = False
    # Create mock modules for type hints
    class cudf:
        DataFrame = dict
        Series = list
    class cp:
        ndarray = np.ndarray

logger = logging.getLogger(__name__)


class EnhancedMetrics:
    """
    Advanced financial metrics calculator for risk-adjusted return optimization.
    """
    
    def __init__(self, risk_free_rate: float = 0.02, annualization_factor: int = 252):
        """
        Initialize enhanced metrics calculator.
        
        Args:
            risk_free_rate: Annual risk-free rate (default 2%)
            annualization_factor: Number of trading days per year (default 252)
        """
        self.risk_free_rate = risk_free_rate
        self.annualization_factor = annualization_factor
        self.daily_risk_free = risk_free_rate / annualization_factor
        
    def calculate_sharpe_ratio(self, 
                             returns: Union[cudf.Series, cp.ndarray],
                             risk_free_rate: Optional[float] = None) -> float:
        """
        Calculate Sharpe ratio (risk-adjusted returns using standard deviation).
        
        Sharpe = (Return - Risk Free Rate) / Standard Deviation
        
        Args:
            returns: Series or array of returns
            risk_free_rate: Optional override for risk-free rate
            
        Returns:
            Annualized Sharpe ratio
        """
        if risk_free_rate is None:
            risk_free_rate = self.daily_risk_free
            
        if GPU_AVAILABLE and hasattr(returns, 'values'):
            excess_returns = returns - risk_free_rate
            mean_excess = float(excess_returns.mean())
            std_returns = float(returns.std())
        else:
            returns_gpu = cp.asarray(returns)
            excess_returns = returns_gpu - risk_free_rate
            mean_excess = float(cp.mean(excess_returns))
            std_returns = float(cp.std(returns_gpu))
            
        if std_returns == 0:
            return 0.0
            
        # Annualize
        sharpe_ratio = mean_excess / std_returns * np.sqrt(self.annualization_factor)
        
        return sharpe_ratio
        
    def calculate_sortino_ratio(self,
                              returns: Union[cudf.Series, cp.ndarray],
                              mar: float = 0.0) -> float:
        """
        Calculate Sortino ratio (downside risk-adjusted returns).
        
        Sortino = (Return - MAR) / Downside Deviation
        
        Args:
            returns: Series or array of returns
            mar: Minimum Acceptable Return (default 0)
            
        Returns:
            Annualized Sortino ratio
        """
        if GPU_AVAILABLE and hasattr(returns, 'values'):
            excess_returns = returns - mar
            mean_excess = float(excess_returns.mean())
            
            # Calculate downside deviation (only negative returns)
            downside_returns = returns[returns < mar]
            if len(downside_returns) == 0:
                return float('inf')  # No downside risk
                
            downside_dev = float(downside_returns.std())
        else:
            returns_gpu = cp.asarray(returns)
            excess_returns = returns_gpu - mar
            mean_excess = float(cp.mean(excess_returns))
            
            # Downside deviation
            downside_mask = returns_gpu < mar
            downside_returns = returns_gpu[downside_mask]
            if len(downside_returns) == 0:
                return float('inf')
                
            downside_dev = float(cp.std(downside_returns))
            
        if downside_dev == 0:
            return float('inf')
            
        # Annualize
        sortino_ratio = mean_excess / downside_dev * np.sqrt(self.annualization_factor)
        
        return sortino_ratio
        
    def calculate_calmar_ratio(self,
                             returns: Union[cudf.Series, cp.ndarray],
                             max_drawdown: Optional[float] = None) -> float:
        """
        Calculate Calmar ratio (return to drawdown ratio).
        
        Calmar = Annualized Return / Maximum Drawdown
        
        Args:
            returns: Series or array of returns
            max_drawdown: Pre-calculated max drawdown (optional)
            
        Returns:
            Calmar ratio
        """
        if GPU_AVAILABLE and hasattr(returns, 'values'):
            total_return = float((1 + returns).prod() - 1)
            mean_return = float(returns.mean())
        else:
            returns_gpu = cp.asarray(returns)
            total_return = float(cp.prod(1 + returns_gpu) - 1)
            mean_return = float(cp.mean(returns_gpu))
            
        # Annualize return
        periods = len(returns)
        years = periods / self.annualization_factor
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else total_return
        
        # Calculate max drawdown if not provided
        if max_drawdown is None:
            max_drawdown = self.calculate_max_drawdown(returns)
            
        if abs(max_drawdown) < 0.0001:  # Avoid division by zero
            return float('inf') if annualized_return > 0 else 0.0
            
        calmar_ratio = annualized_return / abs(max_drawdown)
        
        return calmar_ratio
        
    def calculate_max_drawdown(self, 
                             returns: Union[cudf.Series, cp.ndarray]) -> float:
        """
        Calculate maximum drawdown from returns series.
        
        Args:
            returns: Series or array of returns
            
        Returns:
            Maximum drawdown (negative value)
        """
        if GPU_AVAILABLE and hasattr(returns, 'values'):
            # Calculate cumulative returns
            cum_returns = (1 + returns).cumprod()
            
            # Calculate running maximum
            running_max = cum_returns.cummax()
            
            # Calculate drawdown series
            drawdown = (cum_returns - running_max) / running_max
            
            # Get maximum drawdown
            max_dd = float(drawdown.min())
        else:
            returns_gpu = cp.asarray(returns)
            cum_returns = cp.cumprod(1 + returns_gpu)
            running_max = cp.maximum.accumulate(cum_returns)
            drawdown = (cum_returns - running_max) / running_max
            max_dd = float(cp.min(drawdown))
            
        return max_dd
        
    def calculate_information_ratio(self,
                                  returns: cudf.Series,
                                  benchmark_returns: cudf.Series) -> float:
        """
        Calculate Information Ratio (active return to tracking error).
        
        IR = (Portfolio Return - Benchmark Return) / Tracking Error
        
        Args:
            returns: Portfolio returns
            benchmark_returns: Benchmark returns
            
        Returns:
            Information ratio
        """
        active_returns = returns - benchmark_returns
        mean_active = float(active_returns.mean())
        tracking_error = float(active_returns.std())
        
        if tracking_error == 0:
            return 0.0
            
        # Annualize
        ir = mean_active / tracking_error * np.sqrt(self.annualization_factor)
        
        return ir
        
    def calculate_omega_ratio(self,
                            returns: Union[cudf.Series, cp.ndarray],
                            threshold: float = 0.0) -> float:
        """
        Calculate Omega ratio (probability of gains vs losses).
        
        Omega = Sum of returns above threshold / Sum of returns below threshold
        
        Args:
            returns: Series or array of returns
            threshold: Return threshold (default 0)
            
        Returns:
            Omega ratio
        """
        if GPU_AVAILABLE and hasattr(returns, 'values'):
            gains = returns[returns > threshold] - threshold
            losses = threshold - returns[returns <= threshold]
            
            sum_gains = float(gains.sum()) if len(gains) > 0 else 0.0
            sum_losses = float(losses.sum()) if len(losses) > 0 else 0.0
        else:
            returns_gpu = cp.asarray(returns)
            gains_mask = returns_gpu > threshold
            losses_mask = returns_gpu <= threshold
            
            gains = returns_gpu[gains_mask] - threshold
            losses = threshold - returns_gpu[losses_mask]
            
            sum_gains = float(cp.sum(gains)) if cp.any(gains_mask) else 0.0
            sum_losses = float(cp.sum(losses)) if cp.any(losses_mask) else 0.0
            
        if sum_losses == 0:
            return float('inf') if sum_gains > 0 else 1.0
            
        omega_ratio = sum_gains / sum_losses
        
        return omega_ratio
        
    def calculate_all_metrics(self,
                            returns: Union[cudf.Series, cp.ndarray],
                            benchmark_returns: Optional[cudf.Series] = None,
                            mar: float = 0.0) -> Dict[str, float]:
        """
        Calculate all enhanced financial metrics.
        
        Args:
            returns: Series or array of returns
            benchmark_returns: Optional benchmark returns for Information Ratio
            mar: Minimum Acceptable Return for Sortino
            
        Returns:
            Dictionary with all calculated metrics
        """
        metrics = {
            'sharpe_ratio': self.calculate_sharpe_ratio(returns),
            'sortino_ratio': self.calculate_sortino_ratio(returns, mar),
            'max_drawdown': self.calculate_max_drawdown(returns),
            'omega_ratio': self.calculate_omega_ratio(returns)
        }
        
        # Add Calmar ratio using calculated max drawdown
        metrics['calmar_ratio'] = self.calculate_calmar_ratio(returns, metrics['max_drawdown'])
        
        # Add Information Ratio if benchmark provided
        if benchmark_returns is not None:
            metrics['information_ratio'] = self.calculate_information_ratio(returns, benchmark_returns)
            
        # Add basic statistics
        if GPU_AVAILABLE and hasattr(returns, 'values'):
            metrics['total_return'] = float((1 + returns).prod() - 1)
            metrics['annualized_return'] = (1 + metrics['total_return']) ** (self.annualization_factor / len(returns)) - 1
            metrics['volatility'] = float(returns.std()) * np.sqrt(self.annualization_factor)
            metrics['skewness'] = float(returns.skew())
            metrics['kurtosis'] = float(returns.kurtosis())
        else:
            returns_gpu = cp.asarray(returns)
            metrics['total_return'] = float(cp.prod(1 + returns_gpu) - 1)
            metrics['annualized_return'] = (1 + metrics['total_return']) ** (self.annualization_factor / len(returns)) - 1
            metrics['volatility'] = float(cp.std(returns_gpu)) * np.sqrt(self.annualization_factor)
            
        return metrics
        
    def calculate_portfolio_metrics(self,
                                  portfolio_returns: cudf.DataFrame,
                                  weights: cudf.Series,
                                  benchmark_returns: Optional[cudf.Series] = None) -> Dict[str, float]:
        """
        Calculate metrics for a weighted portfolio.
        
        Args:
            portfolio_returns: DataFrame with returns for each asset
            weights: Series with portfolio weights
            benchmark_returns: Optional benchmark returns
            
        Returns:
            Dictionary with portfolio metrics
        """
        # Calculate weighted portfolio returns
        portfolio_return_series = (portfolio_returns * weights).sum(axis=1)
        
        return self.calculate_all_metrics(portfolio_return_series, benchmark_returns)


def create_enhanced_metrics_config(config_dict: Dict) -> Dict:
    """
    Create enhanced metrics configuration from config dictionary.
    
    Args:
        config_dict: Configuration dictionary
        
    Returns:
        Enhanced metrics configuration
    """
    metrics_config = {
        'sharpe_enabled': 'sharpe' in config_dict.get('FITNESS_CALCULATION', {}).get('enhanced_metrics', 'sharpe,sortino,calmar,kelly'),
        'sortino_enabled': 'sortino' in config_dict.get('FITNESS_CALCULATION', {}).get('enhanced_metrics', 'sharpe,sortino,calmar,kelly'),
        'calmar_enabled': 'calmar' in config_dict.get('FITNESS_CALCULATION', {}).get('enhanced_metrics', 'sharpe,sortino,calmar,kelly'),
        'risk_free_rate': float(config_dict.get('RISK_METRICS', {}).get('risk_free_rate', 0.02)),
        'mar': float(config_dict.get('RISK_METRICS', {}).get('minimum_acceptable_return', 0.0)),
        'annualization_factor': int(config_dict.get('RISK_METRICS', {}).get('annualization_factor', 252))
    }
    
    return metrics_config