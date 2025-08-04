"""
Value at Risk (VaR) and Conditional Value at Risk (CVaR) calculations with GPU acceleration.
"""
import numpy as np
import logging
from typing import Tuple, List, Dict, Union, Optional

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
        def asarray(x): return np.asarray(x)
        def mean(x): return np.mean(x)
        def std(x): return np.std(x)
        def percentile(x, p): return np.percentile(x, p)

logger = logging.getLogger(__name__)


class VaRCVaRCalculator:
    """
    GPU-accelerated VaR and CVaR calculations for portfolio risk assessment.
    
    Supports both historical and parametric methods for risk calculation.
    """
    
    def __init__(self, confidence_levels: List[float] = [0.95, 0.99]):
        """
        Initialize VaR/CVaR calculator.
        
        Args:
            confidence_levels: List of confidence levels for calculations (e.g., 0.95 = 95%)
        """
        self.confidence_levels = confidence_levels
        
    def calculate_var_historical(self, 
                               returns: Union[cudf.Series, cp.ndarray], 
                               confidence: float = 0.95) -> float:
        """
        Calculate Value at Risk using historical method.
        
        Args:
            returns: Series or array of returns
            confidence: Confidence level (e.g., 0.95 for 95% VaR)
            
        Returns:
            VaR value (negative number representing potential loss)
        """
        if GPU_AVAILABLE and hasattr(returns, 'values'):
            returns_gpu = returns.values
        else:
            returns_gpu = cp.asarray(returns) if GPU_AVAILABLE else np.asarray(returns)
            
        # Calculate percentile (left tail)
        alpha = 1 - confidence
        if GPU_AVAILABLE and not isinstance(returns_gpu, np.ndarray):
            var_value = float(cp.percentile(returns_gpu, alpha * 100))
        else:
            var_value = float(np.percentile(returns_gpu, alpha * 100))
        
        return var_value
        
    def calculate_cvar_historical(self, 
                                returns: Union[cudf.Series, cp.ndarray], 
                                confidence: float = 0.95) -> float:
        """
        Calculate Conditional Value at Risk (Expected Shortfall) using historical method.
        
        CVaR represents the expected loss given that the loss exceeds VaR.
        
        Args:
            returns: Series or array of returns
            confidence: Confidence level
            
        Returns:
            CVaR value (negative number representing expected loss beyond VaR)
        """
        var_threshold = self.calculate_var_historical(returns, confidence)
        
        if isinstance(returns, cudf.Series):
            # Filter returns worse than VaR
            tail_returns = returns[returns <= var_threshold]
            if len(tail_returns) == 0:
                return var_threshold
            cvar_value = float(tail_returns.mean())
        else:
            returns_gpu = cp.asarray(returns)
            tail_returns = returns_gpu[returns_gpu <= var_threshold]
            if len(tail_returns) == 0:
                return var_threshold
            cvar_value = float(cp.mean(tail_returns))
            
        return cvar_value
        
    def calculate_var_parametric(self, 
                               returns: Union[cudf.Series, cp.ndarray],
                               confidence: float = 0.95) -> float:
        """
        Calculate VaR using parametric (variance-covariance) method.
        
        Assumes returns follow normal distribution.
        
        Args:
            returns: Series or array of returns
            confidence: Confidence level
            
        Returns:
            VaR value
        """
        if isinstance(returns, cudf.Series):
            mean_return = float(returns.mean())
            std_return = float(returns.std())
        else:
            returns_gpu = cp.asarray(returns)
            mean_return = float(cp.mean(returns_gpu))
            std_return = float(cp.std(returns_gpu))
            
        # Z-score for given confidence level
        from scipy.stats import norm
        z_score = norm.ppf(1 - confidence)
        
        var_value = mean_return + z_score * std_return
        
        return var_value
        
    def calculate_portfolio_var(self, 
                              portfolio_returns: cudf.DataFrame,
                              weights: cudf.Series,
                              confidence: float = 0.95,
                              method: str = 'historical') -> float:
        """
        Calculate portfolio VaR considering correlations.
        
        Args:
            portfolio_returns: DataFrame with returns for each asset
            weights: Series with portfolio weights
            confidence: Confidence level
            method: 'historical' or 'parametric'
            
        Returns:
            Portfolio VaR
        """
        # Calculate weighted portfolio returns
        portfolio_return_series = (portfolio_returns * weights).sum(axis=1)
        
        if method == 'historical':
            return self.calculate_var_historical(portfolio_return_series, confidence)
        else:
            return self.calculate_var_parametric(portfolio_return_series, confidence)
            
    def calculate_portfolio_cvar(self, 
                               portfolio_returns: cudf.DataFrame,
                               weights: cudf.Series,
                               confidence: float = 0.95) -> float:
        """
        Calculate portfolio CVaR.
        
        Args:
            portfolio_returns: DataFrame with returns for each asset
            weights: Series with portfolio weights  
            confidence: Confidence level
            
        Returns:
            Portfolio CVaR
        """
        # Calculate weighted portfolio returns
        portfolio_return_series = (portfolio_returns * weights).sum(axis=1)
        
        return self.calculate_cvar_historical(portfolio_return_series, confidence)
        
    def calculate_marginal_var(self,
                             portfolio_returns: cudf.DataFrame,
                             weights: cudf.Series,
                             confidence: float = 0.95) -> cudf.Series:
        """
        Calculate marginal VaR contribution of each asset.
        
        Marginal VaR shows how portfolio VaR changes with small changes in position.
        
        Args:
            portfolio_returns: DataFrame with returns for each asset
            weights: Series with portfolio weights
            confidence: Confidence level
            
        Returns:
            Series with marginal VaR for each asset
        """
        portfolio_var = self.calculate_portfolio_var(portfolio_returns, weights, confidence)
        marginal_vars = cudf.Series(index=weights.index, dtype='float64')
        
        # Calculate marginal VaR for each asset
        delta = 0.001  # Small change in weight
        
        for asset in weights.index:
            # Increase weight slightly
            weights_adj = weights.copy()
            weights_adj[asset] += delta
            weights_adj = weights_adj / weights_adj.sum()  # Renormalize
            
            # Calculate new VaR
            var_adj = self.calculate_portfolio_var(portfolio_returns, weights_adj, confidence)
            
            # Marginal VaR
            marginal_vars[asset] = (var_adj - portfolio_var) / delta
            
        return marginal_vars
        
    def calculate_component_var(self,
                              portfolio_returns: cudf.DataFrame,
                              weights: cudf.Series,
                              confidence: float = 0.95) -> cudf.Series:
        """
        Calculate component VaR (contribution of each asset to total VaR).
        
        Component VaR = weight * marginal VaR
        
        Args:
            portfolio_returns: DataFrame with returns for each asset
            weights: Series with portfolio weights
            confidence: Confidence level
            
        Returns:
            Series with component VaR for each asset
        """
        marginal_vars = self.calculate_marginal_var(portfolio_returns, weights, confidence)
        component_vars = weights * marginal_vars
        
        return component_vars
        
    def var_backtesting(self,
                       returns: cudf.Series,
                       var_series: cudf.Series,
                       confidence: float = 0.95) -> Dict[str, float]:
        """
        Backtest VaR model by counting violations.
        
        Args:
            returns: Actual returns
            var_series: Predicted VaR values
            confidence: Confidence level used for VaR
            
        Returns:
            Dictionary with backtesting metrics
        """
        # Count violations (returns worse than VaR)
        violations = (returns < var_series).sum()
        total_observations = len(returns)
        violation_rate = float(violations / total_observations)
        
        # Expected violation rate
        expected_rate = 1 - confidence
        
        # Kupiec test statistic
        if violations > 0:
            lr_stat = -2 * np.log((expected_rate**violations * (1-expected_rate)**(total_observations-violations)) /
                                 (violation_rate**violations * (1-violation_rate)**(total_observations-violations)))
        else:
            lr_stat = -2 * total_observations * np.log(1 - expected_rate)
            
        return {
            'violations': int(violations),
            'violation_rate': violation_rate,
            'expected_rate': expected_rate,
            'kupiec_lr_stat': lr_stat,
            'kupiec_critical_value': 3.841  # Chi-square 95% with 1 df
        }
        
    def calculate_all_risk_metrics(self,
                                 returns: cudf.Series,
                                 portfolio_returns: Optional[cudf.DataFrame] = None,
                                 weights: Optional[cudf.Series] = None) -> Dict[str, Dict[str, float]]:
        """
        Calculate comprehensive risk metrics at all confidence levels.
        
        Args:
            returns: Series of returns (single asset or portfolio)
            portfolio_returns: Optional DataFrame for portfolio calculations
            weights: Optional weights for portfolio calculations
            
        Returns:
            Dictionary with VaR and CVaR at each confidence level
        """
        metrics = {}
        
        for confidence in self.confidence_levels:
            conf_pct = int(confidence * 100)
            
            if portfolio_returns is not None and weights is not None:
                # Portfolio metrics
                var_hist = self.calculate_portfolio_var(portfolio_returns, weights, confidence, 'historical')
                var_param = self.calculate_portfolio_var(portfolio_returns, weights, confidence, 'parametric')
                cvar = self.calculate_portfolio_cvar(portfolio_returns, weights, confidence)
            else:
                # Single asset metrics
                var_hist = self.calculate_var_historical(returns, confidence)
                var_param = self.calculate_var_parametric(returns, confidence)
                cvar = self.calculate_cvar_historical(returns, confidence)
                
            metrics[f'{conf_pct}%'] = {
                'var_historical': var_hist,
                'var_parametric': var_param,
                'cvar': cvar,
                'var_to_cvar_ratio': var_hist / cvar if cvar != 0 else 0
            }
            
        return metrics


def create_risk_config(config_dict: Dict) -> Dict:
    """
    Create risk management configuration from config dictionary.
    
    Args:
        config_dict: Configuration dictionary
        
    Returns:
        Risk-specific configuration
    """
    risk_config = {
        'var_confidence_levels': [float(x)/100 for x in config_dict.get('RISK_METRICS', {}).get('var_confidence_levels', '95,99').split(',')],
        'cvar_confidence_levels': [float(x)/100 for x in config_dict.get('RISK_METRICS', {}).get('cvar_confidence_levels', '95,99').split(',')],
        'var_limit': float(config_dict.get('RISK_METRICS', {}).get('var_limit', 0.025)),
        'risk_free_rate': float(config_dict.get('RISK_METRICS', {}).get('risk_free_rate', 0.02)),
        'var_method': config_dict.get('RISK_METRICS', {}).get('var_method', 'historical')
    }
    
    return risk_config