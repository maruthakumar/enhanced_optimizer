"""
Risk Metrics Calculator Module

Implements production risk metrics including:
- Value at Risk (VaR) using 82 days of actual returns
- Conditional Value at Risk (CVaR) 
- Maximum drawdown analysis from production data
- Return distribution analysis and tail risks
- Advanced risk-adjusted performance metrics
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from scipy import stats
import warnings

logger = logging.getLogger(__name__)


@dataclass
class RiskMetricsConfig:
    """Configuration for risk metrics calculation"""
    confidence_levels: List[float] = None
    var_methods: List[str] = None
    lookback_periods: List[int] = None
    return_frequency: str = 'daily'
    annualization_factor: int = 252  # Trading days per year
    extreme_percentiles: List[float] = None
    drawdown_threshold: float = -0.05  # 5% drawdown threshold
    
    def __post_init__(self):
        if self.confidence_levels is None:
            self.confidence_levels = [90, 95, 99, 99.5]
        
        if self.var_methods is None:
            self.var_methods = ['historical', 'parametric', 'monte_carlo']
        
        if self.lookback_periods is None:
            self.lookback_periods = [20, 60, 82]  # 20-day, 60-day, and full period
            
        if self.extreme_percentiles is None:
            self.extreme_percentiles = [1, 5, 95, 99]


class RiskMetricsCalculator:
    """
    Calculates comprehensive risk metrics using production data.
    
    Provides:
    - VaR and CVaR calculations using 82 days of actual returns
    - Maximum drawdown analysis from real trading data
    - Return distribution characteristics and tail risk analysis
    - Risk-adjusted performance metrics
    """
    
    def __init__(self, config: Optional[RiskMetricsConfig] = None):
        """
        Initialize risk metrics calculator
        
        Args:
            config: Risk metrics configuration
        """
        self.config = config or RiskMetricsConfig()
        self.risk_cache = {}
        
    def calculate_comprehensive_risk_metrics(self, 
                                           daily_returns: np.ndarray,
                                           portfolio: List[int],
                                           portfolio_weights: np.ndarray,
                                           strategy_names: List[str],
                                           dates: Optional[pd.DatetimeIndex] = None,
                                           correlation_matrix: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Calculate comprehensive risk metrics for portfolio
        
        Args:
            daily_returns: Daily returns matrix (days x strategies)
            portfolio: Portfolio strategy indices
            portfolio_weights: Portfolio weights
            strategy_names: Strategy names
            dates: Trading dates
            
        Returns:
            Dictionary with comprehensive risk metrics
        """
        logger.info("⚖️ Calculating comprehensive risk metrics")
        
        # Calculate portfolio returns
        portfolio_returns = self._calculate_portfolio_returns(
            portfolio, portfolio_weights, daily_returns
        )
        
        # Individual strategy returns for analysis
        strategy_returns = daily_returns[:, portfolio]
        
        # Calculate individual risk components
        var_results = self.calculate_value_at_risk(portfolio_returns)
        cvar_results = self.calculate_conditional_var(portfolio_returns)
        drawdown_results = self.analyze_drawdowns(portfolio_returns, dates)
        
        # Flatten VaR results to expected format
        var_analysis = {}
        
        # Add fallback values if specific keys don't exist
        var_analysis['var_95'] = 0.0
        var_analysis['var_99'] = 0.0
        var_analysis['cvar_95'] = 0.0
        var_analysis['cvar_99'] = 0.0
        
        # Try to extract from complex structure
        try:
            if '95%' in var_results and 'historical' in var_results['95%']:
                var_analysis['var_95'] = var_results['95%']['historical']
            elif '95' in var_results:
                var_analysis['var_95'] = var_results['95']
            
            if '99%' in var_results and 'historical' in var_results['99%']:
                var_analysis['var_99'] = var_results['99%']['historical']
            elif '99' in var_results:
                var_analysis['var_99'] = var_results['99']
                
            if '95%' in cvar_results and 'historical' in cvar_results['95%']:
                var_analysis['cvar_95'] = cvar_results['95%']['historical']
            elif '95' in cvar_results:
                var_analysis['cvar_95'] = cvar_results['95']
                
            if '99%' in cvar_results and 'historical' in cvar_results['99%']:
                var_analysis['cvar_99'] = cvar_results['99%']['historical']
            elif '99' in cvar_results:
                var_analysis['cvar_99'] = cvar_results['99']
        except Exception as e:
            logger.warning(f"⚠️  Error extracting VaR values: {e}")
            # Use simple percentile fallback
            portfolio_returns = self._calculate_portfolio_returns(portfolio, portfolio_weights, daily_returns)
            var_analysis['var_95'] = float(np.percentile(portfolio_returns, 5))
            var_analysis['var_99'] = float(np.percentile(portfolio_returns, 1))
            var_analysis['cvar_95'] = float(np.mean(portfolio_returns[portfolio_returns <= var_analysis['var_95']]))
            var_analysis['cvar_99'] = float(np.mean(portfolio_returns[portfolio_returns <= var_analysis['var_99']]))
        
        # Ensure max_drawdown is available
        if 'max_drawdown' not in drawdown_results:
            portfolio_returns = self._calculate_portfolio_returns(portfolio, portfolio_weights, daily_returns)
            cumulative = np.cumprod(1 + portfolio_returns / 100)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / running_max
            drawdown_results['max_drawdown'] = float(np.min(drawdown))
        
        risk_metrics = {
            'var_analysis': var_analysis,
            'drawdown_analysis': drawdown_results,
            'return_distribution_analysis': self.analyze_return_distribution(portfolio_returns),
            'tail_risk_analysis': self.analyze_tail_risks(portfolio_returns),
            'risk_adjusted_metrics': self.calculate_risk_adjusted_metrics(portfolio_returns),
            'volatility_analysis': self.analyze_volatility_patterns(portfolio_returns, dates),
            'extreme_events': self.identify_extreme_events(portfolio_returns, dates),
            'individual_risk_contributions': self.calculate_strategy_risk_contributions(
                strategy_returns, portfolio_weights, strategy_names, portfolio
            )
        }
        
        # Add risk summary
        risk_metrics['risk_summary'] = self._generate_risk_summary(
            risk_metrics, portfolio_returns
        )
        
        logger.info("✅ Comprehensive risk metrics calculation completed")
        return risk_metrics
    
    def calculate_value_at_risk(self, returns: np.ndarray) -> Dict[str, Any]:
        """
        Calculate Value at Risk using multiple methods
        
        Args:
            returns: Portfolio daily returns
            
        Returns:
            VaR calculations for different confidence levels and methods
        """
        logger.info("📊 Calculating Value at Risk (VaR)")
        
        var_results = {}
        
        for confidence_level in self.config.confidence_levels:
            var_results[f'{confidence_level}%'] = {}
            
            for method in self.config.var_methods:
                try:
                    if method == 'historical':
                        var_value = self._calculate_historical_var(returns, confidence_level)
                    elif method == 'parametric':
                        var_value = self._calculate_parametric_var(returns, confidence_level)
                    elif method == 'monte_carlo':
                        var_value = self._calculate_monte_carlo_var(returns, confidence_level)
                    else:
                        continue
                    
                    var_results[f'{confidence_level}%'][method] = {
                        'var': var_value,
                        'var_pct': var_value * 100,
                        'annualized_var': var_value * np.sqrt(self.config.annualization_factor)
                    }
                    
                except Exception as e:
                    logger.warning(f"VaR calculation failed for {method} at {confidence_level}%: {e}")
                    var_results[f'{confidence_level}%'][method] = {
                        'var': None,
                        'error': str(e)
                    }
        
        # Calculate VaR for different lookback periods
        lookback_var = self._calculate_lookback_var(returns)
        
        return {
            'var_by_confidence_and_method': var_results,
            'var_by_lookback_period': lookback_var,
            'var_comparison': self._compare_var_methods(var_results)
        }
    
    def calculate_conditional_var(self, returns: np.ndarray) -> Dict[str, Any]:
        """
        Calculate Conditional Value at Risk (Expected Shortfall)
        
        Args:
            returns: Portfolio daily returns
            
        Returns:
            CVaR calculations for different confidence levels
        """
        logger.info("📊 Calculating Conditional Value at Risk (CVaR)")
        
        cvar_results = {}
        
        for confidence_level in self.config.confidence_levels:
            # Historical CVaR
            percentile = 100 - confidence_level
            var_threshold = np.percentile(returns, percentile)
            
            # CVaR is the average of returns worse than VaR
            tail_returns = returns[returns <= var_threshold]
            cvar = np.mean(tail_returns) if len(tail_returns) > 0 else var_threshold
            
            # CVaR ratio (CVaR / VaR)
            var_value = var_threshold
            cvar_ratio = cvar / var_value if var_value != 0 else 1
            
            cvar_results[f'{confidence_level}%'] = {
                'cvar': cvar,
                'cvar_pct': cvar * 100,
                'var': var_value,
                'cvar_var_ratio': cvar_ratio,
                'tail_observations': len(tail_returns),
                'tail_percentage': len(tail_returns) / len(returns) * 100,
                'annualized_cvar': cvar * np.sqrt(self.config.annualization_factor)
            }
        
        return cvar_results
    
    def analyze_drawdowns(self, returns: np.ndarray, 
                         dates: Optional[pd.DatetimeIndex] = None) -> Dict[str, Any]:
        """
        Analyze drawdowns from production data
        
        Args:
            returns: Portfolio daily returns
            dates: Trading dates
            
        Returns:
            Comprehensive drawdown analysis
        """
        logger.info("📉 Analyzing drawdowns from production data")
        
        # Calculate cumulative returns
        cumulative_returns = np.cumsum(returns)
        
        # Calculate running maximum
        running_max = np.maximum.accumulate(cumulative_returns)
        
        # Calculate drawdowns
        drawdowns = cumulative_returns - running_max
        
        # Maximum drawdown
        max_drawdown = np.min(drawdowns)
        max_drawdown_idx = np.argmin(drawdowns)
        
        # Find the peak before max drawdown
        peak_idx = np.argmax(running_max[:max_drawdown_idx+1])
        
        # Recovery analysis
        recovery_idx = None
        if max_drawdown_idx < len(cumulative_returns) - 1:
            peak_value = running_max[max_drawdown_idx]
            for i in range(max_drawdown_idx + 1, len(cumulative_returns)):
                if cumulative_returns[i] >= peak_value:
                    recovery_idx = i
                    break
        
        # Duration calculations
        drawdown_duration = max_drawdown_idx - peak_idx
        recovery_duration = (recovery_idx - max_drawdown_idx) if recovery_idx else None
        total_duration = (recovery_idx - peak_idx) if recovery_idx else (len(returns) - peak_idx)
        
        # Identify all significant drawdowns
        significant_drawdowns = self._identify_significant_drawdowns(
            drawdowns, cumulative_returns, dates, self.config.drawdown_threshold
        )
        
        # Drawdown statistics
        drawdown_stats = {
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown * 100,
            'max_drawdown_date': dates[max_drawdown_idx].strftime('%Y-%m-%d') if dates is not None else max_drawdown_idx,
            'peak_date': dates[peak_idx].strftime('%Y-%m-%d') if dates is not None else peak_idx,
            'recovery_date': dates[recovery_idx].strftime('%Y-%m-%d') if dates is not None and recovery_idx else None,
            'drawdown_duration_days': drawdown_duration,
            'recovery_duration_days': recovery_duration,
            'total_duration_days': total_duration,
            'is_recovered': recovery_idx is not None,
            'current_drawdown': drawdowns[-1],
            'current_drawdown_pct': drawdowns[-1] * 100
        }
        
        # Drawdown frequency analysis
        drawdown_frequency = self._analyze_drawdown_frequency(drawdowns)
        
        return {
            'drawdown_statistics': drawdown_stats,
            'significant_drawdowns': significant_drawdowns,
            'drawdown_frequency': drawdown_frequency,
            'drawdown_time_series': drawdowns.tolist(),
            'underwater_curve': self._calculate_underwater_curve(drawdowns)
        }
    
    def analyze_return_distribution(self, returns: np.ndarray) -> Dict[str, Any]:
        """
        Analyze return distribution characteristics and tail risks
        
        Args:
            returns: Portfolio daily returns
            
        Returns:
            Return distribution analysis
        """
        logger.info("📈 Analyzing return distribution characteristics")
        
        # Basic distribution statistics
        distribution_stats = {
            'mean': np.mean(returns),
            'median': np.median(returns),
            'std': np.std(returns),
            'variance': np.var(returns),
            'skewness': stats.skew(returns),
            'kurtosis': stats.kurtosis(returns),
            'excess_kurtosis': stats.kurtosis(returns, fisher=True),
            'min': np.min(returns),
            'max': np.max(returns),
            'range': np.max(returns) - np.min(returns),
            'interquartile_range': np.percentile(returns, 75) - np.percentile(returns, 25)
        }
        
        # Percentile analysis
        percentiles = {}
        for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
            percentiles[f'{p}th'] = np.percentile(returns, p)
        
        # Normality tests
        normality_tests = self._test_return_normality(returns)
        
        # Distribution moments
        moments = {
            'first_moment': np.mean(returns),
            'second_moment': np.mean(returns**2),
            'third_moment': np.mean(returns**3),
            'fourth_moment': np.mean(returns**4),
            'standardized_third_moment': distribution_stats['skewness'],
            'standardized_fourth_moment': distribution_stats['kurtosis'] + 3
        }
        
        # Tail analysis
        left_tail = returns[returns < np.percentile(returns, 5)]
        right_tail = returns[returns > np.percentile(returns, 95)]
        
        tail_analysis = {
            'left_tail_mean': np.mean(left_tail) if len(left_tail) > 0 else None,
            'left_tail_std': np.std(left_tail) if len(left_tail) > 0 else None,
            'right_tail_mean': np.mean(right_tail) if len(right_tail) > 0 else None,
            'right_tail_std': np.std(right_tail) if len(right_tail) > 0 else None,
            'tail_ratio': (np.mean(right_tail) / abs(np.mean(left_tail))) if len(left_tail) > 0 and len(right_tail) > 0 and np.mean(left_tail) != 0 else None
        }
        
        return {
            'distribution_statistics': distribution_stats,
            'percentiles': percentiles,
            'moments': moments,
            'normality_tests': normality_tests,
            'tail_analysis': tail_analysis,
            'return_histogram': self._calculate_return_histogram(returns)
        }
    
    def analyze_tail_risks(self, returns: np.ndarray) -> Dict[str, Any]:
        """
        Analyze tail risks and extreme events
        
        Args:
            returns: Portfolio daily returns
            
        Returns:
            Tail risk analysis
        """
        logger.info("🎯 Analyzing tail risks and extreme events")
        
        # Extreme percentile analysis
        extreme_values = {}
        for percentile in self.config.extreme_percentiles:
            extreme_values[f'{percentile}th_percentile'] = np.percentile(returns, percentile)
        
        # Tail statistics
        left_tail_threshold = np.percentile(returns, 5)
        right_tail_threshold = np.percentile(returns, 95)
        
        left_tail_returns = returns[returns <= left_tail_threshold]
        right_tail_returns = returns[returns >= right_tail_threshold]
        
        tail_statistics = {
            'left_tail_count': len(left_tail_returns),
            'right_tail_count': len(right_tail_returns),
            'left_tail_frequency': len(left_tail_returns) / len(returns),
            'right_tail_frequency': len(right_tail_returns) / len(returns),
            'left_tail_mean': np.mean(left_tail_returns) if len(left_tail_returns) > 0 else 0,
            'right_tail_mean': np.mean(right_tail_returns) if len(right_tail_returns) > 0 else 0,
            'extreme_loss_days': np.sum(returns < np.percentile(returns, 1)),
            'extreme_gain_days': np.sum(returns > np.percentile(returns, 99))
        }
        
        # Tail clustering analysis
        tail_clustering = self._analyze_tail_clustering(returns)
        
        # Expected shortfall at extreme levels
        extreme_shortfall = {}
        for confidence in [99, 99.5, 99.9]:
            if confidence <= 100:
                percentile = 100 - confidence
                threshold = np.percentile(returns, percentile)
                tail_returns = returns[returns <= threshold]
                extreme_shortfall[f'{confidence}%'] = {
                    'threshold': threshold,
                    'expected_shortfall': np.mean(tail_returns) if len(tail_returns) > 0 else threshold,
                    'observations': len(tail_returns)
                }
        
        return {
            'extreme_percentiles': extreme_values,
            'tail_statistics': tail_statistics,
            'tail_clustering': tail_clustering,
            'extreme_shortfall': extreme_shortfall,
            'tail_risk_metrics': self._calculate_tail_risk_metrics(returns)
        }
    
    def calculate_risk_adjusted_metrics(self, returns: np.ndarray) -> Dict[str, Any]:
        """
        Calculate risk-adjusted performance metrics
        
        Args:
            returns: Portfolio daily returns
            
        Returns:
            Risk-adjusted performance metrics
        """
        logger.info("📊 Calculating risk-adjusted performance metrics")
        
        # Basic metrics
        mean_return = np.mean(returns)
        volatility = np.std(returns)
        
        # Risk-adjusted ratios
        risk_metrics = {
            'sharpe_ratio': mean_return / volatility if volatility > 0 else 0,
            'information_ratio': mean_return / volatility if volatility > 0 else 0,  # Same as Sharpe for absolute returns
            'calmar_ratio': self._calculate_calmar_ratio(returns),
            'sortino_ratio': self._calculate_sortino_ratio(returns),
            'omega_ratio': self._calculate_omega_ratio(returns),
            'var_adjusted_return': self._calculate_var_adjusted_return(returns),
            'conditional_sharpe_ratio': self._calculate_conditional_sharpe_ratio(returns)
        }
        
        # Downside risk metrics
        downside_metrics = {
            'downside_deviation': self._calculate_downside_deviation(returns),
            'upside_deviation': self._calculate_upside_deviation(returns),
            'upside_downside_ratio': self._calculate_upside_downside_ratio(returns),
            'gain_loss_ratio': self._calculate_gain_loss_ratio(returns)
        }
        
        # Performance attribution
        performance_attribution = {
            'positive_days': np.sum(returns > 0),
            'negative_days': np.sum(returns < 0),
            'flat_days': np.sum(returns == 0),
            'positive_days_pct': np.sum(returns > 0) / len(returns) * 100,
            'negative_days_pct': np.sum(returns < 0) / len(returns) * 100,
            'avg_positive_return': np.mean(returns[returns > 0]) if np.sum(returns > 0) > 0 else 0,
            'avg_negative_return': np.mean(returns[returns < 0]) if np.sum(returns < 0) > 0 else 0,
            'best_day': np.max(returns),
            'worst_day': np.min(returns),
            'volatility_of_positive_returns': np.std(returns[returns > 0]) if np.sum(returns > 0) > 1 else 0,
            'volatility_of_negative_returns': np.std(returns[returns < 0]) if np.sum(returns < 0) > 1 else 0
        }
        
        return {
            'risk_adjusted_ratios': risk_metrics,
            'downside_risk_metrics': downside_metrics,
            'performance_attribution': performance_attribution,
            'annualized_metrics': self._annualize_risk_metrics(risk_metrics, mean_return, volatility)
        }
    
    def analyze_volatility_patterns(self, returns: np.ndarray,
                                  dates: Optional[pd.DatetimeIndex] = None) -> Dict[str, Any]:
        """
        Analyze volatility patterns over time
        
        Args:
            returns: Portfolio daily returns
            dates: Trading dates
            
        Returns:
            Volatility analysis
        """
        logger.info("📊 Analyzing volatility patterns")
        
        # Rolling volatility analysis
        volatility_analysis = {}
        
        for window in self.config.lookback_periods:
            if window <= len(returns):
                rolling_vol = pd.Series(returns).rolling(window=window).std()
                
                volatility_analysis[f'{window}_day'] = {
                    'current_volatility': rolling_vol.iloc[-1] if not rolling_vol.empty else 0,
                    'average_volatility': rolling_vol.mean(),
                    'max_volatility': rolling_vol.max(),
                    'min_volatility': rolling_vol.min(),
                    'volatility_trend': self._calculate_volatility_trend(rolling_vol.values),
                    'annualized_current': rolling_vol.iloc[-1] * np.sqrt(self.config.annualization_factor) if not rolling_vol.empty else 0
                }
        
        # Volatility clustering analysis
        volatility_clustering = self._analyze_volatility_clustering(returns)
        
        # GARCH-like analysis
        volatility_persistence = self._analyze_volatility_persistence(returns)
        
        return {
            'rolling_volatility': volatility_analysis,
            'volatility_clustering': volatility_clustering,
            'volatility_persistence': volatility_persistence,
            'volatility_regimes': self._identify_volatility_regimes(returns, dates)
        }
    
    def identify_extreme_events(self, returns: np.ndarray,
                              dates: Optional[pd.DatetimeIndex] = None) -> Dict[str, Any]:
        """
        Identify extreme events in the return series
        
        Args:
            returns: Portfolio daily returns
            dates: Trading dates
            
        Returns:
            Extreme events analysis
        """
        logger.info("⚡ Identifying extreme events")
        
        # Define extreme thresholds
        extreme_loss_threshold = np.percentile(returns, 1)
        extreme_gain_threshold = np.percentile(returns, 99)
        
        # Find extreme events
        extreme_losses = []
        extreme_gains = []
        
        for i, ret in enumerate(returns):
            event_date = dates[i].strftime('%Y-%m-%d') if dates is not None else i
            
            if ret <= extreme_loss_threshold:
                extreme_losses.append({
                    'date': event_date,
                    'return': ret,
                    'return_pct': ret * 100,
                    'percentile_rank': stats.percentileofscore(returns, ret)
                })
            
            if ret >= extreme_gain_threshold:
                extreme_gains.append({
                    'date': event_date,
                    'return': ret,
                    'return_pct': ret * 100,
                    'percentile_rank': stats.percentileofscore(returns, ret)
                })
        
        # Sort by magnitude
        extreme_losses.sort(key=lambda x: x['return'])
        extreme_gains.sort(key=lambda x: x['return'], reverse=True)
        
        # Cluster analysis of extreme events
        extreme_event_clustering = self._analyze_extreme_event_clustering(
            extreme_losses, extreme_gains, dates
        )
        
        return {
            'extreme_losses': extreme_losses,
            'extreme_gains': extreme_gains,
            'extreme_event_statistics': {
                'total_extreme_losses': len(extreme_losses),
                'total_extreme_gains': len(extreme_gains),
                'extreme_loss_frequency': len(extreme_losses) / len(returns),
                'extreme_gain_frequency': len(extreme_gains) / len(returns),
                'worst_loss': extreme_losses[0]['return'] if extreme_losses else 0,
                'best_gain': extreme_gains[0]['return'] if extreme_gains else 0
            },
            'extreme_event_clustering': extreme_event_clustering
        }
    
    def calculate_strategy_risk_contributions(self, strategy_returns: np.ndarray,
                                            portfolio_weights: np.ndarray,
                                            strategy_names: List[str],
                                            portfolio: List[int]) -> Dict[str, Any]:
        """
        Calculate individual strategy risk contributions
        
        Args:
            strategy_returns: Individual strategy returns matrix
            portfolio_weights: Portfolio weights
            strategy_names: Strategy names
            portfolio: Portfolio indices
            
        Returns:
            Strategy risk contribution analysis
        """
        logger.info("⚖️ Calculating strategy risk contributions")
        
        # Calculate covariance matrix
        cov_matrix = np.cov(strategy_returns.T)
        
        # Portfolio variance
        portfolio_variance = np.dot(portfolio_weights, np.dot(cov_matrix, portfolio_weights))
        
        # Marginal risk contributions
        marginal_contributions = np.dot(cov_matrix, portfolio_weights)
        
        # Component risk contributions
        component_contributions = portfolio_weights * marginal_contributions
        
        # Risk contributions as percentage of total risk
        risk_contributions_pct = component_contributions / portfolio_variance * 100
        
        # Individual strategy risk analysis
        strategy_risk_analysis = {}
        
        for i, strategy_idx in enumerate(portfolio):
            strategy_name = strategy_names[strategy_idx]
            individual_volatility = np.std(strategy_returns[:, i])
            
            strategy_risk_analysis[strategy_name] = {
                'weight': portfolio_weights[i],
                'individual_volatility': individual_volatility,
                'marginal_risk_contribution': marginal_contributions[i],
                'component_risk_contribution': component_contributions[i],
                'risk_contribution_pct': risk_contributions_pct[i],
                'risk_adjusted_weight': component_contributions[i] / np.sum(np.abs(component_contributions)),
                'diversification_multiplier': marginal_contributions[i] / (individual_volatility ** 2) if individual_volatility > 0 else 0
            }
        
        # Risk concentration analysis
        risk_concentration = {
            'herfindahl_index': np.sum((risk_contributions_pct / 100) ** 2),
            'effective_number_of_risk_factors': 1 / np.sum((risk_contributions_pct / 100) ** 2),
            'top_5_risk_contribution': np.sum(np.sort(risk_contributions_pct)[-5:]),
            'top_10_risk_contribution': np.sum(np.sort(risk_contributions_pct)[-10:]) if len(risk_contributions_pct) >= 10 else np.sum(risk_contributions_pct)
        }
        
        return {
            'individual_contributions': strategy_risk_analysis,
            'risk_concentration': risk_concentration,
            'portfolio_risk_metrics': {
                'portfolio_variance': portfolio_variance,
                'portfolio_volatility': np.sqrt(portfolio_variance),
                'total_risk_contribution': np.sum(component_contributions),
                'risk_budget_utilization': np.sum(np.abs(risk_contributions_pct))
            }
        }
    
    def _calculate_portfolio_returns(self, portfolio: List[int],
                                   portfolio_weights: np.ndarray,
                                   daily_returns: np.ndarray) -> np.ndarray:
        """Calculate portfolio returns from individual strategy returns"""
        portfolio_data = daily_returns[:, portfolio]
        return np.sum(portfolio_data * portfolio_weights, axis=1)
    
    def _calculate_historical_var(self, returns: np.ndarray, confidence_level: float) -> float:
        """Calculate historical VaR"""
        percentile = 100 - confidence_level
        return np.percentile(returns, percentile)
    
    def _calculate_parametric_var(self, returns: np.ndarray, confidence_level: float) -> float:
        """Calculate parametric VaR assuming normal distribution"""
        mean = np.mean(returns)
        std = np.std(returns)
        z_score = stats.norm.ppf((100 - confidence_level) / 100)
        return mean + z_score * std
    
    def _calculate_monte_carlo_var(self, returns: np.ndarray, confidence_level: float, 
                                 n_simulations: int = 10000) -> float:
        """Calculate Monte Carlo VaR"""
        mean = np.mean(returns)
        std = np.std(returns)
        
        # Generate Monte Carlo scenarios
        simulated_returns = np.random.normal(mean, std, n_simulations)
        
        # Calculate VaR from simulations
        percentile = 100 - confidence_level
        return np.percentile(simulated_returns, percentile)
    
    def _calculate_lookback_var(self, returns: np.ndarray) -> Dict[str, Any]:
        """Calculate VaR for different lookback periods"""
        lookback_var = {}
        
        for period in self.config.lookback_periods:
            if period <= len(returns):
                period_returns = returns[-period:]  # Last 'period' observations
                
                lookback_var[f'{period}_days'] = {
                    'var_95': np.percentile(period_returns, 5),
                    'var_99': np.percentile(period_returns, 1),
                    'observations': len(period_returns),
                    'mean_return': np.mean(period_returns),
                    'volatility': np.std(period_returns)
                }
        
        return lookback_var
    
    def _compare_var_methods(self, var_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Compare VaR results across different methods"""
        comparison = {}
        
        for confidence_level, methods in var_results.items():
            method_values = {}
            for method, result in methods.items():
                if 'var' in result and result['var'] is not None:
                    method_values[method] = result['var']
            
            if len(method_values) > 1:
                values = list(method_values.values())
                comparison[confidence_level] = {
                    'method_values': method_values,
                    'max_var': max(values),
                    'min_var': min(values),
                    'var_range': max(values) - min(values),
                    'avg_var': np.mean(values),
                    'var_std': np.std(values),
                    'most_conservative': max(method_values, key=method_values.get),
                    'least_conservative': min(method_values, key=method_values.get)
                }
        
        return comparison
    
    def _identify_significant_drawdowns(self, drawdowns: np.ndarray,
                                      cumulative_returns: np.ndarray,
                                      dates: Optional[pd.DatetimeIndex],
                                      threshold: float) -> List[Dict[str, Any]]:
        """Identify all significant drawdowns"""
        significant_drawdowns = []
        
        # Find local peaks and troughs
        in_drawdown = False
        drawdown_start = 0
        peak_value = 0
        
        for i, (drawdown, cum_return) in enumerate(zip(drawdowns, cumulative_returns)):
            if drawdown == 0 and not in_drawdown:
                # Potential start of new peak
                peak_value = cum_return
                drawdown_start = i
            elif drawdown < threshold and not in_drawdown:
                # Start of significant drawdown
                in_drawdown = True
                peak_idx = drawdown_start
                peak_value = cumulative_returns[peak_idx]
            elif drawdown == 0 and in_drawdown:
                # End of drawdown (recovery)
                in_drawdown = False
                trough_idx = np.argmin(cumulative_returns[peak_idx:i+1]) + peak_idx
                trough_value = cumulative_returns[trough_idx]
                
                significant_drawdowns.append({
                    'peak_date': dates[peak_idx].strftime('%Y-%m-%d') if dates is not None else peak_idx,
                    'trough_date': dates[trough_idx].strftime('%Y-%m-%d') if dates is not None else trough_idx,
                    'recovery_date': dates[i].strftime('%Y-%m-%d') if dates is not None else i,
                    'peak_value': peak_value,
                    'trough_value': trough_value,
                    'drawdown_magnitude': trough_value - peak_value,
                    'drawdown_pct': (trough_value - peak_value) / peak_value * 100 if peak_value != 0 else 0,
                    'drawdown_duration': trough_idx - peak_idx,
                    'recovery_duration': i - trough_idx,
                    'total_duration': i - peak_idx
                })
        
        # Sort by magnitude
        significant_drawdowns.sort(key=lambda x: x['drawdown_magnitude'])
        
        return significant_drawdowns[:10]  # Return top 10 worst drawdowns
    
    def _analyze_drawdown_frequency(self, drawdowns: np.ndarray) -> Dict[str, Any]:
        """Analyze frequency and duration of drawdowns"""
        # Count drawdown periods
        drawdown_periods = 0
        in_drawdown = False
        current_duration = 0
        durations = []
        
        for dd in drawdowns:
            if dd < 0 and not in_drawdown:
                in_drawdown = True
                drawdown_periods += 1
                current_duration = 1
            elif dd < 0 and in_drawdown:
                current_duration += 1
            elif dd == 0 and in_drawdown:
                in_drawdown = False
                durations.append(current_duration)
                current_duration = 0
        
        # If still in drawdown at the end
        if in_drawdown:
            durations.append(current_duration)
        
        return {
            'total_drawdown_periods': drawdown_periods,
            'average_drawdown_duration': np.mean(durations) if durations else 0,
            'max_drawdown_duration': max(durations) if durations else 0,
            'min_drawdown_duration': min(durations) if durations else 0,
            'drawdown_frequency': drawdown_periods / len(drawdowns),
            'time_in_drawdown_pct': np.sum(np.array(drawdowns) < 0) / len(drawdowns) * 100
        }
    
    def _calculate_underwater_curve(self, drawdowns: np.ndarray) -> Dict[str, Any]:
        """Calculate underwater curve statistics"""
        return {
            'current_underwater_pct': drawdowns[-1] * 100,
            'max_underwater_pct': np.min(drawdowns) * 100,
            'average_underwater_pct': np.mean(drawdowns[drawdowns < 0]) * 100 if np.any(drawdowns < 0) else 0,
            'days_underwater': np.sum(drawdowns < 0),
            'days_underwater_pct': np.sum(drawdowns < 0) / len(drawdowns) * 100
        }
    
    def _test_return_normality(self, returns: np.ndarray) -> Dict[str, Any]:
        """Test return distribution for normality"""
        normality_tests = {}
        
        try:
            # Shapiro-Wilk test
            if len(returns) <= 5000:  # Shapiro-Wilk has sample size limitations
                shapiro_stat, shapiro_p = stats.shapiro(returns)
                normality_tests['shapiro_wilk'] = {
                    'statistic': shapiro_stat,
                    'p_value': shapiro_p,
                    'is_normal': shapiro_p > 0.05
                }
        except:
            pass
        
        try:
            # Jarque-Bera test
            jb_stat, jb_p = stats.jarque_bera(returns)
            normality_tests['jarque_bera'] = {
                'statistic': jb_stat,
                'p_value': jb_p,
                'is_normal': jb_p > 0.05
            }
        except:
            pass
        
        try:
            # Kolmogorov-Smirnov test against normal distribution
            # Standardize returns
            standardized_returns = (returns - np.mean(returns)) / np.std(returns)
            ks_stat, ks_p = stats.kstest(standardized_returns, 'norm')
            normality_tests['kolmogorov_smirnov'] = {
                'statistic': ks_stat,
                'p_value': ks_p,
                'is_normal': ks_p > 0.05
            }
        except:
            pass
        
        return normality_tests
    
    def _calculate_return_histogram(self, returns: np.ndarray, bins: int = 50) -> Dict[str, Any]:
        """Calculate return histogram for visualization"""
        hist, bin_edges = np.histogram(returns, bins=bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        return {
            'bins': bin_centers.tolist(),
            'frequencies': hist.tolist(),
            'bin_width': bin_edges[1] - bin_edges[0],
            'total_observations': len(returns)
        }
    
    def _analyze_tail_clustering(self, returns: np.ndarray) -> Dict[str, Any]:
        """Analyze clustering of tail events"""
        # Define tail thresholds
        lower_tail = np.percentile(returns, 5)
        upper_tail = np.percentile(returns, 95)
        
        # Find tail events
        lower_tail_indices = np.where(returns <= lower_tail)[0]
        upper_tail_indices = np.where(returns >= upper_tail)[0]
        
        # Analyze clustering (consecutive tail events)
        def analyze_consecutive_events(indices):
            if len(indices) == 0:
                return {'clusters': 0, 'avg_cluster_size': 0, 'max_cluster_size': 0}
            
            clusters = []
            current_cluster = [indices[0]]
            
            for i in range(1, len(indices)):
                if indices[i] - indices[i-1] == 1:  # Consecutive
                    current_cluster.append(indices[i])
                else:
                    clusters.append(current_cluster)
                    current_cluster = [indices[i]]
            
            clusters.append(current_cluster)
            
            cluster_sizes = [len(cluster) for cluster in clusters]
            
            return {
                'clusters': len(clusters),
                'avg_cluster_size': np.mean(cluster_sizes),
                'max_cluster_size': max(cluster_sizes),
                'single_event_clusters': sum(1 for size in cluster_sizes if size == 1),
                'multi_event_clusters': sum(1 for size in cluster_sizes if size > 1)
            }
        
        return {
            'lower_tail_clustering': analyze_consecutive_events(lower_tail_indices),
            'upper_tail_clustering': analyze_consecutive_events(upper_tail_indices),
            'total_tail_events': len(lower_tail_indices) + len(upper_tail_indices),
            'tail_event_frequency': (len(lower_tail_indices) + len(upper_tail_indices)) / len(returns)
        }
    
    def _calculate_tail_risk_metrics(self, returns: np.ndarray) -> Dict[str, Any]:
        """Calculate additional tail risk metrics"""
        # Lower partial moments
        threshold = 0  # Use zero as threshold
        downside_returns = returns[returns < threshold] - threshold
        
        lpm_1 = np.mean(np.abs(downside_returns)) if len(downside_returns) > 0 else 0
        lpm_2 = np.mean(downside_returns ** 2) if len(downside_returns) > 0 else 0
        
        # Tail expectation
        tail_5_pct = returns[returns <= np.percentile(returns, 5)]
        tail_1_pct = returns[returns <= np.percentile(returns, 1)]
        
        return {
            'lower_partial_moment_1': lpm_1,
            'lower_partial_moment_2': lpm_2,
            'tail_expectation_5pct': np.mean(tail_5_pct) if len(tail_5_pct) > 0 else 0,
            'tail_expectation_1pct': np.mean(tail_1_pct) if len(tail_1_pct) > 0 else 0,
            'tail_variance_5pct': np.var(tail_5_pct) if len(tail_5_pct) > 1 else 0,
            'tail_variance_1pct': np.var(tail_1_pct) if len(tail_1_pct) > 1 else 0
        }
    
    def _calculate_calmar_ratio(self, returns: np.ndarray) -> float:
        """Calculate Calmar ratio (annual return / max drawdown)"""
        annual_return = np.mean(returns) * self.config.annualization_factor
        
        # Calculate max drawdown
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = cumulative - running_max
        max_drawdown = abs(np.min(drawdown))
        
        return annual_return / max_drawdown if max_drawdown > 0 else 0
    
    def _calculate_sortino_ratio(self, returns: np.ndarray, target_return: float = 0) -> float:
        """Calculate Sortino ratio"""
        mean_return = np.mean(returns)
        downside_returns = returns[returns < target_return]
        downside_deviation = np.std(downside_returns) if len(downside_returns) > 0 else 0
        
        return (mean_return - target_return) / downside_deviation if downside_deviation > 0 else 0
    
    def _calculate_omega_ratio(self, returns: np.ndarray, threshold: float = 0) -> float:
        """Calculate Omega ratio"""
        upside_returns = returns[returns > threshold] - threshold
        downside_returns = threshold - returns[returns < threshold]
        
        upside_sum = np.sum(upside_returns) if len(upside_returns) > 0 else 0
        downside_sum = np.sum(downside_returns) if len(downside_returns) > 0 else 1
        
        return upside_sum / downside_sum if downside_sum > 0 else float('inf')
    
    def _calculate_var_adjusted_return(self, returns: np.ndarray) -> float:
        """Calculate VaR-adjusted return"""
        mean_return = np.mean(returns)
        var_95 = np.percentile(returns, 5)
        
        return mean_return / abs(var_95) if var_95 != 0 else 0
    
    def _calculate_conditional_sharpe_ratio(self, returns: np.ndarray) -> float:
        """Calculate Conditional Sharpe ratio using CVaR"""
        mean_return = np.mean(returns)
        
        # Calculate CVaR (5% level)
        var_threshold = np.percentile(returns, 5)
        tail_returns = returns[returns <= var_threshold]
        cvar = np.mean(tail_returns) if len(tail_returns) > 0 else var_threshold
        
        return mean_return / abs(cvar) if cvar != 0 else 0
    
    def _calculate_downside_deviation(self, returns: np.ndarray, target: float = 0) -> float:
        """Calculate downside deviation"""
        downside_returns = returns[returns < target]
        return np.std(downside_returns) if len(downside_returns) > 0 else 0
    
    def _calculate_upside_deviation(self, returns: np.ndarray, target: float = 0) -> float:
        """Calculate upside deviation"""
        upside_returns = returns[returns > target]
        return np.std(upside_returns) if len(upside_returns) > 0 else 0
    
    def _calculate_upside_downside_ratio(self, returns: np.ndarray) -> float:
        """Calculate upside/downside capture ratio"""
        upside_returns = returns[returns > 0]
        downside_returns = returns[returns < 0]
        
        upside_avg = np.mean(upside_returns) if len(upside_returns) > 0 else 0
        downside_avg = abs(np.mean(downside_returns)) if len(downside_returns) > 0 else 1
        
        return upside_avg / downside_avg if downside_avg > 0 else 0
    
    def _calculate_gain_loss_ratio(self, returns: np.ndarray) -> float:
        """Calculate gain/loss ratio"""
        gains = returns[returns > 0]
        losses = returns[returns < 0]
        
        avg_gain = np.mean(gains) if len(gains) > 0 else 0
        avg_loss = abs(np.mean(losses)) if len(losses) > 0 else 1
        
        return avg_gain / avg_loss if avg_loss > 0 else 0
    
    def _annualize_risk_metrics(self, risk_metrics: Dict[str, float],
                              mean_return: float, volatility: float) -> Dict[str, float]:
        """Annualize risk metrics"""
        annualization_factor = self.config.annualization_factor
        
        return {
            'annualized_return': mean_return * annualization_factor,
            'annualized_volatility': volatility * np.sqrt(annualization_factor),
            'annualized_sharpe_ratio': risk_metrics['sharpe_ratio'] * np.sqrt(annualization_factor)
        }
    
    def _calculate_volatility_trend(self, rolling_volatility: np.ndarray) -> str:
        """Calculate volatility trend"""
        if len(rolling_volatility) < 2:
            return 'insufficient_data'
        
        # Remove NaN values
        valid_vol = rolling_volatility[~np.isnan(rolling_volatility)]
        
        if len(valid_vol) < 2:
            return 'insufficient_data'
        
        # Simple trend analysis
        recent_vol = np.mean(valid_vol[-5:]) if len(valid_vol) >= 5 else valid_vol[-1]
        earlier_vol = np.mean(valid_vol[:5]) if len(valid_vol) >= 10 else valid_vol[0]
        
        if recent_vol > earlier_vol * 1.1:
            return 'increasing'
        elif recent_vol < earlier_vol * 0.9:
            return 'decreasing'
        else:
            return 'stable'
    
    def _analyze_volatility_clustering(self, returns: np.ndarray) -> Dict[str, Any]:
        """Analyze volatility clustering (GARCH effects)"""
        # Calculate squared returns as proxy for volatility
        squared_returns = returns ** 2
        
        # Autocorrelation of squared returns
        autocorr_lag1 = np.corrcoef(squared_returns[:-1], squared_returns[1:])[0, 1] if len(squared_returns) > 1 else 0
        autocorr_lag5 = np.corrcoef(squared_returns[:-5], squared_returns[5:])[0, 1] if len(squared_returns) > 5 else 0
        
        return {
            'volatility_autocorr_lag1': autocorr_lag1,
            'volatility_autocorr_lag5': autocorr_lag5,
            'clustering_evidence': autocorr_lag1 > 0.1,  # Simple threshold
            'clustering_strength': 'high' if autocorr_lag1 > 0.3 else 'medium' if autocorr_lag1 > 0.1 else 'low'
        }
    
    def _analyze_volatility_persistence(self, returns: np.ndarray) -> Dict[str, Any]:
        """Analyze volatility persistence"""
        # Simple volatility persistence analysis
        abs_returns = np.abs(returns)
        
        # High volatility periods
        high_vol_threshold = np.percentile(abs_returns, 75)
        high_vol_periods = abs_returns > high_vol_threshold
        
        # Count consecutive high volatility periods
        consecutive_counts = []
        current_count = 0
        
        for is_high_vol in high_vol_periods:
            if is_high_vol:
                current_count += 1
            else:
                if current_count > 0:
                    consecutive_counts.append(current_count)
                current_count = 0
        
        if current_count > 0:
            consecutive_counts.append(current_count)
        
        return {
            'avg_high_vol_persistence': np.mean(consecutive_counts) if consecutive_counts else 0,
            'max_high_vol_persistence': max(consecutive_counts) if consecutive_counts else 0,
            'high_vol_clusters': len(consecutive_counts),
            'persistence_score': np.mean(consecutive_counts) / len(returns) * 100 if consecutive_counts else 0
        }
    
    def _identify_volatility_regimes(self, returns: np.ndarray,
                                   dates: Optional[pd.DatetimeIndex]) -> Dict[str, Any]:
        """Identify different volatility regimes"""
        # Calculate rolling volatility
        window = min(20, len(returns) // 4)  # Use 20-day or quarter of data
        rolling_vol = pd.Series(returns).rolling(window=window).std()
        
        # Define regime thresholds
        low_vol_threshold = np.percentile(rolling_vol.dropna(), 33)
        high_vol_threshold = np.percentile(rolling_vol.dropna(), 67)
        
        # Classify regimes
        regimes = []
        for vol in rolling_vol:
            if np.isnan(vol):
                regimes.append('unknown')
            elif vol <= low_vol_threshold:
                regimes.append('low_volatility')
            elif vol >= high_vol_threshold:
                regimes.append('high_volatility')
            else:
                regimes.append('medium_volatility')
        
        # Analyze regime transitions
        regime_stats = {}
        for regime in ['low_volatility', 'medium_volatility', 'high_volatility']:
            regime_count = regimes.count(regime)
            regime_stats[regime] = {
                'count': regime_count,
                'percentage': regime_count / len(regimes) * 100,
                'avg_return_in_regime': np.mean([returns[i] for i, r in enumerate(regimes) if r == regime]) if regime_count > 0 else 0
            }
        
        return {
            'regime_classification': regimes,
            'regime_statistics': regime_stats,
            'current_regime': regimes[-1] if regimes else 'unknown',
            'regime_thresholds': {
                'low_volatility': low_vol_threshold,
                'high_volatility': high_vol_threshold
            }
        }
    
    def _analyze_extreme_event_clustering(self, extreme_losses: List[Dict[str, Any]],
                                        extreme_gains: List[Dict[str, Any]],
                                        dates: Optional[pd.DatetimeIndex]) -> Dict[str, Any]:
        """Analyze clustering of extreme events"""
        if dates is None:
            return {'error': 'Dates required for clustering analysis'}
        
        # Convert dates to indices for analysis
        def get_date_indices(events, dates):
            indices = []
            for event in events:
                try:
                    event_date = pd.to_datetime(event['date'])
                    # Find closest date index
                    date_diff = np.abs(dates - event_date)
                    closest_idx = np.argmin(date_diff)
                    indices.append(closest_idx)
                except:
                    pass
            return indices
        
        loss_indices = get_date_indices(extreme_losses, dates)
        gain_indices = get_date_indices(extreme_gains, dates)
        
        # Analyze clustering
        def analyze_clustering(indices):
            if len(indices) < 2:
                return {'clusters': 0, 'avg_gap': 0}
            
            gaps = [indices[i] - indices[i-1] for i in range(1, len(indices))]
            clusters = sum(1 for gap in gaps if gap <= 5)  # Events within 5 days
            
            return {
                'total_events': len(indices),
                'clusters': clusters,
                'avg_gap': np.mean(gaps),
                'min_gap': min(gaps),
                'max_gap': max(gaps),
                'clustering_ratio': clusters / len(indices) if len(indices) > 0 else 0
            }
        
        return {
            'extreme_loss_clustering': analyze_clustering(loss_indices),
            'extreme_gain_clustering': analyze_clustering(gain_indices),
            'overall_extreme_event_frequency': (len(extreme_losses) + len(extreme_gains)) / len(dates) * 100
        }
    
    def _generate_risk_summary(self, risk_metrics: Dict[str, Any],
                             portfolio_returns: np.ndarray) -> Dict[str, Any]:
        """Generate comprehensive risk summary"""
        
        # Extract key risk indicators
        var_95 = risk_metrics.get('value_at_risk', {}).get('var_by_confidence_and_method', {}).get('95%', {}).get('historical', {}).get('var', 0)
        cvar_95 = risk_metrics.get('conditional_var', {}).get('95%', {}).get('cvar', 0)
        max_drawdown = risk_metrics.get('drawdown_analysis', {}).get('drawdown_statistics', {}).get('max_drawdown', 0)
        sharpe_ratio = risk_metrics.get('risk_adjusted_metrics', {}).get('risk_adjusted_ratios', {}).get('sharpe_ratio', 0)
        
        # Risk level assessment
        risk_level = 'unknown'
        if var_95 != 0:
            if abs(var_95) > 0.03:  # > 3% daily VaR
                risk_level = 'high'
            elif abs(var_95) > 0.015:  # > 1.5% daily VaR
                risk_level = 'medium'
            else:
                risk_level = 'low'
        
        # Risk-return profile
        return_vol_ratio = np.mean(portfolio_returns) / np.std(portfolio_returns) if np.std(portfolio_returns) > 0 else 0
        
        risk_profile = 'balanced'
        if return_vol_ratio > 0.15:
            risk_profile = 'high_return_low_risk'
        elif return_vol_ratio < 0.05:
            risk_profile = 'low_return_high_risk'
        
        return {
            'overall_risk_level': risk_level,
            'risk_return_profile': risk_profile,
            'key_risk_metrics': {
                'var_95_pct': abs(var_95) * 100,
                'cvar_95_pct': abs(cvar_95) * 100,
                'max_drawdown_pct': abs(max_drawdown) * 100,
                'sharpe_ratio': sharpe_ratio,
                'return_volatility_ratio': return_vol_ratio
            },
            'risk_warnings': self._generate_risk_warnings(risk_metrics),
            'risk_recommendations': self._generate_risk_recommendations(risk_metrics, risk_level)
        }
    
    def _generate_risk_warnings(self, risk_metrics: Dict[str, Any]) -> List[str]:
        """Generate risk warnings based on analysis"""
        warnings = []
        
        # High VaR warning
        var_95 = risk_metrics.get('value_at_risk', {}).get('var_by_confidence_and_method', {}).get('95%', {}).get('historical', {}).get('var', 0)
        if abs(var_95) > 0.05:  # > 5% daily VaR
            warnings.append("High Value at Risk detected - portfolio may experience significant daily losses")
        
        # Large drawdown warning
        max_drawdown = risk_metrics.get('drawdown_analysis', {}).get('drawdown_statistics', {}).get('max_drawdown', 0)
        if abs(max_drawdown) > 0.20:  # > 20% drawdown
            warnings.append("Large historical drawdown detected - portfolio has experienced significant losses")
        
        # Tail risk warning
        tail_stats = risk_metrics.get('tail_risk_analysis', {}).get('tail_statistics', {})
        if tail_stats.get('left_tail_frequency', 0) > 0.1:  # > 10% of days in left tail
            warnings.append("High frequency of extreme loss days detected")
        
        # Volatility clustering warning
        vol_clustering = risk_metrics.get('volatility_analysis', {}).get('volatility_clustering', {})
        if vol_clustering.get('clustering_evidence', False):
            warnings.append("Volatility clustering detected - periods of high volatility tend to persist")
        
        return warnings
    
    def _generate_risk_recommendations(self, risk_metrics: Dict[str, Any], 
                                     risk_level: str) -> List[str]:
        """Generate risk management recommendations"""
        recommendations = []
        
        if risk_level == 'high':
            recommendations.append("Consider reducing portfolio risk through diversification or position sizing")
            recommendations.append("Implement stop-loss or hedging strategies to limit downside exposure")
        
        # Drawdown-based recommendations
        drawdown_stats = risk_metrics.get('drawdown_analysis', {}).get('drawdown_statistics', {})
        if not drawdown_stats.get('is_recovered', True):
            recommendations.append("Portfolio is currently in drawdown - monitor for recovery signals")
        
        # Tail risk recommendations
        extreme_events = risk_metrics.get('extreme_events', {}).get('extreme_event_statistics', {})
        if extreme_events.get('extreme_loss_frequency', 0) > 0.05:
            recommendations.append("High extreme loss frequency - consider tail risk hedging strategies")
        
        # Volatility recommendations
        vol_analysis = risk_metrics.get('volatility_analysis', {}).get('rolling_volatility', {})
        current_vol = None
        for period, data in vol_analysis.items():
            if 'current_volatility' in data:
                current_vol = data['current_volatility']
                break
        
        if current_vol is not None:
            avg_vol = data.get('average_volatility', current_vol)
            if current_vol > avg_vol * 1.5:
                recommendations.append("Current volatility is elevated - consider reducing position sizes")
        
        return recommendations