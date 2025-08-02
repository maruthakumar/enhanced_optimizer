"""
Scenario Modeling Module

Performs scenario analysis using historical market data including:
- Historical scenario modeling using January-July 2024 conditions
- Stress testing based on actual volatility periods  
- Market regime analysis across 82 trading days
- Strategy performance under different market conditions
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from scipy import stats
import warnings

logger = logging.getLogger(__name__)


@dataclass
class ScenarioConfig:
    """Configuration for scenario modeling"""
    volatility_lookback_days: int = 20
    stress_test_percentiles: List[float] = None
    market_regime_thresholds: Dict[str, float] = None
    monte_carlo_scenarios: int = 1000
    confidence_levels: List[float] = None
    scenario_horizon_days: int = 30
    
    def __post_init__(self):
        if self.stress_test_percentiles is None:
            self.stress_test_percentiles = [5, 10, 95, 99]
        
        if self.market_regime_thresholds is None:
            self.market_regime_thresholds = {
                'high_volatility': 0.02,  # Daily volatility > 2%
                'low_volatility': 0.005,  # Daily volatility < 0.5%
                'bull_market': 0.001,     # Daily return > 0.1%
                'bear_market': -0.001     # Daily return < -0.1%
            }
        
        if self.confidence_levels is None:
            self.confidence_levels = [90, 95, 99]


class ScenarioModeler:
    """
    Performs comprehensive scenario modeling using actual production data.
    
    Analyzes scenarios including:
    - Historical market conditions from January-July 2024
    - Stress testing based on actual volatility periods
    - Market regime transitions and their impact
    - Monte Carlo scenario generation
    """
    
    def __init__(self, config: Optional[ScenarioConfig] = None):
        """
        Initialize scenario modeler
        
        Args:
            config: Scenario modeling configuration
        """
        self.config = config or ScenarioConfig()
        self.historical_scenarios = {}
        self.stress_scenarios = {}
        self.market_regimes = {}
        
    def perform_comprehensive_scenario_analysis(self, 
                                              daily_returns: np.ndarray,
                                              portfolio: List[int],
                                              portfolio_weights: np.ndarray,
                                              strategy_names: List[str],
                                              dates: Optional[pd.DatetimeIndex] = None) -> Dict[str, Any]:
        """
        Perform comprehensive scenario analysis
        
        Args:
            daily_returns: Daily returns matrix (days x strategies)
            portfolio: Portfolio strategy indices
            portfolio_weights: Portfolio weights
            strategy_names: Strategy names
            dates: Trading dates
            
        Returns:
            Dictionary with comprehensive scenario analysis
        """
        logger.info("🎭 Starting comprehensive scenario analysis")
        
        if dates is None:
            # Create synthetic dates for 82 trading days starting Jan 4, 2024
            dates = pd.date_range(start='2024-01-04', periods=daily_returns.shape[0], freq='B')
        
        # Calculate portfolio returns
        portfolio_returns = self._calculate_portfolio_returns(
            portfolio, portfolio_weights, daily_returns
        )
        
        scenario_results = {
            'historical_scenarios': self.analyze_historical_scenarios(
                daily_returns, portfolio, portfolio_weights, strategy_names, dates
            ),
            'stress_scenarios': self.perform_stress_testing(
                daily_returns, portfolio, portfolio_weights, strategy_names
            ),
            'market_regime_analysis': self.analyze_market_regimes(
                daily_returns, portfolio_returns, dates
            ),
            'monte_carlo_scenarios': self.generate_monte_carlo_scenarios(
                daily_returns, portfolio, portfolio_weights, strategy_names
            ),
            'scenario_comparison': self.compare_scenario_outcomes(
                daily_returns, portfolio, portfolio_weights, strategy_names
            )
        }
        
        # Add scenario summary
        scenario_results['scenario_summary'] = self._generate_scenario_summary(
            scenario_results, portfolio_returns
        )
        
        logger.info("✅ Comprehensive scenario analysis completed")
        return scenario_results
    
    def analyze_historical_scenarios(self, 
                                   daily_returns: np.ndarray,
                                   portfolio: List[int],
                                   portfolio_weights: np.ndarray,
                                   strategy_names: List[str],
                                   dates: pd.DatetimeIndex) -> Dict[str, Any]:
        """
        Analyze historical scenarios from January-July 2024 market data
        
        Args:
            daily_returns: Daily returns matrix
            portfolio: Portfolio indices
            portfolio_weights: Portfolio weights
            strategy_names: Strategy names
            dates: Trading dates
            
        Returns:
            Historical scenario analysis results
        """
        logger.info("📈 Analyzing historical scenarios from 2024 market data")
        
        portfolio_returns = self._calculate_portfolio_returns(
            portfolio, portfolio_weights, daily_returns
        )
        
        # Identify distinct historical periods
        historical_periods = self._identify_historical_periods(portfolio_returns, dates)
        
        scenario_results = {}
        
        for period_name, period_data in historical_periods.items():
            start_idx, end_idx = period_data['date_range']
            period_returns = portfolio_returns[start_idx:end_idx+1]
            
            # Analyze this historical period
            period_analysis = {
                'period_name': period_name,
                'start_date': dates[start_idx].strftime('%Y-%m-%d'),
                'end_date': dates[end_idx].strftime('%Y-%m-%d'),
                'duration_days': end_idx - start_idx + 1,
                'total_return': np.sum(period_returns),
                'avg_daily_return': np.mean(period_returns),
                'volatility': np.std(period_returns),
                'max_return': np.max(period_returns),
                'min_return': np.min(period_returns),
                'positive_days': np.sum(period_returns > 0),
                'negative_days': np.sum(period_returns < 0),
                'largest_gain': np.max(period_returns),
                'largest_loss': np.min(period_returns),
                'period_characteristics': period_data['characteristics']
            }
            
            # Calculate drawdown for this period
            cumulative_returns = np.cumsum(period_returns)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdown = cumulative_returns - running_max
            period_analysis['max_drawdown'] = np.min(drawdown)
            
            # Strategy-level analysis for this period
            period_strategy_analysis = self._analyze_strategy_performance_in_period(
                daily_returns[start_idx:end_idx+1], portfolio, strategy_names, period_name
            )
            period_analysis['strategy_performance'] = period_strategy_analysis
            
            scenario_results[period_name] = period_analysis
        
        return scenario_results
    
    def perform_stress_testing(self, 
                             daily_returns: np.ndarray,
                             portfolio: List[int],
                             portfolio_weights: np.ndarray,
                             strategy_names: List[str]) -> Dict[str, Any]:
        """
        Perform stress testing based on actual volatility periods
        
        Args:
            daily_returns: Daily returns matrix
            portfolio: Portfolio indices
            portfolio_weights: Portfolio weights
            strategy_names: Strategy names
            
        Returns:
            Stress testing results
        """
        logger.info("⚡ Performing stress testing based on actual volatility periods")
        
        portfolio_returns = self._calculate_portfolio_returns(
            portfolio, portfolio_weights, daily_returns
        )
        
        stress_scenarios = {}
        
        # 1. Extreme volatility scenarios
        volatility_scenarios = self._generate_volatility_stress_scenarios(
            daily_returns, portfolio, portfolio_weights
        )
        stress_scenarios['volatility_stress'] = volatility_scenarios
        
        # 2. Tail risk scenarios
        tail_risk_scenarios = self._generate_tail_risk_scenarios(
            portfolio_returns, daily_returns, portfolio, portfolio_weights
        )
        stress_scenarios['tail_risk'] = tail_risk_scenarios
        
        # 3. Correlation breakdown scenarios
        correlation_stress_scenarios = self._generate_correlation_stress_scenarios(
            daily_returns, portfolio, portfolio_weights, strategy_names
        )
        stress_scenarios['correlation_stress'] = correlation_stress_scenarios
        
        # 4. Regime change scenarios
        regime_change_scenarios = self._generate_regime_change_scenarios(
            daily_returns, portfolio, portfolio_weights
        )
        stress_scenarios['regime_change'] = regime_change_scenarios
        
        return stress_scenarios
    
    def analyze_market_regimes(self, 
                             daily_returns: np.ndarray,
                             portfolio_returns: np.ndarray,
                             dates: pd.DatetimeIndex) -> Dict[str, Any]:
        """
        Analyze market regimes across 82 trading days
        
        Args:
            daily_returns: Daily returns matrix
            portfolio_returns: Portfolio daily returns
            dates: Trading dates
            
        Returns:
            Market regime analysis results
        """
        logger.info("📊 Analyzing market regimes across trading period")
        
        # Calculate rolling volatility
        volatility_window = self.config.volatility_lookback_days
        rolling_volatility = pd.Series(portfolio_returns).rolling(
            window=min(volatility_window, len(portfolio_returns))
        ).std()
        
        # Identify market regimes
        regime_classification = self._classify_market_regimes(
            portfolio_returns, rolling_volatility.values
        )
        
        # Analyze performance in each regime
        regime_analysis = {}
        unique_regimes = np.unique(regime_classification)
        
        for regime in unique_regimes:
            regime_mask = regime_classification == regime
            regime_returns = portfolio_returns[regime_mask]
            regime_dates = dates[regime_mask]
            
            if len(regime_returns) > 0:
                regime_analysis[regime] = {
                    'regime_name': regime,
                    'days_in_regime': len(regime_returns),
                    'percentage_of_period': len(regime_returns) / len(portfolio_returns) * 100,
                    'avg_daily_return': np.mean(regime_returns),
                    'volatility': np.std(regime_returns),
                    'total_return': np.sum(regime_returns),
                    'max_return': np.max(regime_returns),
                    'min_return': np.min(regime_returns),
                    'regime_dates': [d.strftime('%Y-%m-%d') for d in regime_dates[:5]]  # First 5 dates
                }
        
        # Regime transition analysis
        regime_transitions = self._analyze_regime_transitions(regime_classification)
        
        return {
            'regime_classification': regime_classification.tolist(),
            'regime_analysis': regime_analysis,
            'regime_transitions': regime_transitions,
            'regime_persistence': self._calculate_regime_persistence(regime_classification)
        }
    
    def generate_monte_carlo_scenarios(self, 
                                     daily_returns: np.ndarray,
                                     portfolio: List[int],
                                     portfolio_weights: np.ndarray,
                                     strategy_names: List[str]) -> Dict[str, Any]:
        """
        Generate Monte Carlo scenarios for future performance
        
        Args:
            daily_returns: Daily returns matrix
            portfolio: Portfolio indices
            portfolio_weights: Portfolio weights
            strategy_names: Strategy names
            
        Returns:
            Monte Carlo scenario results
        """
        logger.info("🎲 Generating Monte Carlo scenarios")
        
        portfolio_data = daily_returns[:, portfolio]
        
        # Calculate statistical parameters from historical data
        mean_returns = np.mean(portfolio_data, axis=0)
        cov_matrix = np.cov(portfolio_data.T)
        
        # Generate scenarios
        scenarios = []
        scenario_horizon = self.config.scenario_horizon_days
        
        for scenario_idx in range(self.config.monte_carlo_scenarios):
            # Generate correlated random returns
            random_returns = np.random.multivariate_normal(
                mean_returns, cov_matrix, scenario_horizon
            )
            
            # Calculate portfolio returns for this scenario
            scenario_portfolio_returns = np.sum(
                random_returns * portfolio_weights, axis=1
            )
            
            # Calculate scenario metrics
            scenario_total_return = np.sum(scenario_portfolio_returns)
            scenario_volatility = np.std(scenario_portfolio_returns)
            scenario_max_drawdown = self._calculate_max_drawdown(scenario_portfolio_returns)
            
            scenarios.append({
                'scenario_id': scenario_idx,
                'total_return': scenario_total_return,
                'volatility': scenario_volatility,
                'max_drawdown': scenario_max_drawdown,
                'daily_returns': scenario_portfolio_returns.tolist()
            })
        
        # Analyze scenario distribution
        total_returns = [s['total_return'] for s in scenarios]
        volatilities = [s['volatility'] for s in scenarios]
        max_drawdowns = [s['max_drawdown'] for s in scenarios]
        
        scenario_analysis = {
            'total_return_stats': {
                'mean': np.mean(total_returns),
                'std': np.std(total_returns),
                'min': np.min(total_returns),
                'max': np.max(total_returns),
                'percentiles': {f'{p}th': np.percentile(total_returns, p) 
                              for p in [5, 10, 25, 50, 75, 90, 95, 99]}
            },
            'volatility_stats': {
                'mean': np.mean(volatilities),
                'std': np.std(volatilities),
                'min': np.min(volatilities),
                'max': np.max(volatilities)
            },
            'max_drawdown_stats': {
                'mean': np.mean(max_drawdowns),
                'std': np.std(max_drawdowns),
                'worst': np.min(max_drawdowns)
            }
        }
        
        # Calculate confidence intervals
        confidence_intervals = {}
        for confidence_level in self.config.confidence_levels:
            lower_percentile = (100 - confidence_level) / 2
            upper_percentile = 100 - lower_percentile
            
            confidence_intervals[f'{confidence_level}%'] = {
                'lower_bound': np.percentile(total_returns, lower_percentile),
                'upper_bound': np.percentile(total_returns, upper_percentile)
            }
        
        return {
            'scenarios': scenarios,
            'scenario_analysis': scenario_analysis,
            'confidence_intervals': confidence_intervals,
            'parameters': {
                'num_scenarios': self.config.monte_carlo_scenarios,
                'horizon_days': scenario_horizon,
                'portfolio_size': len(portfolio)
            }
        }
    
    def compare_scenario_outcomes(self, 
                                daily_returns: np.ndarray,
                                portfolio: List[int],
                                portfolio_weights: np.ndarray,
                                strategy_names: List[str]) -> Dict[str, Any]:
        """
        Compare outcomes across different scenario types
        
        Args:
            daily_returns: Daily returns matrix
            portfolio: Portfolio indices
            portfolio_weights: Portfolio weights
            strategy_names: Strategy names
            
        Returns:
            Scenario comparison results
        """
        logger.info("📊 Comparing scenario outcomes")
        
        # Base case (historical performance)
        historical_returns = self._calculate_portfolio_returns(
            portfolio, portfolio_weights, daily_returns
        )
        
        base_case = {
            'scenario_type': 'historical',
            'total_return': np.sum(historical_returns),
            'volatility': np.std(historical_returns),
            'max_drawdown': self._calculate_max_drawdown(historical_returns),
            'sharpe_ratio': np.mean(historical_returns) / np.std(historical_returns) if np.std(historical_returns) > 0 else 0
        }
        
        # Generate alternative scenarios for comparison
        alternative_scenarios = []
        
        # High volatility scenario
        high_vol_returns = self._generate_high_volatility_scenario(
            daily_returns, portfolio, portfolio_weights
        )
        alternative_scenarios.append({
            'scenario_type': 'high_volatility',
            'total_return': np.sum(high_vol_returns),
            'volatility': np.std(high_vol_returns),
            'max_drawdown': self._calculate_max_drawdown(high_vol_returns),
            'sharpe_ratio': np.mean(high_vol_returns) / np.std(high_vol_returns) if np.std(high_vol_returns) > 0 else 0
        })
        
        # Low volatility scenario
        low_vol_returns = self._generate_low_volatility_scenario(
            daily_returns, portfolio, portfolio_weights
        )
        alternative_scenarios.append({
            'scenario_type': 'low_volatility',
            'total_return': np.sum(low_vol_returns),
            'volatility': np.std(low_vol_returns),
            'max_drawdown': self._calculate_max_drawdown(low_vol_returns),
            'sharpe_ratio': np.mean(low_vol_returns) / np.std(low_vol_returns) if np.std(low_vol_returns) > 0 else 0
        })
        
        # Bear market scenario
        bear_market_returns = self._generate_bear_market_scenario(
            daily_returns, portfolio, portfolio_weights
        )
        alternative_scenarios.append({
            'scenario_type': 'bear_market',
            'total_return': np.sum(bear_market_returns),
            'volatility': np.std(bear_market_returns),
            'max_drawdown': self._calculate_max_drawdown(bear_market_returns),
            'sharpe_ratio': np.mean(bear_market_returns) / np.std(bear_market_returns) if np.std(bear_market_returns) > 0 else 0
        })
        
        # Bull market scenario
        bull_market_returns = self._generate_bull_market_scenario(
            daily_returns, portfolio, portfolio_weights
        )
        alternative_scenarios.append({
            'scenario_type': 'bull_market',
            'total_return': np.sum(bull_market_returns),
            'volatility': np.std(bull_market_returns),
            'max_drawdown': self._calculate_max_drawdown(bull_market_returns),
            'sharpe_ratio': np.mean(bull_market_returns) / np.std(bull_market_returns) if np.std(bull_market_returns) > 0 else 0
        })
        
        # Rank scenarios by different metrics
        all_scenarios = [base_case] + alternative_scenarios
        
        rankings = {
            'by_total_return': sorted(all_scenarios, key=lambda x: x['total_return'], reverse=True),
            'by_sharpe_ratio': sorted(all_scenarios, key=lambda x: x['sharpe_ratio'], reverse=True),
            'by_max_drawdown': sorted(all_scenarios, key=lambda x: x['max_drawdown'], reverse=True)  # Higher (less negative) is better
        }
        
        return {
            'base_case': base_case,
            'alternative_scenarios': alternative_scenarios,
            'scenario_rankings': rankings,
            'scenario_comparison_summary': self._generate_scenario_comparison_summary(all_scenarios)
        }
    
    def _calculate_portfolio_returns(self, portfolio: List[int],
                                   portfolio_weights: np.ndarray,
                                   daily_returns: np.ndarray) -> np.ndarray:
        """Calculate portfolio returns from individual strategy returns"""
        portfolio_data = daily_returns[:, portfolio]
        return np.sum(portfolio_data * portfolio_weights, axis=1)
    
    def _identify_historical_periods(self, portfolio_returns: np.ndarray,
                                   dates: pd.DatetimeIndex) -> Dict[str, Any]:
        """Identify distinct historical periods in the data"""
        periods = {}
        
        # Monthly periods
        df = pd.DataFrame({'returns': portfolio_returns, 'date': dates})
        df['month'] = df['date'].dt.to_period('M')
        
        for month, group in df.groupby('month'):
            period_returns = group['returns'].values
            volatility = np.std(period_returns)
            avg_return = np.mean(period_returns)
            
            # Classify period characteristics
            characteristics = []
            if volatility > np.std(portfolio_returns) * 1.2:
                characteristics.append('high_volatility')
            elif volatility < np.std(portfolio_returns) * 0.8:
                characteristics.append('low_volatility')
            
            if avg_return > np.mean(portfolio_returns) * 1.5:
                characteristics.append('strong_performance')
            elif avg_return < np.mean(portfolio_returns) * 0.5:
                characteristics.append('weak_performance')
            
            periods[f'month_{month}'] = {
                'date_range': (group.index[0], group.index[-1]),
                'characteristics': characteristics
            }
        
        # Add specific notable periods
        # First 20 days (market settling period)
        periods['initial_period'] = {
            'date_range': (0, min(19, len(portfolio_returns) - 1)),
            'characteristics': ['initial_settling']
        }
        
        # Last 20 days (recent performance)
        periods['recent_period'] = {
            'date_range': (max(0, len(portfolio_returns) - 20), len(portfolio_returns) - 1),
            'characteristics': ['recent_performance']
        }
        
        return periods
    
    def _analyze_strategy_performance_in_period(self, period_returns: np.ndarray,
                                              portfolio: List[int],
                                              strategy_names: List[str],
                                              period_name: str) -> Dict[str, Any]:
        """Analyze individual strategy performance within a specific period"""
        strategy_analysis = {}
        
        for i, strategy_idx in enumerate(portfolio):
            strategy_returns = period_returns[:, strategy_idx]
            
            strategy_analysis[strategy_names[strategy_idx]] = {
                'total_return': np.sum(strategy_returns),
                'avg_daily_return': np.mean(strategy_returns),
                'volatility': np.std(strategy_returns),
                'best_day': np.max(strategy_returns),
                'worst_day': np.min(strategy_returns),
                'positive_days_pct': np.sum(strategy_returns > 0) / len(strategy_returns) * 100
            }
        
        return strategy_analysis
    
    def _generate_volatility_stress_scenarios(self, daily_returns: np.ndarray,
                                            portfolio: List[int],
                                            portfolio_weights: np.ndarray) -> Dict[str, Any]:
        """Generate stress scenarios based on extreme volatility"""
        portfolio_data = daily_returns[:, portfolio]
        base_volatility = np.std(portfolio_data, axis=0)
        
        scenarios = {}
        
        # 2x volatility scenario
        high_vol_returns = np.random.normal(
            np.mean(portfolio_data, axis=0),
            base_volatility * 2,
            portfolio_data.shape
        )
        high_vol_portfolio_returns = np.sum(high_vol_returns * portfolio_weights, axis=1)
        
        scenarios['2x_volatility'] = {
            'scenario_name': '2x Volatility Stress',
            'volatility_multiplier': 2.0,
            'total_return': np.sum(high_vol_portfolio_returns),
            'volatility': np.std(high_vol_portfolio_returns),
            'max_drawdown': self._calculate_max_drawdown(high_vol_portfolio_returns)
        }
        
        # 3x volatility scenario
        extreme_vol_returns = np.random.normal(
            np.mean(portfolio_data, axis=0),
            base_volatility * 3,
            portfolio_data.shape
        )
        extreme_vol_portfolio_returns = np.sum(extreme_vol_returns * portfolio_weights, axis=1)
        
        scenarios['3x_volatility'] = {
            'scenario_name': '3x Volatility Stress',
            'volatility_multiplier': 3.0,
            'total_return': np.sum(extreme_vol_portfolio_returns),
            'volatility': np.std(extreme_vol_portfolio_returns),
            'max_drawdown': self._calculate_max_drawdown(extreme_vol_portfolio_returns)
        }
        
        return scenarios
    
    def _generate_tail_risk_scenarios(self, portfolio_returns: np.ndarray,
                                    daily_returns: np.ndarray,
                                    portfolio: List[int],
                                    portfolio_weights: np.ndarray) -> Dict[str, Any]:
        """Generate tail risk scenarios"""
        scenarios = {}
        
        for percentile in self.config.stress_test_percentiles:
            if percentile < 50:  # Downside scenarios
                threshold = np.percentile(portfolio_returns, percentile)
                scenario_name = f'worst_{percentile}th_percentile'
                description = f'Worst {percentile}th percentile scenario'
            else:  # Upside scenarios
                threshold = np.percentile(portfolio_returns, percentile)
                scenario_name = f'best_{100-percentile}th_percentile'
                description = f'Best {100-percentile}th percentile scenario'
            
            # Find days matching this percentile
            matching_days = np.where(
                (portfolio_returns <= threshold) if percentile < 50 
                else (portfolio_returns >= threshold)
            )[0]
            
            if len(matching_days) > 0:
                scenarios[scenario_name] = {
                    'scenario_name': description,
                    'threshold': threshold,
                    'matching_days': len(matching_days),
                    'avg_return_on_matching_days': np.mean(portfolio_returns[matching_days]),
                    'worst_single_day': np.min(portfolio_returns[matching_days]) if percentile < 50 else np.max(portfolio_returns[matching_days])
                }
        
        return scenarios
    
    def _generate_correlation_stress_scenarios(self, daily_returns: np.ndarray,
                                             portfolio: List[int],
                                             portfolio_weights: np.ndarray,
                                             strategy_names: List[str]) -> Dict[str, Any]:
        """Generate scenarios with correlation breakdown"""
        portfolio_data = daily_returns[:, portfolio]
        
        scenarios = {}
        
        # Perfect correlation scenario
        mean_return = np.mean(portfolio_data)
        volatility = np.mean(np.std(portfolio_data, axis=0))
        
        perfect_corr_returns = np.random.normal(mean_return, volatility, len(portfolio_data))
        perfect_corr_returns = np.tile(perfect_corr_returns.reshape(-1, 1), (1, len(portfolio)))
        perfect_corr_portfolio_returns = np.sum(perfect_corr_returns * portfolio_weights, axis=1)
        
        scenarios['perfect_correlation'] = {
            'scenario_name': 'Perfect Correlation Stress',
            'correlation_level': 1.0,
            'total_return': np.sum(perfect_corr_portfolio_returns),
            'volatility': np.std(perfect_corr_portfolio_returns),
            'diversification_benefit': 0.0
        }
        
        # Zero correlation scenario
        zero_corr_returns = np.random.normal(
            np.mean(portfolio_data, axis=0),
            np.std(portfolio_data, axis=0),
            portfolio_data.shape
        )
        zero_corr_portfolio_returns = np.sum(zero_corr_returns * portfolio_weights, axis=1)
        
        scenarios['zero_correlation'] = {
            'scenario_name': 'Zero Correlation Scenario',
            'correlation_level': 0.0,
            'total_return': np.sum(zero_corr_portfolio_returns),
            'volatility': np.std(zero_corr_portfolio_returns),
            'diversification_benefit': (np.std(perfect_corr_portfolio_returns) - np.std(zero_corr_portfolio_returns)) / np.std(perfect_corr_portfolio_returns)
        }
        
        return scenarios
    
    def _generate_regime_change_scenarios(self, daily_returns: np.ndarray,
                                        portfolio: List[int],
                                        portfolio_weights: np.ndarray) -> Dict[str, Any]:
        """Generate scenarios with regime changes"""
        portfolio_data = daily_returns[:, portfolio]
        base_returns = self._calculate_portfolio_returns(portfolio, portfolio_weights, daily_returns)
        
        scenarios = {}
        
        # Regime shift to high volatility
        half_point = len(portfolio_data) // 2
        regime_shift_returns = base_returns.copy()
        regime_shift_returns[half_point:] *= 2  # Double volatility in second half
        
        scenarios['mid_period_volatility_spike'] = {
            'scenario_name': 'Mid-Period Volatility Spike',
            'regime_change_point': half_point,
            'first_half_volatility': np.std(regime_shift_returns[:half_point]),
            'second_half_volatility': np.std(regime_shift_returns[half_point:]),
            'total_return': np.sum(regime_shift_returns),
            'max_drawdown': self._calculate_max_drawdown(regime_shift_returns)
        }
        
        return scenarios
    
    def _classify_market_regimes(self, portfolio_returns: np.ndarray,
                               rolling_volatility: np.ndarray) -> np.ndarray:
        """Classify market regimes based on returns and volatility"""
        regimes = np.full(len(portfolio_returns), 'normal', dtype=object)
        
        thresholds = self.config.market_regime_thresholds
        
        for i, (ret, vol) in enumerate(zip(portfolio_returns, rolling_volatility)):
            if not np.isnan(vol):
                if vol > thresholds['high_volatility']:
                    regimes[i] = 'high_volatility'
                elif vol < thresholds['low_volatility']:
                    regimes[i] = 'low_volatility'
                elif ret > thresholds['bull_market']:
                    regimes[i] = 'bull_market'
                elif ret < thresholds['bear_market']:
                    regimes[i] = 'bear_market'
        
        return regimes
    
    def _analyze_regime_transitions(self, regime_classification: np.ndarray) -> Dict[str, Any]:
        """Analyze transitions between market regimes"""
        transitions = {}
        
        for i in range(1, len(regime_classification)):
            prev_regime = regime_classification[i-1]
            curr_regime = regime_classification[i]
            
            if prev_regime != curr_regime:
                transition_key = f'{prev_regime}_to_{curr_regime}'
                transitions[transition_key] = transitions.get(transition_key, 0) + 1
        
        return {
            'transition_counts': transitions,
            'total_transitions': sum(transitions.values()),
            'most_common_transition': max(transitions.items(), key=lambda x: x[1]) if transitions else None
        }
    
    def _calculate_regime_persistence(self, regime_classification: np.ndarray) -> Dict[str, Any]:
        """Calculate how persistent each regime is"""
        unique_regimes = np.unique(regime_classification)
        persistence = {}
        
        for regime in unique_regimes:
            regime_days = np.where(regime_classification == regime)[0]
            
            # Calculate average run length
            runs = []
            current_run = 1
            
            for i in range(1, len(regime_classification)):
                if regime_classification[i] == regime and regime_classification[i-1] == regime:
                    current_run += 1
                elif regime_classification[i-1] == regime:
                    runs.append(current_run)
                    current_run = 1
            
            # Don't forget the last run if it ends with the regime
            if len(regime_classification) > 0 and regime_classification[-1] == regime:
                runs.append(current_run)
            
            persistence[regime] = {
                'total_days': len(regime_days),
                'average_run_length': np.mean(runs) if runs else 0,
                'longest_run': max(runs) if runs else 0,
                'number_of_runs': len(runs)
            }
        
        return persistence
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown from returns series"""
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = cumulative - running_max
        return np.min(drawdown)
    
    def _generate_high_volatility_scenario(self, daily_returns: np.ndarray,
                                         portfolio: List[int],
                                         portfolio_weights: np.ndarray) -> np.ndarray:
        """Generate high volatility scenario"""
        portfolio_data = daily_returns[:, portfolio]
        mean_returns = np.mean(portfolio_data, axis=0)
        volatilities = np.std(portfolio_data, axis=0) * 1.5  # 50% higher volatility
        
        scenario_returns = np.random.normal(
            mean_returns, volatilities, portfolio_data.shape
        )
        return np.sum(scenario_returns * portfolio_weights, axis=1)
    
    def _generate_low_volatility_scenario(self, daily_returns: np.ndarray,
                                        portfolio: List[int],
                                        portfolio_weights: np.ndarray) -> np.ndarray:
        """Generate low volatility scenario"""
        portfolio_data = daily_returns[:, portfolio]
        mean_returns = np.mean(portfolio_data, axis=0)
        volatilities = np.std(portfolio_data, axis=0) * 0.5  # 50% lower volatility
        
        scenario_returns = np.random.normal(
            mean_returns, volatilities, portfolio_data.shape
        )
        return np.sum(scenario_returns * portfolio_weights, axis=1)
    
    def _generate_bear_market_scenario(self, daily_returns: np.ndarray,
                                     portfolio: List[int],
                                     portfolio_weights: np.ndarray) -> np.ndarray:
        """Generate bear market scenario"""
        portfolio_data = daily_returns[:, portfolio]
        mean_returns = np.mean(portfolio_data, axis=0) - 0.002  # Shift down by 0.2% daily
        volatilities = np.std(portfolio_data, axis=0) * 1.2  # 20% higher volatility
        
        scenario_returns = np.random.normal(
            mean_returns, volatilities, portfolio_data.shape
        )
        return np.sum(scenario_returns * portfolio_weights, axis=1)
    
    def _generate_bull_market_scenario(self, daily_returns: np.ndarray,
                                     portfolio: List[int],
                                     portfolio_weights: np.ndarray) -> np.ndarray:
        """Generate bull market scenario"""
        portfolio_data = daily_returns[:, portfolio]
        mean_returns = np.mean(portfolio_data, axis=0) + 0.001  # Shift up by 0.1% daily
        volatilities = np.std(portfolio_data, axis=0) * 0.8  # 20% lower volatility
        
        scenario_returns = np.random.normal(
            mean_returns, volatilities, portfolio_data.shape
        )
        return np.sum(scenario_returns * portfolio_weights, axis=1)
    
    def _generate_scenario_comparison_summary(self, all_scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary comparison of all scenarios"""
        return {
            'best_total_return': max(s['total_return'] for s in all_scenarios),
            'worst_total_return': min(s['total_return'] for s in all_scenarios),
            'best_sharpe_ratio': max(s['sharpe_ratio'] for s in all_scenarios),
            'worst_max_drawdown': min(s['max_drawdown'] for s in all_scenarios),
            'return_range': max(s['total_return'] for s in all_scenarios) - min(s['total_return'] for s in all_scenarios),
            'scenario_count': len(all_scenarios)
        }
    
    def _generate_scenario_summary(self, scenario_results: Dict[str, Any],
                                 portfolio_returns: np.ndarray) -> Dict[str, Any]:
        """Generate comprehensive scenario analysis summary"""
        return {
            'analysis_timestamp': datetime.now().isoformat(),
            'historical_performance': {
                'total_return': np.sum(portfolio_returns),
                'volatility': np.std(portfolio_returns),
                'max_drawdown': self._calculate_max_drawdown(portfolio_returns)
            },
            'scenario_types_analyzed': list(scenario_results.keys()),
            'key_insights': self._extract_key_scenario_insights(scenario_results),
            'risk_warnings': self._identify_scenario_risk_warnings(scenario_results)
        }
    
    def _extract_key_scenario_insights(self, scenario_results: Dict[str, Any]) -> List[str]:
        """Extract key insights from scenario analysis"""
        insights = []
        
        # Add insights based on analysis results
        if 'monte_carlo_scenarios' in scenario_results:
            mc_results = scenario_results['monte_carlo_scenarios']
            confidence_90 = mc_results['confidence_intervals'].get('90%', {})
            
            if confidence_90:
                insights.append(
                    f"90% confidence interval for future returns: "
                    f"{confidence_90['lower_bound']:.2f} to {confidence_90['upper_bound']:.2f}"
                )
        
        if 'market_regime_analysis' in scenario_results:
            regime_analysis = scenario_results['market_regime_analysis']['regime_analysis']
            
            # Find most common regime
            regime_durations = {k: v['days_in_regime'] for k, v in regime_analysis.items()}
            if regime_durations:
                most_common_regime = max(regime_durations, key=regime_durations.get)
                insights.append(f"Most common market regime: {most_common_regime}")
        
        return insights
    
    def _identify_scenario_risk_warnings(self, scenario_results: Dict[str, Any]) -> List[str]:
        """Identify potential risk warnings from scenario analysis"""
        warnings = []
        
        # Check stress test results
        if 'stress_scenarios' in scenario_results:
            stress_results = scenario_results['stress_scenarios']
            
            if 'volatility_stress' in stress_results:
                vol_scenarios = stress_results['volatility_stress']
                for scenario_name, scenario_data in vol_scenarios.items():
                    if scenario_data.get('max_drawdown', 0) < -0.20:  # More than 20% drawdown
                        warnings.append(
                            f"High stress scenario '{scenario_name}' shows significant drawdown risk"
                        )
        
        return warnings