"""
Sensitivity Analysis Module

Performs sensitivity analysis for portfolio optimization parameters including:
- Correlation penalty sensitivity
- Portfolio size sensitivity  
- Risk parameter sensitivity
- Stop Loss/Take Profit threshold sensitivity
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import warnings

logger = logging.getLogger(__name__)


@dataclass
class SensitivityConfig:
    """Configuration for sensitivity analysis"""
    correlation_penalty_range: Tuple[float, float] = (0.0, 0.5)
    correlation_penalty_steps: int = 11
    portfolio_size_range: Tuple[int, int] = (10, 100)
    portfolio_size_steps: int = 10
    risk_param_range: Tuple[float, float] = (0.5, 2.0)
    risk_param_steps: int = 8
    sl_tp_range: Tuple[float, float] = (0.05, 0.95)
    sl_tp_steps: int = 10
    monte_carlo_runs: int = 100
    parallel_execution: bool = True
    max_workers: int = 4


class SensitivityAnalyzer:
    """
    Performs comprehensive sensitivity analysis for portfolio optimization.
    
    Analyzes sensitivity to:
    - Correlation penalty weights (using 25,544×25,544 correlation matrix)
    - Portfolio sizes (10-100 strategies from production universe)
    - Risk parameters using actual volatility from 82-day period
    - Stop Loss/Take Profit threshold variations
    """
    
    def __init__(self, config: Optional[SensitivityConfig] = None):
        """
        Initialize sensitivity analyzer
        
        Args:
            config: Sensitivity analysis configuration
        """
        self.config = config or SensitivityConfig()
        self.base_results = None
        self.sensitivity_cache = {}
        
    def perform_comprehensive_sensitivity_analysis(self, 
                                                 daily_returns: np.ndarray,
                                                 strategy_names: List[str],
                                                 correlation_matrix: np.ndarray,
                                                 optimization_function: Callable,
                                                 base_portfolio: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Perform comprehensive sensitivity analysis across all parameters
        
        Args:
            daily_returns: Daily returns matrix (days x strategies)
            strategy_names: Names of all strategies
            correlation_matrix: Strategy correlation matrix
            optimization_function: Function to call for optimization
            base_portfolio: Base portfolio for comparison
            
        Returns:
            Dictionary with comprehensive sensitivity results
        """
        logger.info("🔍 Starting comprehensive sensitivity analysis")
        
        sensitivity_results = {
            'correlation_penalty_sensitivity': self.analyze_correlation_penalty_sensitivity(
                daily_returns, strategy_names, correlation_matrix, optimization_function
            ),
            'portfolio_size_sensitivity': self.analyze_portfolio_size_sensitivity(
                daily_returns, strategy_names, optimization_function
            ),
            'risk_parameter_sensitivity': self.analyze_risk_parameter_sensitivity(
                daily_returns, strategy_names, optimization_function
            ),
            'stop_loss_sensitivity': self.analyze_stop_loss_sensitivity(
                daily_returns, strategy_names, optimization_function
            ),
            'take_profit_sensitivity': self.analyze_take_profit_sensitivity(
                daily_returns, strategy_names, optimization_function
            ),
            'monte_carlo_sensitivity': self.perform_monte_carlo_sensitivity(
                daily_returns, strategy_names, optimization_function
            )
        }
        
        # Add summary analysis
        sensitivity_results['sensitivity_summary'] = self._generate_sensitivity_summary(
            sensitivity_results, base_portfolio
        )
        
        logger.info("✅ Comprehensive sensitivity analysis completed")
        return sensitivity_results
    
    def analyze_correlation_penalty_sensitivity(self, 
                                              daily_returns: np.ndarray,
                                              strategy_names: List[str],
                                              correlation_matrix: np.ndarray,
                                              optimization_function: Callable) -> Dict[str, Any]:
        """
        Analyze sensitivity to correlation penalty weights
        
        Args:
            daily_returns: Daily returns matrix
            strategy_names: Strategy names
            correlation_matrix: Pre-calculated correlation matrix
            optimization_function: Optimization function to test
            
        Returns:
            Correlation penalty sensitivity results
        """
        logger.info("📊 Analyzing correlation penalty sensitivity")
        
        penalty_values = np.linspace(
            self.config.correlation_penalty_range[0],
            self.config.correlation_penalty_range[1],
            self.config.correlation_penalty_steps
        )
        
        results = {}
        
        for penalty in penalty_values:
            logger.info(f"🔍 Testing correlation penalty: {penalty:.3f}")
            
            try:
                # Run optimization with specific penalty
                optimization_result = optimization_function(
                    daily_returns=daily_returns,
                    correlation_penalty=penalty,
                    correlation_matrix=correlation_matrix
                )
                
                if optimization_result and 'best_portfolio' in optimization_result:
                    portfolio = optimization_result['best_portfolio']
                    
                    # Calculate metrics for this penalty level
                    portfolio_metrics = self._calculate_portfolio_metrics(
                        portfolio, daily_returns, strategy_names, correlation_matrix
                    )
                    
                    results[f'penalty_{penalty:.3f}'] = {
                        'penalty_weight': penalty,
                        'portfolio': portfolio,
                        'fitness': optimization_result.get('best_fitness', 0),
                        'portfolio_size': len(portfolio),
                        'metrics': portfolio_metrics
                    }
                
            except Exception as e:
                logger.warning(f"⚠️  Correlation penalty {penalty:.3f} failed: {e}")
                results[f'penalty_{penalty:.3f}'] = {
                    'penalty_weight': penalty,
                    'error': str(e),
                    'status': 'failed'
                }
        
        # Analyze sensitivity patterns
        sensitivity_analysis = self._analyze_correlation_penalty_patterns(results)
        
        return {
            'penalty_results': results,
            'sensitivity_patterns': sensitivity_analysis,
            'optimal_penalty_range': self._find_optimal_penalty_range(results)
        }
    
    def analyze_portfolio_size_sensitivity(self, 
                                         daily_returns: np.ndarray,
                                         strategy_names: List[str],
                                         optimization_function: Callable) -> Dict[str, Any]:
        """
        Analyze sensitivity to portfolio size variations
        
        Args:
            daily_returns: Daily returns matrix
            strategy_names: Strategy names  
            optimization_function: Optimization function to test
            
        Returns:
            Portfolio size sensitivity results
        """
        logger.info("📊 Analyzing portfolio size sensitivity")
        
        size_values = np.linspace(
            self.config.portfolio_size_range[0],
            self.config.portfolio_size_range[1],
            self.config.portfolio_size_steps,
            dtype=int
        )
        
        results = {}
        
        for portfolio_size in size_values:
            logger.info(f"🔍 Testing portfolio size: {portfolio_size}")
            
            try:
                # Run optimization with specific portfolio size
                optimization_result = optimization_function(
                    daily_returns=daily_returns,
                    portfolio_size=portfolio_size
                )
                
                if optimization_result and 'best_portfolio' in optimization_result:
                    portfolio = optimization_result['best_portfolio']
                    
                    # Calculate metrics for this size
                    portfolio_metrics = self._calculate_portfolio_metrics(
                        portfolio, daily_returns, strategy_names
                    )
                    
                    results[f'size_{portfolio_size}'] = {
                        'target_size': portfolio_size,
                        'actual_size': len(portfolio),
                        'portfolio': portfolio,
                        'fitness': optimization_result.get('best_fitness', 0),
                        'metrics': portfolio_metrics
                    }
                
            except Exception as e:
                logger.warning(f"⚠️  Portfolio size {portfolio_size} failed: {e}")
                results[f'size_{portfolio_size}'] = {
                    'target_size': portfolio_size,
                    'error': str(e),
                    'status': 'failed'
                }
        
        # Analyze size sensitivity patterns
        sensitivity_analysis = self._analyze_size_sensitivity_patterns(results)
        
        return {
            'size_results': results,
            'sensitivity_patterns': sensitivity_analysis,
            'optimal_size_range': self._find_optimal_size_range(results)
        }
    
    def analyze_risk_parameter_sensitivity(self, 
                                         daily_returns: np.ndarray,
                                         strategy_names: List[str],
                                         optimization_function: Callable) -> Dict[str, Any]:
        """
        Analyze sensitivity to risk parameter variations
        
        Args:
            daily_returns: Daily returns matrix
            strategy_names: Strategy names
            optimization_function: Optimization function to test
            
        Returns:
            Risk parameter sensitivity results
        """
        logger.info("📊 Analyzing risk parameter sensitivity")
        
        risk_values = np.linspace(
            self.config.risk_param_range[0],
            self.config.risk_param_range[1],
            self.config.risk_param_steps
        )
        
        results = {}
        
        for risk_param in risk_values:
            logger.info(f"🔍 Testing risk parameter: {risk_param:.3f}")
            
            try:
                # Run optimization with specific risk parameter
                optimization_result = optimization_function(
                    daily_returns=daily_returns,
                    risk_parameter=risk_param
                )
                
                if optimization_result and 'best_portfolio' in optimization_result:
                    portfolio = optimization_result['best_portfolio']
                    
                    # Calculate risk-adjusted metrics
                    portfolio_metrics = self._calculate_risk_adjusted_metrics(
                        portfolio, daily_returns, strategy_names, risk_param
                    )
                    
                    results[f'risk_{risk_param:.3f}'] = {
                        'risk_parameter': risk_param,
                        'portfolio': portfolio,
                        'fitness': optimization_result.get('best_fitness', 0),
                        'portfolio_size': len(portfolio),
                        'metrics': portfolio_metrics
                    }
                
            except Exception as e:
                logger.warning(f"⚠️  Risk parameter {risk_param:.3f} failed: {e}")
                results[f'risk_{risk_param:.3f}'] = {
                    'risk_parameter': risk_param,
                    'error': str(e),
                    'status': 'failed'
                }
        
        return {
            'risk_results': results,
            'sensitivity_patterns': self._analyze_risk_sensitivity_patterns(results),
            'optimal_risk_range': self._find_optimal_risk_range(results)
        }
    
    def analyze_stop_loss_sensitivity(self, 
                                    daily_returns: np.ndarray,
                                    strategy_names: List[str],
                                    optimization_function: Callable) -> Dict[str, Any]:
        """
        Analyze sensitivity to Stop Loss threshold variations
        
        Args:
            daily_returns: Daily returns matrix
            strategy_names: Strategy names
            optimization_function: Optimization function to test
            
        Returns:
            Stop Loss sensitivity results
        """
        logger.info("📊 Analyzing Stop Loss threshold sensitivity")
        
        sl_values = np.linspace(
            self.config.sl_tp_range[0],
            self.config.sl_tp_range[1],
            self.config.sl_tp_steps
        )
        
        results = {}
        
        for sl_threshold in sl_values:
            logger.info(f"🔍 Testing Stop Loss threshold: {sl_threshold:.3f}")
            
            try:
                # Filter strategies by Stop Loss level
                filtered_strategies = self._filter_strategies_by_sl(
                    strategy_names, sl_threshold
                )
                
                if len(filtered_strategies) < 10:  # Minimum portfolio size
                    results[f'sl_{sl_threshold:.3f}'] = {
                        'sl_threshold': sl_threshold,
                        'error': 'Insufficient strategies for portfolio',
                        'status': 'failed'
                    }
                    continue
                
                # Run optimization with filtered strategies
                filtered_returns = daily_returns[:, filtered_strategies]
                filtered_names = [strategy_names[i] for i in filtered_strategies]
                
                optimization_result = optimization_function(
                    daily_returns=filtered_returns,
                    strategy_names=filtered_names
                )
                
                if optimization_result and 'best_portfolio' in optimization_result:
                    portfolio = optimization_result['best_portfolio']
                    
                    # Map back to original indices
                    original_portfolio = [filtered_strategies[i] for i in portfolio]
                    
                    portfolio_metrics = self._calculate_portfolio_metrics(
                        original_portfolio, daily_returns, strategy_names
                    )
                    
                    results[f'sl_{sl_threshold:.3f}'] = {
                        'sl_threshold': sl_threshold,
                        'available_strategies': len(filtered_strategies),
                        'portfolio': original_portfolio,
                        'fitness': optimization_result.get('best_fitness', 0),
                        'portfolio_size': len(original_portfolio),
                        'metrics': portfolio_metrics
                    }
                
            except Exception as e:
                logger.warning(f"⚠️  Stop Loss {sl_threshold:.3f} failed: {e}")
                results[f'sl_{sl_threshold:.3f}'] = {
                    'sl_threshold': sl_threshold,
                    'error': str(e),
                    'status': 'failed'
                }
        
        return {
            'sl_results': results,
            'sensitivity_patterns': self._analyze_sl_sensitivity_patterns(results)
        }
    
    def analyze_take_profit_sensitivity(self, 
                                      daily_returns: np.ndarray,
                                      strategy_names: List[str],
                                      optimization_function: Callable) -> Dict[str, Any]:
        """
        Analyze sensitivity to Take Profit threshold variations
        """
        logger.info("📊 Analyzing Take Profit threshold sensitivity")
        
        tp_values = np.linspace(
            self.config.sl_tp_range[0],
            self.config.sl_tp_range[1],
            self.config.sl_tp_steps
        )
        
        results = {}
        
        for tp_threshold in tp_values:
            logger.info(f"🔍 Testing Take Profit threshold: {tp_threshold:.3f}")
            
            try:
                # Filter strategies by Take Profit level
                filtered_strategies = self._filter_strategies_by_tp(
                    strategy_names, tp_threshold
                )
                
                if len(filtered_strategies) < 10:  # Minimum portfolio size
                    results[f'tp_{tp_threshold:.3f}'] = {
                        'tp_threshold': tp_threshold,
                        'error': 'Insufficient strategies for portfolio',
                        'status': 'failed'
                    }
                    continue
                
                # Run optimization with filtered strategies
                filtered_returns = daily_returns[:, filtered_strategies]
                filtered_names = [strategy_names[i] for i in filtered_strategies]
                
                optimization_result = optimization_function(
                    daily_returns=filtered_returns,
                    strategy_names=filtered_names
                )
                
                if optimization_result and 'best_portfolio' in optimization_result:
                    portfolio = optimization_result['best_portfolio']
                    original_portfolio = [filtered_strategies[i] for i in portfolio]
                    
                    portfolio_metrics = self._calculate_portfolio_metrics(
                        original_portfolio, daily_returns, strategy_names
                    )
                    
                    results[f'tp_{tp_threshold:.3f}'] = {
                        'tp_threshold': tp_threshold,
                        'available_strategies': len(filtered_strategies),
                        'portfolio': original_portfolio,
                        'fitness': optimization_result.get('best_fitness', 0),
                        'portfolio_size': len(original_portfolio),
                        'metrics': portfolio_metrics
                    }
                
            except Exception as e:
                logger.warning(f"⚠️  Take Profit {tp_threshold:.3f} failed: {e}")
                results[f'tp_{tp_threshold:.3f}'] = {
                    'tp_threshold': tp_threshold,
                    'error': str(e),
                    'status': 'failed'
                }
        
        return {
            'tp_results': results,
            'sensitivity_patterns': self._analyze_tp_sensitivity_patterns(results)
        }
    
    def perform_monte_carlo_sensitivity(self, 
                                      daily_returns: np.ndarray,
                                      strategy_names: List[str],
                                      optimization_function: Callable) -> Dict[str, Any]:
        """
        Perform Monte Carlo sensitivity analysis with parameter variations
        """
        logger.info("🎲 Performing Monte Carlo sensitivity analysis")
        
        results = []
        
        for run in range(self.config.monte_carlo_runs):
            if run % 10 == 0:
                logger.info(f"🎲 Monte Carlo run {run + 1}/{self.config.monte_carlo_runs}")
            
            # Generate random parameter values
            random_params = {
                'correlation_penalty': np.random.uniform(*self.config.correlation_penalty_range),
                'portfolio_size': np.random.randint(*self.config.portfolio_size_range),
                'risk_parameter': np.random.uniform(*self.config.risk_param_range)
            }
            
            try:
                optimization_result = optimization_function(
                    daily_returns=daily_returns,
                    **random_params
                )
                
                if optimization_result and 'best_portfolio' in optimization_result:
                    portfolio = optimization_result['best_portfolio']
                    
                    portfolio_metrics = self._calculate_portfolio_metrics(
                        portfolio, daily_returns, strategy_names
                    )
                    
                    results.append({
                        'run': run,
                        'parameters': random_params,
                        'portfolio': portfolio,
                        'fitness': optimization_result.get('best_fitness', 0),
                        'metrics': portfolio_metrics
                    })
            
            except Exception as e:
                logger.warning(f"⚠️  Monte Carlo run {run} failed: {e}")
        
        # Analyze Monte Carlo results
        mc_analysis = self._analyze_monte_carlo_results(results)
        
        return {
            'monte_carlo_runs': results,
            'analysis': mc_analysis,
            'parameter_correlations': self._calculate_parameter_correlations(results)
        }
    
    def _calculate_portfolio_metrics(self, portfolio: List[int], 
                                   daily_returns: np.ndarray,
                                   strategy_names: List[str],
                                   correlation_matrix: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Calculate standard portfolio metrics"""
        if not portfolio:
            return {}
        
        portfolio_returns = daily_returns[:, portfolio]
        equal_weights = np.ones(len(portfolio)) / len(portfolio)
        portfolio_daily_returns = np.sum(portfolio_returns * equal_weights, axis=1)
        
        metrics = {
            'total_return': np.sum(portfolio_daily_returns),
            'avg_daily_return': np.mean(portfolio_daily_returns),
            'volatility': np.std(portfolio_daily_returns),
            'sharpe_ratio': np.mean(portfolio_daily_returns) / np.std(portfolio_daily_returns) if np.std(portfolio_daily_returns) > 0 else 0,
            'max_return': np.max(portfolio_daily_returns),
            'min_return': np.min(portfolio_daily_returns)
        }
        
        # Add correlation-based metrics if matrix provided
        if correlation_matrix is not None and len(portfolio) > 1:
            portfolio_corr_matrix = correlation_matrix[np.ix_(portfolio, portfolio)]
            upper_triangle = np.triu(portfolio_corr_matrix, k=1)
            correlations = upper_triangle[upper_triangle != 0]
            
            metrics.update({
                'avg_correlation': np.mean(np.abs(correlations)),
                'max_correlation': np.max(np.abs(correlations)),
                'correlation_std': np.std(correlations)
            })
        
        return metrics
    
    def _calculate_risk_adjusted_metrics(self, portfolio: List[int],
                                       daily_returns: np.ndarray,
                                       strategy_names: List[str],
                                       risk_parameter: float) -> Dict[str, Any]:
        """Calculate risk-adjusted portfolio metrics"""
        base_metrics = self._calculate_portfolio_metrics(portfolio, daily_returns, strategy_names)
        
        # Add risk-adjusted calculations
        if base_metrics:
            portfolio_returns = daily_returns[:, portfolio]
            equal_weights = np.ones(len(portfolio)) / len(portfolio)
            portfolio_daily_returns = np.sum(portfolio_returns * equal_weights, axis=1)
            
            # Risk-adjusted return
            risk_adjusted_return = base_metrics['avg_daily_return'] / (base_metrics['volatility'] * risk_parameter)
            
            base_metrics.update({
                'risk_parameter': risk_parameter,
                'risk_adjusted_return': risk_adjusted_return,
                'risk_penalty': base_metrics['volatility'] * risk_parameter
            })
        
        return base_metrics
    
    def _filter_strategies_by_sl(self, strategy_names: List[str], sl_threshold: float) -> List[int]:
        """Filter strategies by Stop Loss threshold"""
        import re
        filtered_indices = []
        
        for i, name in enumerate(strategy_names):
            sl_match = re.search(r'SL(\d+)', name)
            if sl_match:
                sl_value = int(sl_match.group(1)) / 100.0  # Convert to decimal
                if sl_value >= sl_threshold:
                    filtered_indices.append(i)
        
        return filtered_indices
    
    def _filter_strategies_by_tp(self, strategy_names: List[str], tp_threshold: float) -> List[int]:
        """Filter strategies by Take Profit threshold"""
        import re
        filtered_indices = []
        
        for i, name in enumerate(strategy_names):
            tp_match = re.search(r'TP(\d+)', name)
            if tp_match:
                tp_value = int(tp_match.group(1)) / 100.0  # Convert to decimal
                if tp_value >= tp_threshold:
                    filtered_indices.append(i)
        
        return filtered_indices
    
    def _analyze_correlation_penalty_patterns(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze patterns in correlation penalty sensitivity"""
        successful_results = {k: v for k, v in results.items() if 'error' not in v}
        
        if not successful_results:
            return {'error': 'No successful results to analyze'}
        
        penalty_values = [v['penalty_weight'] for v in successful_results.values()]
        fitness_values = [v['fitness'] for v in successful_results.values()]
        portfolio_sizes = [v['portfolio_size'] for v in successful_results.values()]
        
        return {
            'penalty_range': (min(penalty_values), max(penalty_values)),
            'fitness_range': (min(fitness_values), max(fitness_values)),
            'fitness_correlation_with_penalty': np.corrcoef(penalty_values, fitness_values)[0, 1],
            'size_correlation_with_penalty': np.corrcoef(penalty_values, portfolio_sizes)[0, 1],
            'optimal_penalty': penalty_values[np.argmax(fitness_values)]
        }
    
    def _analyze_size_sensitivity_patterns(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze patterns in portfolio size sensitivity"""
        successful_results = {k: v for k, v in results.items() if 'error' not in v}
        
        if not successful_results:
            return {'error': 'No successful results to analyze'}
        
        sizes = [v['target_size'] for v in successful_results.values()]
        fitness_values = [v['fitness'] for v in successful_results.values()]
        
        return {
            'size_range': (min(sizes), max(sizes)),
            'fitness_range': (min(fitness_values), max(fitness_values)),
            'fitness_correlation_with_size': np.corrcoef(sizes, fitness_values)[0, 1],
            'optimal_size': sizes[np.argmax(fitness_values)]
        }
    
    def _analyze_risk_sensitivity_patterns(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze patterns in risk parameter sensitivity"""
        successful_results = {k: v for k, v in results.items() if 'error' not in v}
        
        if not successful_results:
            return {'error': 'No successful results to analyze'}
        
        risk_params = [v['risk_parameter'] for v in successful_results.values()]
        fitness_values = [v['fitness'] for v in successful_results.values()]
        
        return {
            'risk_range': (min(risk_params), max(risk_params)),
            'fitness_range': (min(fitness_values), max(fitness_values)),
            'fitness_correlation_with_risk': np.corrcoef(risk_params, fitness_values)[0, 1],
            'optimal_risk_parameter': risk_params[np.argmax(fitness_values)]
        }
    
    def _analyze_sl_sensitivity_patterns(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze patterns in Stop Loss sensitivity"""
        successful_results = {k: v for k, v in results.items() if 'error' not in v}
        
        if not successful_results:
            return {'error': 'No successful results to analyze'}
        
        sl_thresholds = [v['sl_threshold'] for v in successful_results.values()]
        fitness_values = [v['fitness'] for v in successful_results.values()]
        available_strategies = [v['available_strategies'] for v in successful_results.values()]
        
        return {
            'sl_range': (min(sl_thresholds), max(sl_thresholds)),
            'fitness_range': (min(fitness_values), max(fitness_values)),
            'strategy_availability': {
                'min': min(available_strategies),
                'max': max(available_strategies),
                'avg': np.mean(available_strategies)
            },
            'optimal_sl_threshold': sl_thresholds[np.argmax(fitness_values)]
        }
    
    def _analyze_tp_sensitivity_patterns(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze patterns in Take Profit sensitivity"""
        successful_results = {k: v for k, v in results.items() if 'error' not in v}
        
        if not successful_results:
            return {'error': 'No successful results to analyze'}
        
        tp_thresholds = [v['tp_threshold'] for v in successful_results.values()]
        fitness_values = [v['fitness'] for v in successful_results.values()]
        available_strategies = [v['available_strategies'] for v in successful_results.values()]
        
        return {
            'tp_range': (min(tp_thresholds), max(tp_thresholds)),
            'fitness_range': (min(fitness_values), max(fitness_values)),
            'strategy_availability': {
                'min': min(available_strategies),
                'max': max(available_strategies),
                'avg': np.mean(available_strategies)
            },
            'optimal_tp_threshold': tp_thresholds[np.argmax(fitness_values)]
        }
    
    def _analyze_monte_carlo_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze Monte Carlo sensitivity results"""
        if not results:
            return {'error': 'No Monte Carlo results to analyze'}
        
        fitness_values = [r['fitness'] for r in results]
        correlation_penalties = [r['parameters']['correlation_penalty'] for r in results]
        portfolio_sizes = [r['parameters']['portfolio_size'] for r in results]
        risk_parameters = [r['parameters']['risk_parameter'] for r in results]
        
        return {
            'fitness_statistics': {
                'mean': np.mean(fitness_values),
                'std': np.std(fitness_values),
                'min': np.min(fitness_values),
                'max': np.max(fitness_values),
                'percentiles': {
                    '25th': np.percentile(fitness_values, 25),
                    '50th': np.percentile(fitness_values, 50),
                    '75th': np.percentile(fitness_values, 75),
                    '95th': np.percentile(fitness_values, 95)
                }
            },
            'parameter_statistics': {
                'correlation_penalty': {
                    'mean': np.mean(correlation_penalties),
                    'std': np.std(correlation_penalties)
                },
                'portfolio_size': {
                    'mean': np.mean(portfolio_sizes),
                    'std': np.std(portfolio_sizes)
                },
                'risk_parameter': {
                    'mean': np.mean(risk_parameters),
                    'std': np.std(risk_parameters)
                }
            },
            'best_run': results[np.argmax(fitness_values)],
            'success_rate': len(results) / self.config.monte_carlo_runs
        }
    
    def _calculate_parameter_correlations(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate correlations between parameters and outcomes"""
        if not results:
            return {}
        
        fitness_values = np.array([r['fitness'] for r in results])
        correlation_penalties = np.array([r['parameters']['correlation_penalty'] for r in results])
        portfolio_sizes = np.array([r['parameters']['portfolio_size'] for r in results])
        risk_parameters = np.array([r['parameters']['risk_parameter'] for r in results])
        
        return {
            'fitness_vs_correlation_penalty': np.corrcoef(fitness_values, correlation_penalties)[0, 1],
            'fitness_vs_portfolio_size': np.corrcoef(fitness_values, portfolio_sizes)[0, 1],
            'fitness_vs_risk_parameter': np.corrcoef(fitness_values, risk_parameters)[0, 1],
            'correlation_penalty_vs_portfolio_size': np.corrcoef(correlation_penalties, portfolio_sizes)[0, 1],
            'correlation_penalty_vs_risk_parameter': np.corrcoef(correlation_penalties, risk_parameters)[0, 1],
            'portfolio_size_vs_risk_parameter': np.corrcoef(portfolio_sizes, risk_parameters)[0, 1]
        }
    
    def _find_optimal_penalty_range(self, results: Dict[str, Any]) -> Tuple[float, float]:
        """Find optimal correlation penalty range based on results"""
        successful_results = {k: v for k, v in results.items() if 'error' not in v}
        
        if not successful_results:
            return (0.1, 0.3)  # Default range
        
        # Find top 20% of results
        fitness_values = [(k, v['fitness']) for k, v in successful_results.items()]
        fitness_values.sort(key=lambda x: x[1], reverse=True)
        top_20_pct = fitness_values[:max(1, len(fitness_values) // 5)]
        
        # Extract penalty values from top performers
        top_penalties = [successful_results[k]['penalty_weight'] for k, _ in top_20_pct]
        
        return (min(top_penalties), max(top_penalties))
    
    def _find_optimal_size_range(self, results: Dict[str, Any]) -> Tuple[int, int]:
        """Find optimal portfolio size range based on results"""
        successful_results = {k: v for k, v in results.items() if 'error' not in v}
        
        if not successful_results:
            return (20, 50)  # Default range
        
        # Find top 20% of results
        fitness_values = [(k, v['fitness']) for k, v in successful_results.items()]
        fitness_values.sort(key=lambda x: x[1], reverse=True)
        top_20_pct = fitness_values[:max(1, len(fitness_values) // 5)]
        
        # Extract sizes from top performers
        top_sizes = [successful_results[k]['target_size'] for k, _ in top_20_pct]
        
        return (min(top_sizes), max(top_sizes))
    
    def _find_optimal_risk_range(self, results: Dict[str, Any]) -> Tuple[float, float]:
        """Find optimal risk parameter range based on results"""
        successful_results = {k: v for k, v in results.items() if 'error' not in v}
        
        if not successful_results:
            return (0.8, 1.5)  # Default range
        
        # Find top 20% of results
        fitness_values = [(k, v['fitness']) for k, v in successful_results.items()]
        fitness_values.sort(key=lambda x: x[1], reverse=True)
        top_20_pct = fitness_values[:max(1, len(fitness_values) // 5)]
        
        # Extract risk parameters from top performers
        top_risks = [successful_results[k]['risk_parameter'] for k, _ in top_20_pct]
        
        return (min(top_risks), max(top_risks))
    
    def _generate_sensitivity_summary(self, sensitivity_results: Dict[str, Any], 
                                    base_portfolio: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive sensitivity analysis summary"""
        summary = {
            'analysis_timestamp': datetime.now().isoformat(),
            'most_sensitive_parameter': None,
            'least_sensitive_parameter': None,
            'recommended_ranges': {},
            'stability_metrics': {}
        }
        
        # Analyze parameter sensitivity levels
        sensitivity_levels = {}
        
        for param_name, param_results in sensitivity_results.items():
            if param_name.endswith('_sensitivity') and 'sensitivity_patterns' in param_results:
                patterns = param_results['sensitivity_patterns']
                if 'fitness_correlation_with_penalty' in patterns:
                    sensitivity_levels[param_name] = abs(patterns['fitness_correlation_with_penalty'])
                elif 'fitness_correlation_with_size' in patterns:
                    sensitivity_levels[param_name] = abs(patterns['fitness_correlation_with_size'])
                elif 'fitness_correlation_with_risk' in patterns:
                    sensitivity_levels[param_name] = abs(patterns['fitness_correlation_with_risk'])
        
        if sensitivity_levels:
            summary['most_sensitive_parameter'] = max(sensitivity_levels, key=sensitivity_levels.get)
            summary['least_sensitive_parameter'] = min(sensitivity_levels, key=sensitivity_levels.get)
            summary['sensitivity_rankings'] = dict(sorted(sensitivity_levels.items(), 
                                                         key=lambda x: x[1], reverse=True))
        
        return summary