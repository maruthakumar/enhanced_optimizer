"""
Performance Attribution Analysis Module

Analyzes portfolio performance attribution using actual production data from 25,544 strategies.
Breaks down performance by Stop Loss levels, Take Profit configurations, zones, and time periods.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class AttributionConfig:
    """Configuration for performance attribution analysis"""
    min_strategy_weight: float = 0.001  # Minimum weight to consider significant
    attribution_tolerance: float = 0.0001  # Tolerance for attribution sum validation
    category_regex_patterns: Dict[str, str] = None
    
    def __post_init__(self):
        if self.category_regex_patterns is None:
            self.category_regex_patterns = {
                'stop_loss': r'SL(\d+)',
                'take_profit': r'TP(\d+)', 
                'zone_range': r'SENSEX\s+(\d+)',
                'strategy_type': r'(SENSEX.*?)(?:\s|$)'
            }


class PerformanceAttributionAnalyzer:
    """
    Analyzes portfolio performance attribution for production strategy universe.
    
    Breaks down performance by:
    - Stop Loss levels (7%-88% range)
    - Take Profit configurations (32%-42%)
    - Zone-based attribution for SENSEX ranges
    - Time-based attribution across trading days
    """
    
    def __init__(self, config: Optional[AttributionConfig] = None):
        """
        Initialize performance attribution analyzer
        
        Args:
            config: Attribution configuration
        """
        self.config = config or AttributionConfig()
        self.strategy_metadata = {}
        self.attribution_cache = {}
        
    def analyze_portfolio_attribution(self, portfolio: List[int], 
                                    portfolio_weights: np.ndarray,
                                    daily_returns: np.ndarray,
                                    strategy_names: List[str],
                                    dates: Optional[pd.DatetimeIndex] = None) -> Dict[str, Any]:
        """
        Perform comprehensive performance attribution analysis
        
        Args:
            portfolio: List of strategy indices in portfolio
            portfolio_weights: Weights for each strategy in portfolio
            daily_returns: Daily returns matrix (days x strategies)
            strategy_names: Names of all strategies
            dates: Trading dates for time-based attribution
            
        Returns:
            Dictionary with comprehensive attribution analysis
        """
        logger.info(f"📊 Starting performance attribution for {len(portfolio)} strategy portfolio")
        
        # Validate inputs
        if len(portfolio) != len(portfolio_weights):
            raise ValueError("Portfolio and weights must have same length")
        
        # Extract strategy metadata for attribution
        self._extract_strategy_metadata(strategy_names)
        
        # Calculate portfolio returns
        portfolio_returns = self._calculate_portfolio_returns(
            portfolio, portfolio_weights, daily_returns
        )
        
        # Perform attribution analysis
        attribution_results = {
            'portfolio_summary': self._calculate_portfolio_summary(
                portfolio, portfolio_weights, portfolio_returns, strategy_names
            ),
            'stop_loss_attribution': self._analyze_stop_loss_attribution(
                portfolio, portfolio_weights, daily_returns, strategy_names
            ),
            'take_profit_attribution': self._analyze_take_profit_attribution(
                portfolio, portfolio_weights, daily_returns, strategy_names
            ),
            'zone_attribution': self._analyze_zone_attribution(
                portfolio, portfolio_weights, daily_returns, strategy_names
            ),
            'time_attribution': self._analyze_time_attribution(
                portfolio, portfolio_weights, daily_returns, dates
            ),
            'strategy_contribution': self._analyze_strategy_contribution(
                portfolio, portfolio_weights, daily_returns, strategy_names
            ),
            'risk_attribution': self._analyze_risk_attribution(
                portfolio, portfolio_weights, daily_returns, strategy_names
            )
        }
        
        # Validate attribution sums
        self._validate_attribution(attribution_results)
        
        logger.info("✅ Performance attribution analysis completed")
        return attribution_results
    
    def _extract_strategy_metadata(self, strategy_names: List[str]) -> None:
        """Extract metadata from strategy names using regex patterns"""
        logger.info("🔍 Extracting strategy metadata from names")
        
        for i, name in enumerate(strategy_names):
            metadata = {
                'index': i,
                'name': name,
                'stop_loss': None,
                'take_profit': None,
                'zone_range': None,
                'strategy_type': None
            }
            
            # Extract Stop Loss
            sl_match = re.search(self.config.category_regex_patterns['stop_loss'], name)
            if sl_match:
                metadata['stop_loss'] = int(sl_match.group(1))
            
            # Extract Take Profit
            tp_match = re.search(self.config.category_regex_patterns['take_profit'], name)
            if tp_match:
                metadata['take_profit'] = int(tp_match.group(1))
            
            # Extract Zone Range
            zone_match = re.search(self.config.category_regex_patterns['zone_range'], name)
            if zone_match:
                metadata['zone_range'] = int(zone_match.group(1))
            
            # Extract Strategy Type
            type_match = re.search(self.config.category_regex_patterns['strategy_type'], name)
            if type_match:
                metadata['strategy_type'] = type_match.group(1).strip()
            
            self.strategy_metadata[i] = metadata
        
        logger.info(f"📋 Extracted metadata for {len(self.strategy_metadata)} strategies")
    
    def _calculate_portfolio_returns(self, portfolio: List[int], 
                                   portfolio_weights: np.ndarray,
                                   daily_returns: np.ndarray) -> np.ndarray:
        """Calculate daily portfolio returns"""
        portfolio_data = daily_returns[:, portfolio]
        portfolio_returns = np.sum(portfolio_data * portfolio_weights, axis=1)
        return portfolio_returns
    
    def _calculate_portfolio_summary(self, portfolio: List[int],
                                   portfolio_weights: np.ndarray,
                                   portfolio_returns: np.ndarray,
                                   strategy_names: List[str]) -> Dict[str, Any]:
        """Calculate overall portfolio performance summary"""
        total_return = np.sum(portfolio_returns)
        avg_daily_return = np.mean(portfolio_returns)
        volatility = np.std(portfolio_returns)
        sharpe_ratio = avg_daily_return / volatility if volatility > 0 else 0
        
        # Calculate maximum drawdown
        cumulative = np.cumprod(1 + portfolio_returns / 100)  # Assuming returns in %
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        return {
            'total_return': total_return,
            'avg_daily_return': avg_daily_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'num_strategies': len(portfolio),
            'strategy_names': [strategy_names[i] for i in portfolio],
            'strategy_weights': portfolio_weights.tolist(),
            'trading_days': len(portfolio_returns)
        }
    
    def _analyze_stop_loss_attribution(self, portfolio: List[int],
                                     portfolio_weights: np.ndarray,
                                     daily_returns: np.ndarray,
                                     strategy_names: List[str]) -> Dict[str, Any]:
        """Analyze performance attribution by Stop Loss levels"""
        logger.info("📊 Analyzing Stop Loss attribution")
        
        sl_attribution = {}
        sl_groups = {}
        
        # Group strategies by Stop Loss level
        for i, strategy_idx in enumerate(portfolio):
            metadata = self.strategy_metadata.get(strategy_idx, {})
            sl_level = metadata.get('stop_loss')
            
            if sl_level is not None:
                if sl_level not in sl_groups:
                    sl_groups[sl_level] = {'indices': [], 'weights': [], 'names': []}
                
                sl_groups[sl_level]['indices'].append(strategy_idx)
                sl_groups[sl_level]['weights'].append(portfolio_weights[i])
                sl_groups[sl_level]['names'].append(strategy_names[strategy_idx])
        
        # Calculate attribution for each Stop Loss group
        for sl_level, group_data in sl_groups.items():
            group_indices = group_data['indices']
            group_weights = np.array(group_data['weights'])
            group_weight_sum = np.sum(group_weights)
            
            if group_weight_sum > self.config.min_strategy_weight:
                # Normalize weights within group
                normalized_weights = group_weights / group_weight_sum
                
                # Calculate group returns
                group_returns = daily_returns[:, group_indices]
                group_portfolio_returns = np.sum(group_returns * normalized_weights, axis=1)
                
                # Weight by group's total portfolio weight
                attributed_returns = group_portfolio_returns * group_weight_sum
                
                sl_attribution[f'SL_{sl_level}%'] = {
                    'total_return': np.sum(attributed_returns),
                    'avg_daily_return': np.mean(attributed_returns),
                    'volatility': np.std(attributed_returns),
                    'weight_in_portfolio': group_weight_sum,
                    'num_strategies': len(group_indices),
                    'strategy_names': group_data['names'],
                    'daily_attribution': attributed_returns.tolist()
                }
        
        return sl_attribution
    
    def _analyze_take_profit_attribution(self, portfolio: List[int],
                                       portfolio_weights: np.ndarray,
                                       daily_returns: np.ndarray,
                                       strategy_names: List[str]) -> Dict[str, Any]:
        """Analyze performance attribution by Take Profit levels"""
        logger.info("📊 Analyzing Take Profit attribution")
        
        tp_attribution = {}
        tp_groups = {}
        
        # Group strategies by Take Profit level
        for i, strategy_idx in enumerate(portfolio):
            metadata = self.strategy_metadata.get(strategy_idx, {})
            tp_level = metadata.get('take_profit')
            
            if tp_level is not None:
                if tp_level not in tp_groups:
                    tp_groups[tp_level] = {'indices': [], 'weights': [], 'names': []}
                
                tp_groups[tp_level]['indices'].append(strategy_idx)
                tp_groups[tp_level]['weights'].append(portfolio_weights[i])
                tp_groups[tp_level]['names'].append(strategy_names[strategy_idx])
        
        # Calculate attribution for each Take Profit group
        for tp_level, group_data in tp_groups.items():
            group_indices = group_data['indices']
            group_weights = np.array(group_data['weights'])
            group_weight_sum = np.sum(group_weights)
            
            if group_weight_sum > self.config.min_strategy_weight:
                normalized_weights = group_weights / group_weight_sum
                group_returns = daily_returns[:, group_indices]
                group_portfolio_returns = np.sum(group_returns * normalized_weights, axis=1)
                attributed_returns = group_portfolio_returns * group_weight_sum
                
                tp_attribution[f'TP_{tp_level}%'] = {
                    'total_return': np.sum(attributed_returns),
                    'avg_daily_return': np.mean(attributed_returns),
                    'volatility': np.std(attributed_returns),
                    'weight_in_portfolio': group_weight_sum,
                    'num_strategies': len(group_indices),
                    'strategy_names': group_data['names'],
                    'daily_attribution': attributed_returns.tolist()
                }
        
        return tp_attribution
    
    def _analyze_zone_attribution(self, portfolio: List[int],
                                portfolio_weights: np.ndarray,
                                daily_returns: np.ndarray,
                                strategy_names: List[str]) -> Dict[str, Any]:
        """Analyze performance attribution by SENSEX zones"""
        logger.info("📊 Analyzing Zone attribution")
        
        zone_attribution = {}
        zone_groups = {}
        
        # Group strategies by zone ranges
        for i, strategy_idx in enumerate(portfolio):
            metadata = self.strategy_metadata.get(strategy_idx, {})
            zone_range = metadata.get('zone_range')
            
            if zone_range is not None:
                # Categorize zones into ranges
                zone_category = self._categorize_zone_range(zone_range)
                
                if zone_category not in zone_groups:
                    zone_groups[zone_category] = {'indices': [], 'weights': [], 'names': []}
                
                zone_groups[zone_category]['indices'].append(strategy_idx)
                zone_groups[zone_category]['weights'].append(portfolio_weights[i])
                zone_groups[zone_category]['names'].append(strategy_names[strategy_idx])
        
        # Calculate attribution for each zone
        for zone_cat, group_data in zone_groups.items():
            group_indices = group_data['indices']
            group_weights = np.array(group_data['weights'])
            group_weight_sum = np.sum(group_weights)
            
            if group_weight_sum > self.config.min_strategy_weight:
                normalized_weights = group_weights / group_weight_sum
                group_returns = daily_returns[:, group_indices]
                group_portfolio_returns = np.sum(group_returns * normalized_weights, axis=1)
                attributed_returns = group_portfolio_returns * group_weight_sum
                
                zone_attribution[zone_cat] = {
                    'total_return': np.sum(attributed_returns),
                    'avg_daily_return': np.mean(attributed_returns),
                    'volatility': np.std(attributed_returns),
                    'weight_in_portfolio': group_weight_sum,
                    'num_strategies': len(group_indices),
                    'strategy_names': group_data['names'],
                    'daily_attribution': attributed_returns.tolist()
                }
        
        return zone_attribution
    
    def _categorize_zone_range(self, zone_value: int) -> str:
        """Categorize zone ranges into meaningful groups"""
        if zone_value < 1100:
            return 'Low_Range_1000-1099'
        elif zone_value < 1200:
            return 'Mid_Range_1100-1199' 
        elif zone_value < 1300:
            return 'High_Range_1200-1299'
        else:
            return 'Very_High_Range_1300+'
    
    def _analyze_time_attribution(self, portfolio: List[int],
                                portfolio_weights: np.ndarray,
                                daily_returns: np.ndarray,
                                dates: Optional[pd.DatetimeIndex]) -> Dict[str, Any]:
        """Analyze performance attribution over time periods"""
        logger.info("📊 Analyzing time-based attribution")
        
        if dates is None:
            # Create synthetic dates for 82 trading days
            dates = pd.date_range(start='2024-01-04', periods=daily_returns.shape[0], freq='B')
        
        # Calculate portfolio returns
        portfolio_data = daily_returns[:, portfolio]
        daily_portfolio_returns = np.sum(portfolio_data * portfolio_weights, axis=1)
        
        # Time-based attribution
        time_attribution = {
            'daily_returns': daily_portfolio_returns.tolist(),
            'dates': dates.strftime('%Y-%m-%d').tolist() if hasattr(dates, 'strftime') else dates,
            'monthly_attribution': self._calculate_monthly_attribution(daily_portfolio_returns, dates),
            'weekly_attribution': self._calculate_weekly_attribution(daily_portfolio_returns, dates),
            'cumulative_returns': np.cumsum(daily_portfolio_returns).tolist()
        }
        
        return time_attribution
    
    def _calculate_monthly_attribution(self, daily_returns: np.ndarray, 
                                     dates: pd.DatetimeIndex) -> Dict[str, Any]:
        """Calculate monthly performance attribution"""
        df = pd.DataFrame({'returns': daily_returns, 'date': dates})
        df['month'] = df['date'].dt.to_period('M')
        
        monthly_stats = df.groupby('month')['returns'].agg([
            'sum', 'mean', 'std', 'count'
        ]).to_dict('index')
        
        return {str(month): {
            'total_return': stats['sum'],
            'avg_daily_return': stats['mean'],
            'volatility': stats['std'],
            'trading_days': stats['count']
        } for month, stats in monthly_stats.items()}
    
    def _calculate_weekly_attribution(self, daily_returns: np.ndarray,
                                    dates: pd.DatetimeIndex) -> Dict[str, Any]:
        """Calculate weekly performance attribution"""
        df = pd.DataFrame({'returns': daily_returns, 'date': dates})
        df['week'] = df['date'].dt.to_period('W')
        
        weekly_stats = df.groupby('week')['returns'].agg([
            'sum', 'mean', 'std', 'count'
        ]).to_dict('index')
        
        return {str(week): {
            'total_return': stats['sum'],
            'avg_daily_return': stats['mean'],
            'volatility': stats['std'],
            'trading_days': stats['count']
        } for week, stats in weekly_stats.items()}
    
    def _analyze_strategy_contribution(self, portfolio: List[int],
                                     portfolio_weights: np.ndarray,
                                     daily_returns: np.ndarray,
                                     strategy_names: List[str]) -> Dict[str, Any]:
        """Analyze individual strategy contributions"""
        strategy_contributions = {}
        
        for i, strategy_idx in enumerate(portfolio):
            strategy_returns = daily_returns[:, strategy_idx]
            strategy_weight = portfolio_weights[i]
            attributed_returns = strategy_returns * strategy_weight
            
            strategy_contributions[strategy_names[strategy_idx]] = {
                'total_contribution': np.sum(attributed_returns),
                'avg_daily_contribution': np.mean(attributed_returns),
                'volatility_contribution': np.std(attributed_returns) * strategy_weight,
                'weight': strategy_weight,
                'daily_contributions': attributed_returns.tolist()
            }
        
        return strategy_contributions
    
    def _analyze_risk_attribution(self, portfolio: List[int],
                                portfolio_weights: np.ndarray,
                                daily_returns: np.ndarray,
                                strategy_names: List[str]) -> Dict[str, Any]:
        """Analyze risk contribution by strategies and categories"""
        # Calculate portfolio variance
        portfolio_data = daily_returns[:, portfolio]
        cov_matrix = np.cov(portfolio_data.T)
        portfolio_variance = np.dot(portfolio_weights, np.dot(cov_matrix, portfolio_weights))
        
        # Individual risk contributions
        risk_contributions = {}
        marginal_contributions = np.dot(cov_matrix, portfolio_weights)
        
        for i, strategy_idx in enumerate(portfolio):
            risk_contrib = portfolio_weights[i] * marginal_contributions[i] / portfolio_variance
            risk_contributions[strategy_names[strategy_idx]] = {
                'risk_contribution': risk_contrib,
                'marginal_contribution': marginal_contributions[i],
                'weight': portfolio_weights[i]
            }
        
        return {
            'individual_risk_contributions': risk_contributions,
            'portfolio_variance': portfolio_variance,
            'portfolio_volatility': np.sqrt(portfolio_variance)
        }
    
    def _validate_attribution(self, attribution_results: Dict[str, Any]) -> None:
        """Validate that attribution components sum to total portfolio performance"""
        total_return = attribution_results['portfolio_summary']['total_return']
        tolerance = self.config.attribution_tolerance
        
        # Validate Stop Loss attribution
        sl_sum = sum(attr['total_return'] for attr in attribution_results['stop_loss_attribution'].values())
        if abs(sl_sum - total_return) > tolerance:
            logger.warning(f"⚠️  Stop Loss attribution sum ({sl_sum:.6f}) differs from total return ({total_return:.6f})")
        
        # Validate strategy contribution
        strategy_sum = sum(contrib['total_contribution'] for contrib in attribution_results['strategy_contribution'].values())
        if abs(strategy_sum - total_return) > tolerance:
            logger.warning(f"⚠️  Strategy contribution sum ({strategy_sum:.6f}) differs from total return ({total_return:.6f})")
        
        logger.info("✅ Attribution validation completed")