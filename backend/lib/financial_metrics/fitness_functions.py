"""
Enhanced fitness functions integrating Kelly, Sharpe, VaR, and regime optimization.
"""
import cudf
import logging
from typing import Dict, Union, Optional, Callable
import numpy as np

from .kelly_criterion import KellyCriterion
from .enhanced_metrics import EnhancedMetrics
from ..risk_management.var_cvar_calculator import VaRCVaRCalculator
from ..regime_optimization.regime_handler import RegimeHandler

logger = logging.getLogger(__name__)


class EnhancedFitnessCalculator:
    """
    Configurable fitness calculator supporting legacy and enhanced modes.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize fitness calculator with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.mode = config.get('FITNESS_CALCULATION', {}).get('mode', 'legacy')
        
        # Initialize components
        self.kelly = KellyCriterion(
            max_position_size=float(config.get('KELLY_CRITERION', {}).get('max_position_size', 0.25)),
            min_position_size=float(config.get('KELLY_CRITERION', {}).get('min_position_size', 0.01))
        )
        
        self.metrics = EnhancedMetrics(
            risk_free_rate=float(config.get('RISK_METRICS', {}).get('risk_free_rate', 0.02))
        )
        
        self.risk_calc = VaRCVaRCalculator(
            confidence_levels=[0.95, 0.99]
        )
        
        self.regime_handler = RegimeHandler(
            min_confidence_threshold=float(config.get('MARKET_REGIME_CONFIG', {}).get('min_confidence_threshold', 70)) / 100.0
        )
        
    def calculate_legacy_fitness(self,
                               strategy_data: Dict[str, float]) -> float:
        """
        Calculate fitness using legacy mode (backward compatible).
        
        Legacy formula: roi/drawdown - penalty
        
        Args:
            strategy_data: Dictionary with strategy metrics
            
        Returns:
            Legacy fitness score
        """
        roi = strategy_data.get('total_roi', 0.0)
        drawdown = abs(strategy_data.get('max_drawdown', 1.0))
        
        # Avoid division by zero
        if drawdown < 0.0001:
            drawdown = 0.0001
            
        # Base fitness
        fitness = roi / drawdown
        
        # Apply penalties
        penalty = 0.0
        
        # Penalty for low win rate
        win_rate = strategy_data.get('win_rate', 0.5)
        if win_rate < 0.4:
            penalty += (0.4 - win_rate) * 10
            
        # Penalty for high drawdown
        if drawdown > 500000:  # Based on typical range from story
            penalty += (drawdown - 500000) / 100000
            
        return fitness - penalty
        
    def calculate_enhanced_fitness(self,
                                 strategy_data: Dict[str, float],
                                 returns: Optional[cudf.Series] = None) -> float:
        """
        Calculate fitness using enhanced mode with professional metrics.
        
        Enhanced formula: kelly_weight * sharpe * regime_factor - var_penalty
        
        Args:
            strategy_data: Dictionary with strategy metrics
            returns: Optional returns series for advanced calculations
            
        Returns:
            Enhanced fitness score
        """
        # Kelly weight
        win_rate = strategy_data.get('win_rate', 0.5)
        avg_win = strategy_data.get('avg_win', 1.0)
        avg_loss = abs(strategy_data.get('avg_loss', 1.0))
        
        kelly_fraction = self.kelly.calculate_kelly_fraction(win_rate, avg_win, avg_loss)
        
        # Sharpe ratio (use approximation if returns not available)
        if returns is not None:
            sharpe = self.metrics.calculate_sharpe_ratio(returns)
        else:
            # Approximate from ROI and win rate
            roi = strategy_data.get('total_roi', 0.0)
            volatility = strategy_data.get('volatility', 1.0)
            if volatility > 0:
                sharpe = roi / volatility * np.sqrt(252)  # Annualized
            else:
                sharpe = 0.0
                
        # Regime confidence factor
        regime_confidence = strategy_data.get('regime_confidence', 1.0)
        if regime_confidence > 1:  # Convert from percentage
            regime_confidence = regime_confidence / 100.0
            
        # VaR penalty (if available)
        var_penalty = 0.0
        if 'var_95' in strategy_data:
            var_95 = abs(strategy_data['var_95'])
            var_limit = float(self.config.get('RISK_METRICS', {}).get('var_limit', 0.025))
            if var_95 > var_limit:
                var_penalty = (var_95 - var_limit) * 100
                
        # Enhanced fitness calculation
        fitness = kelly_fraction * max(sharpe, 0.1) * regime_confidence - var_penalty
        
        # Add Sortino bonus for downside protection
        if 'sortino_ratio' in strategy_data:
            sortino = strategy_data['sortino_ratio']
            if sortino > sharpe:  # Reward better downside protection
                fitness *= (1 + (sortino - sharpe) * 0.1)
                
        return fitness
        
    def calculate_hybrid_fitness(self,
                               strategy_data: Dict[str, float],
                               returns: Optional[cudf.Series] = None) -> float:
        """
        Calculate fitness using hybrid mode combining legacy and enhanced.
        
        Args:
            strategy_data: Dictionary with strategy metrics
            returns: Optional returns series
            
        Returns:
            Hybrid fitness score
        """
        legacy_score = self.calculate_legacy_fitness(strategy_data)
        enhanced_score = self.calculate_enhanced_fitness(strategy_data, returns)
        
        # Weight combination (configurable)
        legacy_weight = float(self.config.get('FITNESS_CALCULATION', {}).get('legacy_weight', 0.3))
        enhanced_weight = 1.0 - legacy_weight
        
        return legacy_weight * legacy_score + enhanced_weight * enhanced_score
        
    def calculate_fitness(self,
                        strategy_data: Dict[str, float],
                        returns: Optional[cudf.Series] = None) -> float:
        """
        Calculate fitness score based on configured mode.
        
        Args:
            strategy_data: Dictionary with strategy metrics
            returns: Optional returns series for advanced calculations
            
        Returns:
            Fitness score
        """
        if self.mode == 'legacy':
            return self.calculate_legacy_fitness(strategy_data)
        elif self.mode == 'enhanced':
            return self.calculate_enhanced_fitness(strategy_data, returns)
        elif self.mode == 'hybrid':
            return self.calculate_hybrid_fitness(strategy_data, returns)
        else:
            logger.warning(f"Unknown fitness mode: {self.mode}. Using legacy.")
            return self.calculate_legacy_fitness(strategy_data)
            
    def calculate_portfolio_fitness(self,
                                  portfolio_df: cudf.DataFrame,
                                  portfolio_returns: Optional[cudf.DataFrame] = None) -> float:
        """
        Calculate fitness for entire portfolio considering correlations.
        
        Args:
            portfolio_df: DataFrame with portfolio strategies
            portfolio_returns: Optional returns DataFrame
            
        Returns:
            Portfolio fitness score
        """
        # Calculate weighted metrics
        if 'regime_weight' in portfolio_df.columns:
            weights = portfolio_df['regime_weight']
        elif 'kelly_position_size' in portfolio_df.columns:
            weights = portfolio_df['kelly_position_size']
            weights = weights / weights.sum()  # Normalize
        else:
            weights = cudf.Series([1.0 / len(portfolio_df)] * len(portfolio_df))
            
        # Weighted average of individual fitness scores
        if 'fitness_score' in portfolio_df.columns:
            weighted_fitness = (portfolio_df['fitness_score'] * weights).sum()
        else:
            # Calculate fitness for each strategy
            fitness_scores = []
            for idx, row in portfolio_df.iterrows():
                strategy_data = row.to_dict()
                fitness_scores.append(self.calculate_fitness(strategy_data))
            portfolio_df['fitness_score'] = fitness_scores
            weighted_fitness = (portfolio_df['fitness_score'] * weights).sum()
            
        # Portfolio-level adjustments
        if portfolio_returns is not None:
            # Calculate portfolio Sharpe
            portfolio_return_series = (portfolio_returns * weights).sum(axis=1)
            portfolio_sharpe = self.metrics.calculate_sharpe_ratio(portfolio_return_series)
            
            # Adjust fitness based on portfolio efficiency
            if portfolio_sharpe > 2.0:  # Excellent portfolio
                weighted_fitness *= 1.2
            elif portfolio_sharpe < 0.5:  # Poor portfolio
                weighted_fitness *= 0.8
                
        return float(weighted_fitness)
        
    def rank_strategies(self,
                      strategies_df: cudf.DataFrame,
                      returns_df: Optional[cudf.DataFrame] = None) -> cudf.DataFrame:
        """
        Rank strategies by fitness score.
        
        Args:
            strategies_df: DataFrame with strategy data
            returns_df: Optional DataFrame with returns
            
        Returns:
            DataFrame sorted by fitness score
        """
        # Calculate fitness for each strategy
        fitness_scores = []
        
        for idx in range(len(strategies_df)):
            strategy_data = strategies_df.iloc[idx].to_dict()
            
            # Get returns if available
            if returns_df is not None and idx < len(returns_df.columns):
                returns = returns_df.iloc[:, idx]
            else:
                returns = None
                
            fitness = self.calculate_fitness(strategy_data, returns)
            fitness_scores.append(fitness)
            
        strategies_df['fitness_score'] = fitness_scores
        
        # Sort by fitness (descending)
        ranked_df = strategies_df.sort_values('fitness_score', ascending=False)
        
        return ranked_df


def create_fitness_calculator(config: Dict) -> EnhancedFitnessCalculator:
    """
    Factory function to create fitness calculator.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured fitness calculator
    """
    return EnhancedFitnessCalculator(config)


# Standalone fitness functions for backward compatibility
def calculate_fitness_legacy(roi: float, 
                           drawdown: float,
                           win_rate: float = 0.5,
                           penalty_factor: float = 1.0) -> float:
    """
    Legacy fitness calculation for backward compatibility.
    
    Args:
        roi: Total return on investment
        drawdown: Maximum drawdown (positive value)
        win_rate: Win rate (0-1)
        penalty_factor: Additional penalty factor
        
    Returns:
        Legacy fitness score
    """
    if drawdown < 0.0001:
        drawdown = 0.0001
        
    fitness = roi / drawdown
    
    # Apply penalties
    penalty = 0.0
    if win_rate < 0.4:
        penalty += (0.4 - win_rate) * 10
        
    return fitness - (penalty * penalty_factor)


def calculate_fitness_enhanced(kelly_fraction: float,
                             sharpe_ratio: float,
                             regime_confidence: float,
                             var_penalty: float = 0.0) -> float:
    """
    Enhanced fitness calculation for standalone use.
    
    Args:
        kelly_fraction: Kelly criterion position size
        sharpe_ratio: Sharpe ratio
        regime_confidence: Market regime confidence (0-1)
        var_penalty: Value at Risk penalty
        
    Returns:
        Enhanced fitness score
    """
    return kelly_fraction * max(sharpe_ratio, 0.1) * regime_confidence - var_penalty