"""
Kelly Criterion position sizing implementation for optimal capital allocation.
"""
import numpy as np
import logging
from typing import Dict, Tuple, Union, Optional

# Try to import cuDF
try:
    import cudf
    CUDF_AVAILABLE = True
except (ImportError, RuntimeError):
    CUDF_AVAILABLE = False
    # Create a mock cudf module for type hints
    class cudf:
        DataFrame = dict
        Series = list

logger = logging.getLogger(__name__)


class KellyCriterion:
    """
    Kelly Criterion calculator for optimal position sizing based on win/loss probabilities.
    
    The Kelly formula determines the optimal fraction of capital to allocate
    to maximize long-term growth while avoiding ruin.
    """
    
    def __init__(self, max_position_size: float = 0.25, min_position_size: float = 0.01):
        """
        Initialize Kelly Criterion calculator.
        
        Args:
            max_position_size: Maximum allowed position size (default 25% cap for safety)
            min_position_size: Minimum position size (default 1%)
        """
        self.max_position_size = max_position_size
        self.min_position_size = min_position_size
        
    def calculate_kelly_fraction(self, 
                               win_rate: float, 
                               avg_win: float, 
                               avg_loss: float) -> float:
        """
        Calculate Kelly fraction for a single strategy.
        
        Kelly formula: f = (p*b - q) / b
        where:
            f = fraction of capital to wager
            p = probability of winning
            b = ratio of win amount to loss amount
            q = probability of losing (1 - p)
            
        Args:
            win_rate: Probability of winning (0-1)
            avg_win: Average winning amount
            avg_loss: Average losing amount (positive value)
            
        Returns:
            Optimal fraction of capital to allocate (capped between min and max)
        """
        if avg_loss <= 0 or avg_win <= 0:
            logger.warning(f"Invalid win/loss values: avg_win={avg_win}, avg_loss={avg_loss}")
            return self.min_position_size
            
        # Calculate b (odds ratio)
        b = avg_win / avg_loss
        
        # Calculate Kelly fraction
        p = win_rate
        q = 1 - win_rate
        
        kelly_fraction = (p * b - q) / b
        
        # Apply safety constraints
        if kelly_fraction <= 0:
            return self.min_position_size
        elif kelly_fraction > self.max_position_size:
            return self.max_position_size
        else:
            return max(kelly_fraction, self.min_position_size)
            
    def calculate_kelly_fractions_batch(self, 
                                      strategies_df: cudf.DataFrame) -> cudf.Series:
        """
        Calculate Kelly fractions for multiple strategies using GPU acceleration.
        
        Args:
            strategies_df: DataFrame with columns:
                - 'win_rate': Win rate for each strategy
                - 'avg_win': Average winning amount
                - 'avg_loss': Average losing amount
                
        Returns:
            Series of Kelly fractions for each strategy
        """
        # Ensure required columns exist
        required_cols = ['win_rate', 'avg_win', 'avg_loss']
        if not all(col in strategies_df.columns for col in required_cols):
            raise ValueError(f"DataFrame must contain columns: {required_cols}")
            
        # GPU-accelerated calculation
        b = strategies_df['avg_win'] / strategies_df['avg_loss'].clip(lower=0.001)
        p = strategies_df['win_rate']
        q = 1 - p
        
        kelly_fractions = (p * b - q) / b
        
        # Apply constraints
        kelly_fractions = kelly_fractions.clip(lower=self.min_position_size, 
                                             upper=self.max_position_size)
        
        # Handle invalid cases (NaN, inf)
        kelly_fractions = kelly_fractions.fillna(self.min_position_size)
        
        return kelly_fractions
        
    def apply_kelly_sizing(self, 
                         portfolio_df: cudf.DataFrame,
                         total_capital: float = 1.0,
                         fallback_mode: str = 'equal_weight') -> cudf.DataFrame:
        """
        Apply Kelly sizing to a portfolio of strategies.
        
        Args:
            portfolio_df: DataFrame with strategy data and Kelly inputs
            total_capital: Total capital to allocate (default 1.0 for percentage)
            fallback_mode: How to handle when Kelly suggests zero allocation
                          ('equal_weight', 'min_position', 'exclude')
                          
        Returns:
            DataFrame with added 'kelly_position_size' column
        """
        # Calculate Kelly fractions
        kelly_fractions = self.calculate_kelly_fractions_batch(portfolio_df)
        
        # Handle cases where total Kelly allocation exceeds capital
        total_kelly = kelly_fractions.sum()
        
        if total_kelly > 1.0:
            # Scale down proportionally
            kelly_fractions = kelly_fractions / total_kelly
            logger.info(f"Scaled Kelly fractions from {total_kelly:.3f} to 1.0")
            
        # Apply fallback for zero/minimal allocations
        if fallback_mode == 'equal_weight':
            num_strategies = len(portfolio_df)
            min_total = self.min_position_size * num_strategies
            
            if total_kelly < min_total:
                kelly_fractions = cudf.Series([1.0 / num_strategies] * num_strategies,
                                            index=portfolio_df.index)
                logger.info("Applied equal weight fallback due to low Kelly allocations")
                
        # Calculate actual position sizes
        portfolio_df['kelly_fraction'] = kelly_fractions
        portfolio_df['kelly_position_size'] = kelly_fractions * total_capital
        
        return portfolio_df
        
    def calculate_kelly_leverage(self, 
                               strategies_df: cudf.DataFrame,
                               max_leverage: float = 2.0) -> float:
        """
        Calculate optimal leverage based on Kelly Criterion.
        
        Args:
            strategies_df: DataFrame with strategy performance metrics
            max_leverage: Maximum allowed leverage
            
        Returns:
            Optimal leverage factor
        """
        kelly_fractions = self.calculate_kelly_fractions_batch(strategies_df)
        
        # Sum of Kelly fractions indicates optimal leverage
        optimal_leverage = kelly_fractions.sum()
        
        # Cap at maximum allowed leverage
        return min(optimal_leverage, max_leverage)
        
    def kelly_risk_adjustment(self, 
                            kelly_fraction: float,
                            confidence_level: float,
                            volatility: float) -> float:
        """
        Adjust Kelly fraction based on confidence and volatility.
        
        Args:
            kelly_fraction: Base Kelly fraction
            confidence_level: Confidence in the edge (0-1)
            volatility: Strategy volatility
            
        Returns:
            Risk-adjusted Kelly fraction
        """
        # Reduce position size based on uncertainty
        confidence_adj = kelly_fraction * confidence_level
        
        # Further reduce based on volatility (high vol = smaller position)
        vol_adj = confidence_adj * (1 - min(volatility, 0.5))
        
        return max(vol_adj, self.min_position_size)


def create_kelly_config(config_dict: Dict) -> Dict:
    """
    Create Kelly Criterion configuration from config dictionary.
    
    Args:
        config_dict: Configuration dictionary
        
    Returns:
        Kelly-specific configuration
    """
    kelly_config = {
        'enabled': config_dict.get('KELLY_CRITERION', {}).get('enabled', True),
        'max_position_size': float(config_dict.get('KELLY_CRITERION', {}).get('max_position_size', 0.25)),
        'min_position_size': float(config_dict.get('KELLY_CRITERION', {}).get('min_position_size', 0.01)),
        'fallback_mode': config_dict.get('KELLY_CRITERION', {}).get('fallback_mode', 'equal_weight'),
        'use_leverage': config_dict.get('KELLY_CRITERION', {}).get('use_leverage', False),
        'max_leverage': float(config_dict.get('KELLY_CRITERION', {}).get('max_leverage', 2.0))
    }
    
    return kelly_config