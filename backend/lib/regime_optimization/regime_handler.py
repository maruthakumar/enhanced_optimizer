"""
Market regime confidence-based optimization and position sizing.
"""
import logging
from typing import Dict, Tuple, Optional, Union
import numpy as np

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
    cp = np

logger = logging.getLogger(__name__)


class RegimeHandler:
    """
    Handle market regime-based strategy selection and position sizing.
    
    Uses regime confidence scores to filter strategies and adjust position sizes
    based on market conditions.
    """
    
    def __init__(self, 
                 min_confidence_threshold: float = 0.7,
                 regime_column: str = 'market_regime',
                 confidence_column: str = 'Regime_Confidence_%',
                 transition_threshold_column: str = 'Market_regime_transition_threshold'):
        """
        Initialize regime handler.
        
        Args:
            min_confidence_threshold: Minimum confidence required (0-1 scale)
            regime_column: Column name for market regime
            confidence_column: Column name for regime confidence percentage
            transition_threshold_column: Column for transition detection
        """
        self.min_confidence_threshold = min_confidence_threshold
        self.regime_column = regime_column
        self.confidence_column = confidence_column
        self.transition_threshold_column = transition_threshold_column
        
    def filter_by_confidence(self, 
                           strategies_df: cudf.DataFrame,
                           threshold_override: Optional[float] = None) -> cudf.DataFrame:
        """
        Filter strategies based on regime confidence threshold.
        
        Args:
            strategies_df: DataFrame with strategy data and regime confidence
            threshold_override: Optional override for confidence threshold
            
        Returns:
            Filtered DataFrame with only high-confidence strategies
        """
        threshold = threshold_override if threshold_override is not None else self.min_confidence_threshold
        
        # Convert percentage to decimal if needed
        if self.confidence_column in strategies_df.columns:
            confidence_values = strategies_df[self.confidence_column]
            
            # Check if values are in percentage form (>1)
            if (confidence_values > 1).any():
                confidence_decimal = confidence_values / 100.0
            else:
                confidence_decimal = confidence_values
                
            # Filter strategies
            mask = confidence_decimal >= threshold
            filtered_df = strategies_df[mask].copy()
            
            logger.info(f"Filtered {len(strategies_df)} strategies to {len(filtered_df)} "
                       f"with confidence >= {threshold:.1%}")
            
            return filtered_df
        else:
            logger.warning(f"Confidence column '{self.confidence_column}' not found. "
                          "Returning all strategies.")
            return strategies_df
            
    def apply_confidence_weighting(self, 
                                 strategies_df: cudf.DataFrame,
                                 base_weights: Optional[cudf.Series] = None) -> cudf.DataFrame:
        """
        Apply confidence-based position sizing to strategies.
        
        Position size is scaled by regime confidence (0-100%).
        
        Args:
            strategies_df: DataFrame with strategy data
            base_weights: Optional base weights to scale
            
        Returns:
            DataFrame with 'regime_weight' column added
        """
        if self.confidence_column not in strategies_df.columns:
            logger.warning(f"Confidence column '{self.confidence_column}' not found. "
                          "Using equal weights.")
            strategies_df['regime_weight'] = 1.0 / len(strategies_df)
            return strategies_df
            
        # Get confidence values and convert to decimal
        confidence_values = strategies_df[self.confidence_column]
        if (confidence_values > 1).any():
            confidence_decimal = confidence_values / 100.0
        else:
            confidence_decimal = confidence_values
            
        # Apply confidence weighting
        if base_weights is not None:
            # Scale existing weights by confidence
            regime_weights = base_weights * confidence_decimal
        else:
            # Use confidence as weight directly
            regime_weights = confidence_decimal
            
        # Normalize weights to sum to 1
        weight_sum = regime_weights.sum()
        if weight_sum > 0:
            strategies_df['regime_weight'] = regime_weights / weight_sum
        else:
            strategies_df['regime_weight'] = 1.0 / len(strategies_df)
            
        # Add raw confidence factor for reference
        strategies_df['confidence_factor'] = confidence_decimal
        
        return strategies_df
        
    def detect_regime_transition(self, 
                               market_data: cudf.DataFrame) -> bool:
        """
        Detect if market is in regime transition period.
        
        Args:
            market_data: DataFrame with market regime data
            
        Returns:
            True if in transition period
        """
        if self.transition_threshold_column not in market_data.columns:
            return False
            
        # Check if transition threshold is exceeded
        transition_values = market_data[self.transition_threshold_column]
        
        # Assume transition if threshold > 0.5 or some configured value
        in_transition = (transition_values > 0.5).any()
        
        if in_transition:
            logger.info("Market regime transition detected")
            
        return bool(in_transition)
        
    def get_regime_composition(self, 
                             strategies_df: cudf.DataFrame) -> Dict[str, int]:
        """
        Get count of strategies by market regime.
        
        Args:
            strategies_df: DataFrame with regime information
            
        Returns:
            Dictionary mapping regime to strategy count
        """
        if self.regime_column not in strategies_df.columns:
            return {}
            
        regime_counts = strategies_df[self.regime_column].value_counts()
        
        return regime_counts.to_dict()
        
    def create_regime_specific_portfolios(self, 
                                        strategies_df: cudf.DataFrame,
                                        portfolio_size: int) -> Dict[str, cudf.DataFrame]:
        """
        Create separate portfolios for each market regime.
        
        Args:
            strategies_df: DataFrame with all strategies
            portfolio_size: Target size for each regime portfolio
            
        Returns:
            Dictionary mapping regime to portfolio DataFrame
        """
        if self.regime_column not in strategies_df.columns:
            return {'all': strategies_df.head(portfolio_size)}
            
        regime_portfolios = {}
        
        for regime in strategies_df[self.regime_column].unique():
            regime_strategies = strategies_df[strategies_df[self.regime_column] == regime]
            
            # Apply confidence filtering
            filtered_strategies = self.filter_by_confidence(regime_strategies)
            
            # Select top strategies up to portfolio size
            if len(filtered_strategies) > 0:
                portfolio = filtered_strategies.nlargest(
                    min(portfolio_size, len(filtered_strategies)),
                    'fitness_score'
                )
                regime_portfolios[regime] = portfolio
                
        return regime_portfolios
        
    def adjust_position_during_transition(self,
                                        weights: cudf.Series,
                                        reduction_factor: float = 0.5) -> cudf.Series:
        """
        Reduce position sizes during regime transition periods.
        
        Args:
            weights: Current position weights
            reduction_factor: Factor to reduce positions (0.5 = 50% reduction)
            
        Returns:
            Adjusted weights
        """
        adjusted_weights = weights * reduction_factor
        
        # Keep some cash during transition
        cash_weight = 1.0 - adjusted_weights.sum()
        
        logger.info(f"Reduced positions by {(1-reduction_factor)*100:.0f}% during transition. "
                   f"Cash weight: {cash_weight:.1%}")
        
        return adjusted_weights
        
    def calculate_regime_risk_limits(self,
                                   regime: str,
                                   base_var_limit: float = 0.025) -> Dict[str, float]:
        """
        Calculate regime-specific risk limits.
        
        Different regimes may have different risk tolerances.
        
        Args:
            regime: Current market regime
            base_var_limit: Base VaR limit
            
        Returns:
            Dictionary with regime-specific risk limits
        """
        # Define regime risk multipliers
        regime_risk_multipliers = {
            'bull': 1.2,      # Higher risk tolerance in bull markets
            'bear': 0.8,      # Lower risk tolerance in bear markets
            'neutral': 1.0,   # Normal risk tolerance
            'volatile': 0.6   # Much lower risk in volatile regimes
        }
        
        multiplier = regime_risk_multipliers.get(regime.lower(), 1.0)
        
        return {
            'var_limit': base_var_limit * multiplier,
            'max_position_size': 0.25 * multiplier,
            'max_leverage': 2.0 * multiplier if multiplier > 0.8 else 1.0
        }
        
    def generate_regime_report(self, 
                             strategies_df: cudf.DataFrame,
                             selected_portfolio: cudf.DataFrame) -> Dict:
        """
        Generate detailed regime analysis report.
        
        Args:
            strategies_df: All available strategies
            selected_portfolio: Selected portfolio
            
        Returns:
            Dictionary with regime analysis
        """
        report = {
            'total_strategies': len(strategies_df),
            'portfolio_size': len(selected_portfolio)
        }
        
        # Regime composition
        if self.regime_column in strategies_df.columns:
            all_regimes = self.get_regime_composition(strategies_df)
            portfolio_regimes = self.get_regime_composition(selected_portfolio)
            
            report['all_strategies_by_regime'] = all_regimes
            report['portfolio_by_regime'] = portfolio_regimes
            
        # Confidence statistics
        if self.confidence_column in strategies_df.columns:
            all_confidence = strategies_df[self.confidence_column]
            portfolio_confidence = selected_portfolio[self.confidence_column]
            
            # Convert to percentage if needed
            if (all_confidence > 1).any():
                all_confidence = all_confidence / 100.0
                portfolio_confidence = portfolio_confidence / 100.0
                
            report['confidence_stats'] = {
                'all_strategies': {
                    'mean': float(all_confidence.mean()),
                    'min': float(all_confidence.min()),
                    'max': float(all_confidence.max()),
                    'above_threshold': float((all_confidence >= self.min_confidence_threshold).sum() / len(all_confidence))
                },
                'portfolio': {
                    'mean': float(portfolio_confidence.mean()),
                    'min': float(portfolio_confidence.min()),
                    'max': float(portfolio_confidence.max())
                }
            }
            
        # Weight distribution
        if 'regime_weight' in selected_portfolio.columns:
            weights = selected_portfolio['regime_weight']
            report['weight_distribution'] = {
                'min': float(weights.min()),
                'max': float(weights.max()),
                'concentration': float(weights.nlargest(5).sum())  # Top 5 concentration
            }
            
        return report


def create_regime_config(config_dict: Dict) -> Dict:
    """
    Create regime optimization configuration from config dictionary.
    
    Args:
        config_dict: Configuration dictionary
        
    Returns:
        Regime-specific configuration
    """
    regime_config = {
        'regime_column': config_dict.get('MARKET_REGIME_CONFIG', {}).get('regime_column', 'market_regime'),
        'confidence_column': config_dict.get('MARKET_REGIME_CONFIG', {}).get('confidence_column', 'Regime_Confidence_%'),
        'min_confidence_threshold': float(config_dict.get('MARKET_REGIME_CONFIG', {}).get('min_confidence_threshold', 70)) / 100.0,
        'confidence_weighting': config_dict.get('MARKET_REGIME_CONFIG', {}).get('confidence_weighting', 'true').lower() == 'true',
        'regime_specific_portfolios': config_dict.get('MARKET_REGIME_CONFIG', {}).get('regime_specific_portfolios', 'true').lower() == 'true',
        'transition_reduction_factor': float(config_dict.get('MARKET_REGIME_CONFIG', {}).get('transition_reduction_factor', 0.5))
    }
    
    return regime_config