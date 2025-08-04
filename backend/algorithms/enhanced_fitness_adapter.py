"""
Adapter to integrate enhanced financial metrics with existing algorithm framework.
"""
import logging
import numpy as np
from typing import Union, List, Dict, Callable, Optional
import sys
import os

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'lib'))

# Import existing fitness calculator
from algorithms.fitness_functions import FitnessCalculator

# Import enhanced components
from lib.financial_metrics import (
    EnhancedFitnessCalculator,
    create_fitness_calculator,
    KellyCriterion,
    EnhancedMetrics
)
from lib.risk_management import VaRCVaRCalculator
from lib.regime_optimization import RegimeHandler

# Import GPU utilities
from lib.gpu_utils import get_cudf_safe, get_cupy_safe, CUDF_AVAILABLE

cudf = get_cudf_safe()
cp = get_cupy_safe()

logger = logging.getLogger(__name__)


class EnhancedFitnessAdapter(FitnessCalculator):
    """
    Adapter that extends the existing FitnessCalculator to support enhanced metrics
    while maintaining backward compatibility.
    """
    
    def __init__(self, use_gpu: bool = True, metrics_config: Dict[str, float] = None, 
                 enhanced_config: Optional[Dict] = None):
        """
        Initialize enhanced fitness adapter.
        
        Args:
            use_gpu: Whether to use GPU acceleration when available
            metrics_config: Configuration for metric weights (legacy)
            enhanced_config: Configuration for enhanced metrics
        """
        super().__init__(use_gpu, metrics_config)
        
        # Create enhanced calculator if config provided
        if enhanced_config:
            self.enhanced_calculator = create_fitness_calculator(enhanced_config)
            self.fitness_mode = enhanced_config.get('FITNESS_CALCULATION', {}).get('mode', 'legacy')
            
            # Initialize components for enhanced calculations
            self.kelly_enabled = enhanced_config.get('KELLY_CRITERION', {}).get('enabled', False)
            self.regime_weighting = enhanced_config.get('MARKET_REGIME_CONFIG', {}).get('confidence_weighting', False)
            
            logger.info(f"Enhanced fitness adapter initialized with mode: {self.fitness_mode}")
        else:
            self.enhanced_calculator = None
            self.fitness_mode = 'legacy'
            self.kelly_enabled = False
            self.regime_weighting = False
            
    def create_fitness_function(self, 
                               data: Union[np.ndarray, 'cudf.DataFrame'],
                               data_type: str = 'auto',
                               strategy_metadata: Optional[Dict] = None) -> Callable:
        """
        Create a fitness function compatible with the given data type.
        Supports both legacy and enhanced modes.
        
        Args:
            data: Strategy data (numpy array or cuDF DataFrame)
            data_type: 'auto', 'numpy', or 'cudf'
            strategy_metadata: Optional metadata for enhanced calculations
            
        Returns:
            Fitness function that takes portfolio and returns fitness score
        """
        # Use parent class for legacy mode
        if self.fitness_mode == 'legacy' or not self.enhanced_calculator:
            return super().create_fitness_function(data, data_type)
            
        # Determine data type
        if data_type == 'auto':
            if isinstance(data, np.ndarray):
                data_type = 'numpy'
            elif CUDF_AVAILABLE and isinstance(data, cudf.DataFrame):
                data_type = 'cudf'
            else:
                raise ValueError(f"Cannot determine data type for {type(data)}")
                
        # Create enhanced fitness function
        if data_type == 'cudf' and self.use_gpu:
            return self._create_enhanced_cudf_fitness_function(data, strategy_metadata)
        else:
            return self._create_enhanced_numpy_fitness_function(data, strategy_metadata)
            
    def _create_enhanced_cudf_fitness_function(self, 
                                             df: 'cudf.DataFrame',
                                             metadata: Optional[Dict] = None) -> Callable:
        """
        Create GPU-accelerated enhanced fitness function for cuDF data.
        
        Args:
            df: cuDF DataFrame with strategy data
            metadata: Optional strategy metadata
            
        Returns:
            Enhanced fitness function
        """
        def fitness_function(portfolio: List[Union[int, str]]) -> float:
            try:
                # Convert portfolio indices to column names if needed
                if isinstance(portfolio[0], int):
                    strategy_cols = [col for col in df.columns if col not in ['Date', 'date', 'DATE']]
                    portfolio_cols = [strategy_cols[i] for i in portfolio]
                else:
                    portfolio_cols = portfolio
                
                # Calculate basic metrics using parent class
                basic_metrics = self.calculate_detailed_metrics(df, portfolio_cols)
                
                # Calculate returns for enhanced metrics
                portfolio_df = df[portfolio_cols]
                portfolio_returns = portfolio_df.sum(axis=1)
                
                # Add enhanced metrics if available
                if metadata:
                    # Add regime confidence if available
                    if 'regime_confidence' in metadata:
                        regime_confidences = [metadata['regime_confidence'].get(col, 1.0) for col in portfolio_cols]
                        avg_confidence = np.mean(regime_confidences)
                        basic_metrics['regime_confidence'] = avg_confidence
                        
                    # Add win/loss data for Kelly if available
                    if 'strategy_metrics' in metadata:
                        for col in portfolio_cols:
                            if col in metadata['strategy_metrics']:
                                col_metrics = metadata['strategy_metrics'][col]
                                basic_metrics.update({
                                    'win_rate': col_metrics.get('win_rate', 0.5),
                                    'avg_win': col_metrics.get('avg_win', 1.0),
                                    'avg_loss': col_metrics.get('avg_loss', 1.0)
                                })
                                break  # Use first strategy's metrics for now
                
                # Calculate enhanced fitness
                if self.fitness_mode == 'enhanced':
                    fitness = self.enhanced_calculator.calculate_enhanced_fitness(
                        basic_metrics, portfolio_returns
                    )
                elif self.fitness_mode == 'hybrid':
                    fitness = self.enhanced_calculator.calculate_hybrid_fitness(
                        basic_metrics, portfolio_returns
                    )
                else:
                    fitness = basic_metrics.get('fitness_score', 0.0)
                    
                return fitness
                
            except Exception as e:
                logger.error(f"Error in enhanced cuDF fitness calculation: {str(e)}")
                return 0.0
                
        return fitness_function
        
    def _create_enhanced_numpy_fitness_function(self,
                                              data: Union[np.ndarray, 'cudf.DataFrame'],
                                              metadata: Optional[Dict] = None) -> Callable:
        """
        Create CPU-based enhanced fitness function.
        
        Args:
            data: Strategy data
            metadata: Optional strategy metadata
            
        Returns:
            Enhanced fitness function
        """
        # Convert cuDF to numpy if needed
        if CUDF_AVAILABLE and isinstance(data, cudf.DataFrame):
            strategy_cols = [col for col in data.columns if col not in ['Date', 'date', 'DATE']]
            daily_matrix = data[strategy_cols].to_numpy()
            col_mapping = {i: col for i, col in enumerate(strategy_cols)}
        else:
            daily_matrix = data
            col_mapping = None
            
        def fitness_function(portfolio: List[int]) -> float:
            try:
                # Calculate basic metrics using parent class
                basic_metrics = self.calculate_detailed_metrics(daily_matrix, portfolio)
                
                # Calculate portfolio returns
                portfolio_returns = daily_matrix[:, portfolio].sum(axis=1)
                
                # Add enhanced metrics if available
                if metadata and col_mapping:
                    # Map indices to column names
                    portfolio_cols = [col_mapping[i] for i in portfolio]
                    
                    # Add regime confidence
                    if 'regime_confidence' in metadata:
                        regime_confidences = [metadata['regime_confidence'].get(col, 1.0) for col in portfolio_cols]
                        avg_confidence = np.mean(regime_confidences)
                        basic_metrics['regime_confidence'] = avg_confidence
                        
                    # Add strategy-specific metrics
                    if 'strategy_metrics' in metadata:
                        for col in portfolio_cols:
                            if col in metadata['strategy_metrics']:
                                col_metrics = metadata['strategy_metrics'][col]
                                basic_metrics.update({
                                    'win_rate': col_metrics.get('win_rate', 0.5),
                                    'avg_win': col_metrics.get('avg_win', 1.0),
                                    'avg_loss': col_metrics.get('avg_loss', 1.0)
                                })
                                break
                
                # Calculate enhanced fitness
                if self.fitness_mode == 'enhanced' and self.enhanced_calculator:
                    # Convert numpy returns to cudf Series if possible
                    if CUDF_AVAILABLE:
                        returns_series = cudf.Series(portfolio_returns)
                    else:
                        returns_series = None
                        
                    fitness = self.enhanced_calculator.calculate_enhanced_fitness(
                        basic_metrics, returns_series
                    )
                elif self.fitness_mode == 'hybrid' and self.enhanced_calculator:
                    if CUDF_AVAILABLE:
                        returns_series = cudf.Series(portfolio_returns)
                    else:
                        returns_series = None
                        
                    fitness = self.enhanced_calculator.calculate_hybrid_fitness(
                        basic_metrics, returns_series
                    )
                else:
                    fitness = basic_metrics.get('fitness_score', 0.0)
                    
                return fitness
                
            except Exception as e:
                logger.error(f"Error in enhanced numpy fitness calculation: {str(e)}")
                return 0.0
                
        return fitness_function
        
    def apply_kelly_sizing(self,
                         portfolio_df: 'cudf.DataFrame',
                         strategy_metrics: Dict[str, Dict]) -> 'cudf.DataFrame':
        """
        Apply Kelly Criterion position sizing to portfolio.
        
        Args:
            portfolio_df: DataFrame with selected strategies
            strategy_metrics: Metrics for each strategy
            
        Returns:
            DataFrame with Kelly position sizes
        """
        if not self.kelly_enabled or not self.enhanced_calculator:
            return portfolio_df
            
        # Prepare data for Kelly calculation
        kelly_data = []
        for strategy in portfolio_df.columns:
            if strategy in strategy_metrics:
                metrics = strategy_metrics[strategy]
                kelly_data.append({
                    'strategy': strategy,
                    'win_rate': metrics.get('win_rate', 0.5),
                    'avg_win': metrics.get('avg_win', 1.0),
                    'avg_loss': metrics.get('avg_loss', 1.0)
                })
                
        if kelly_data:
            kelly_df = cudf.DataFrame(kelly_data)
            kelly_df = self.enhanced_calculator.kelly.apply_kelly_sizing(kelly_df)
            
            # Add Kelly positions to portfolio
            for _, row in kelly_df.iterrows():
                strategy = row['strategy']
                if strategy in portfolio_df.columns:
                    portfolio_df[f'{strategy}_kelly_size'] = row['kelly_position_size']
                    
        return portfolio_df
        
    def apply_regime_weighting(self,
                             portfolio_df: 'cudf.DataFrame',
                             regime_data: Dict[str, float]) -> 'cudf.DataFrame':
        """
        Apply market regime confidence weighting.
        
        Args:
            portfolio_df: DataFrame with selected strategies
            regime_data: Regime confidence data
            
        Returns:
            DataFrame with regime weights
        """
        if not self.regime_weighting or not self.enhanced_calculator:
            return portfolio_df
            
        # Add regime confidence data
        for strategy in portfolio_df.columns:
            if strategy in regime_data:
                portfolio_df[f'{strategy}_regime_confidence'] = regime_data[strategy]
                
        # Apply regime handler
        portfolio_df = self.enhanced_calculator.regime_handler.apply_confidence_weighting(portfolio_df)
        
        return portfolio_df


def create_enhanced_fitness_adapter(config: Dict) -> EnhancedFitnessAdapter:
    """
    Factory function to create enhanced fitness adapter.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured enhanced fitness adapter
    """
    # Extract legacy metrics config
    metrics_config = {
        'roi_dd_ratio_weight': config.get('METRICS', {}).get('roi_dd_ratio_weight', 1.0),
        'total_roi_weight': config.get('METRICS', {}).get('total_roi_weight', 0.0),
        'max_drawdown_weight': config.get('METRICS', {}).get('max_drawdown_weight', 0.0),
        'win_rate_weight': config.get('METRICS', {}).get('win_rate_weight', 0.0),
        'profit_factor_weight': config.get('METRICS', {}).get('profit_factor_weight', 0.0)
    }
    
    # Determine if GPU should be used
    use_gpu = config.get('SYSTEM', {}).get('use_gpu', True)
    
    return EnhancedFitnessAdapter(
        use_gpu=use_gpu,
        metrics_config=metrics_config,
        enhanced_config=config
    )