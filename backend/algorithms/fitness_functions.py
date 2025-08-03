#!/usr/bin/env python3
"""
Fitness Functions for Heavy Optimizer Platform
Supports both legacy numpy arrays and GPU-accelerated cuDF DataFrames

CREATED FOR STORY 1.1R - Retrofitting algorithms for Parquet/Arrow/cuDF
"""

import logging
import numpy as np
from typing import Union, List, Dict, Callable

# Try to import cuDF for GPU support
try:
    import cudf
    import cupy as cp
    CUDF_AVAILABLE = True
except (ImportError, RuntimeError) as e:
    CUDF_AVAILABLE = False
    logging.warning(f"cuDF/cuPy not available ({str(e)}), GPU acceleration will be disabled")

# Import GPU fitness calculation functions
GPU_CALCULATOR_AVAILABLE = False
calculate_fitness_cudf = None
calculate_roi_cudf = None
calculate_drawdown_cudf = None
calculate_win_rate_cudf = None
calculate_profit_factor_cudf = None

try:
    from lib.cudf_engine.gpu_calculator import (
        calculate_fitness_cudf,
        calculate_roi_cudf,
        calculate_drawdown_cudf,
        calculate_win_rate_cudf,
        calculate_profit_factor_cudf
    )
    GPU_CALCULATOR_AVAILABLE = True
except (ImportError, RuntimeError) as e:
    logging.warning(f"GPU calculator not available: {str(e)}")

logger = logging.getLogger(__name__)


class FitnessCalculator:
    """
    Unified fitness calculator supporting both legacy and GPU-accelerated calculations
    """
    
    def __init__(self, use_gpu: bool = True, metrics_config: Dict[str, float] = None):
        """
        Initialize fitness calculator
        
        Args:
            use_gpu: Whether to use GPU acceleration when available
            metrics_config: Configuration for metric weights
        """
        self.use_gpu = use_gpu and CUDF_AVAILABLE and GPU_CALCULATOR_AVAILABLE
        self.metrics_config = metrics_config or self._get_default_metrics_config()
        
        logger.info(f"Initialized FitnessCalculator with GPU mode: {self.use_gpu}")
    
    def _get_default_metrics_config(self) -> Dict[str, float]:
        """Get default metrics configuration matching legacy system"""
        return {
            'roi_dd_ratio_weight': 1.0,  # Primary metric - ROI/Drawdown ratio
            'total_roi_weight': 0.0,
            'max_drawdown_weight': 0.0,
            'win_rate_weight': 0.0,
            'profit_factor_weight': 0.0
        }
    
    def create_fitness_function(self, 
                               data: Union[np.ndarray, 'cudf.DataFrame'],
                               data_type: str = 'auto') -> Callable:
        """
        Create a fitness function compatible with the given data type
        
        Args:
            data: Strategy data (numpy array or cuDF DataFrame)
            data_type: 'auto', 'numpy', or 'cudf'
            
        Returns:
            Fitness function that takes portfolio and returns fitness score
        """
        if data_type == 'auto':
            if isinstance(data, np.ndarray):
                data_type = 'numpy'
            elif CUDF_AVAILABLE and isinstance(data, cudf.DataFrame):
                data_type = 'cudf'
            else:
                raise ValueError(f"Cannot determine data type for {type(data)}")
        
        if data_type == 'cudf' and self.use_gpu:
            return self._create_cudf_fitness_function(data)
        else:
            return self._create_numpy_fitness_function(data)
    
    def _create_cudf_fitness_function(self, df: 'cudf.DataFrame') -> Callable:
        """
        Create GPU-accelerated fitness function for cuDF data
        
        Args:
            df: cuDF DataFrame with strategy data
            
        Returns:
            Fitness function
        """
        def fitness_function(portfolio: List[Union[int, str]]) -> float:
            try:
                # Convert portfolio indices to column names if needed
                if isinstance(portfolio[0], int):
                    # Get strategy columns (exclude Date)
                    strategy_cols = [col for col in df.columns if col not in ['Date', 'date', 'DATE']]
                    portfolio_cols = [strategy_cols[i] for i in portfolio]
                else:
                    portfolio_cols = portfolio
                
                # Use GPU-accelerated fitness calculation
                metrics = calculate_fitness_cudf(df, portfolio_cols, self.metrics_config)
                
                # Return primary fitness score (ROI/Drawdown ratio by default)
                return metrics.get('fitness_score', metrics.get('roi_dd_ratio', 0.0))
                
            except Exception as e:
                logger.error(f"Error in cuDF fitness calculation: {str(e)}")
                return 0.0
        
        return fitness_function
    
    def _create_numpy_fitness_function(self, 
                                     data: Union[np.ndarray, 'cudf.DataFrame']) -> Callable:
        """
        Create CPU-based fitness function for numpy data (legacy compatibility)
        
        Args:
            data: Strategy data
            
        Returns:
            Fitness function
        """
        # Convert cuDF to numpy if needed
        if CUDF_AVAILABLE and isinstance(data, cudf.DataFrame):
            strategy_cols = [col for col in data.columns if col not in ['Date', 'date', 'DATE']]
            daily_matrix = data[strategy_cols].to_numpy()
        else:
            daily_matrix = data
        
        def fitness_function(portfolio: List[int]) -> float:
            try:
                # Calculate portfolio returns
                portfolio_returns = daily_matrix[:, portfolio].sum(axis=1)
                
                # ROI calculation
                roi = portfolio_returns.sum()
                
                # Drawdown calculation
                cumulative_returns = np.cumsum(portfolio_returns)
                running_max = np.maximum.accumulate(cumulative_returns)
                drawdown = cumulative_returns - running_max
                max_drawdown = abs(drawdown.min()) if len(drawdown) > 0 else 0.01
                
                # Additional metrics for completeness
                winning_days = (portfolio_returns > 0).sum()
                total_days = len(portfolio_returns)
                win_rate = winning_days / total_days if total_days > 0 else 0
                
                gains = portfolio_returns[portfolio_returns > 0].sum()
                losses = abs(portfolio_returns[portfolio_returns < 0].sum())
                profit_factor = gains / losses if losses > 0 else gains
                
                # Legacy fitness formula: ROI/Drawdown ratio
                if max_drawdown > 0:
                    fitness = roi / max_drawdown
                else:
                    fitness = roi * 100 if roi > 0 else 0.0
                
                return fitness
                
            except Exception as e:
                logger.error(f"Error in numpy fitness calculation: {str(e)}")
                return 0.0
        
        return fitness_function
    
    def calculate_detailed_metrics(self, 
                                 data: Union[np.ndarray, 'cudf.DataFrame'],
                                 portfolio: List[Union[int, str]]) -> Dict[str, float]:
        """
        Calculate detailed metrics for a portfolio
        
        Args:
            data: Strategy data
            portfolio: Portfolio composition
            
        Returns:
            Dictionary of detailed metrics
        """
        data_type = 'cudf' if CUDF_AVAILABLE and isinstance(data, cudf.DataFrame) else 'numpy'
        
        if data_type == 'cudf' and self.use_gpu:
            # Convert portfolio indices to column names if needed
            if isinstance(portfolio[0], int):
                strategy_cols = [col for col in data.columns if col not in ['Date', 'date', 'DATE']]
                portfolio_cols = [strategy_cols[i] for i in portfolio]
            else:
                portfolio_cols = portfolio
            
            return calculate_fitness_cudf(data, portfolio_cols, self.metrics_config)
        else:
            # CPU calculation
            if CUDF_AVAILABLE and isinstance(data, cudf.DataFrame):
                strategy_cols = [col for col in data.columns if col not in ['Date', 'date', 'DATE']]
                daily_matrix = data[strategy_cols].to_numpy()
                portfolio_indices = portfolio if isinstance(portfolio[0], int) else [strategy_cols.index(col) for col in portfolio]
            else:
                daily_matrix = data
                portfolio_indices = portfolio
            
            portfolio_returns = daily_matrix[:, portfolio_indices].sum(axis=1)
            
            # Calculate metrics
            roi = portfolio_returns.sum()
            
            cumulative_returns = np.cumsum(portfolio_returns)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdown = cumulative_returns - running_max
            max_drawdown = abs(drawdown.min()) if len(drawdown) > 0 else 0.01
            
            winning_days = (portfolio_returns > 0).sum()
            total_days = len(portfolio_returns)
            win_rate = winning_days / total_days if total_days > 0 else 0
            
            gains = portfolio_returns[portfolio_returns > 0].sum()
            losses = abs(portfolio_returns[portfolio_returns < 0].sum())
            profit_factor = gains / losses if losses > 0 else gains
            
            roi_dd_ratio = roi / max_drawdown if max_drawdown > 0 else (roi * 100 if roi > 0 else 0.0)
            
            return {
                'total_roi': float(roi),
                'max_drawdown': float(max_drawdown),
                'win_rate': float(win_rate),
                'profit_factor': float(profit_factor),
                'roi_dd_ratio': float(roi_dd_ratio),
                'fitness_score': float(roi_dd_ratio)
            }


def create_legacy_fitness_function(daily_matrix: np.ndarray) -> Callable:
    """
    Create legacy fitness function for backward compatibility
    
    Args:
        daily_matrix: Numpy array of daily returns
        
    Returns:
        Legacy fitness function
    """
    calculator = FitnessCalculator(use_gpu=False)
    return calculator._create_numpy_fitness_function(daily_matrix)


def create_gpu_fitness_function(df: 'cudf.DataFrame', 
                               metrics_config: Dict[str, float] = None) -> Callable:
    """
    Create GPU-accelerated fitness function
    
    Args:
        df: cuDF DataFrame with strategy data
        metrics_config: Metrics configuration
        
    Returns:
        GPU fitness function
    """
    calculator = FitnessCalculator(use_gpu=True, metrics_config=metrics_config)
    return calculator._create_cudf_fitness_function(df)