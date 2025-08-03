#!/usr/bin/env python3
"""
Base Algorithm Class for Heavy Optimizer Platform

This abstract base class provides common functionality for all optimization algorithms.
Each algorithm must inherit from this class and implement the optimize() method.

RETROFITTED FOR PARQUET/ARROW/CUDF SUPPORT (Story 1.1R)
Now supports both legacy numpy arrays and new cuDF DataFrames for GPU acceleration.
"""

import configparser
import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import numpy as np

# Import GPU utilities for centralized cuDF handling
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'lib'))
from gpu_utils import get_cudf_safe, get_cupy_safe, ensure_gpu_compatibility, CUDF_AVAILABLE

# Get cuDF and cuPy safely
cudf = get_cudf_safe()
cp = get_cupy_safe()


class BaseOptimizationAlgorithm(ABC):
    """Abstract base class for all optimization algorithms"""
    
    def __init__(self, config_path: Optional[str] = None, use_gpu: bool = True):
        """
        Initialize the algorithm with configuration
        
        Args:
            config_path: Path to the .ini configuration file
            use_gpu: Whether to use GPU acceleration with cuDF
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = {}
        self.algorithm_name = self.__class__.__name__.replace('Algorithm', '')
        
        # GPU configuration using centralized utilities
        self.use_gpu, self.data_type = ensure_gpu_compatibility(use_gpu)
        
        # Log GPU status only once per algorithm type
        if not hasattr(self.__class__, '_gpu_status_logged'):
            if self.use_gpu:
                self.logger.info(f"🚀 {self.algorithm_name} initialized with GPU acceleration")
            else:
                self.logger.info(f"💻 {self.algorithm_name} initialized in CPU mode")
            self.__class__._gpu_status_logged = True
        
        # Load configuration if provided
        if config_path:
            self._load_config(config_path)
            
    def _load_config(self, config_path: str) -> None:
        """Load algorithm-specific configuration from .ini file"""
        config = configparser.ConfigParser()
        config.read(config_path)
        
        # Try to load algorithm-specific section
        section_name = self.algorithm_name.upper()
        if config.has_section(section_name):
            self.config = dict(config.items(section_name))
            self.logger.info(f"Loaded configuration for {section_name}")
        else:
            self.logger.warning(f"No configuration section found for {section_name}")
            
        # Load general algorithm settings if available
        if config.has_section('ALGORITHMS'):
            for key, value in config.items('ALGORITHMS'):
                if self.algorithm_name.lower() in key.lower() and key not in self.config:
                    self.config[key] = value
                    
    def _get_config_value(self, key: str, default: Union[str, int, float], 
                         value_type: type = str) -> Union[str, int, float]:
        """
        Get configuration value with type conversion
        
        Args:
            key: Configuration key
            default: Default value if key not found
            value_type: Type to convert the value to
            
        Returns:
            Configuration value converted to specified type
        """
        value = self.config.get(key, default)
        
        try:
            if value_type == int:
                return int(value)
            elif value_type == float:
                return float(value)
            elif value_type == bool:
                return str(value).lower() in ('true', '1', 'yes', 'on')
            else:
                return str(value)
        except (ValueError, TypeError):
            self.logger.warning(f"Failed to convert {key}={value} to {value_type}, using default={default}")
            return default
            
    @abstractmethod
    def optimize(self, 
                data: Union[np.ndarray, 'cudf.DataFrame'], 
                portfolio_size: Union[int, Tuple[int, int]], 
                fitness_function: callable,
                zone_data: Optional[Dict] = None) -> Dict:
        """
        Run the optimization algorithm
        
        Args:
            data: Strategy data - either numpy array (legacy) or cuDF DataFrame (new GPU pipeline)
                  If numpy: Matrix of daily returns (days x strategies)
                  If cuDF: DataFrame with strategy columns
            portfolio_size: Either fixed size (int) or (min_size, max_size) tuple
            fitness_function: Function to evaluate portfolio fitness
            zone_data: Optional zone-specific data for zone-wise optimization
            
        Returns:
            Dictionary containing:
                - best_portfolio: List of strategy indices/names
                - best_fitness: Final fitness score
                - execution_time: Time taken to run algorithm
                - algorithm_name: Name of the algorithm
                - data_type: 'numpy' or 'cudf' to indicate input type
                - gpu_accelerated: Boolean indicating if GPU was used
                - additional algorithm-specific metrics
        """
        pass
        
    def _determine_portfolio_size(self, size_spec: Union[int, Tuple[int, int]]) -> int:
        """
        Determine actual portfolio size from specification
        
        Args:
            size_spec: Either fixed size or (min_size, max_size) tuple
            
        Returns:
            Actual portfolio size to use
        """
        if isinstance(size_spec, int):
            return size_spec
        elif isinstance(size_spec, tuple) and len(size_spec) == 2:
            min_size, max_size = size_spec
            # Can be overridden by specific algorithms for dynamic sizing
            return min_size
        else:
            raise ValueError(f"Invalid portfolio size specification: {size_spec}")
            
    def _calculate_execution_time(self, start_time: float) -> float:
        """Calculate execution time in seconds"""
        return round(time.time() - start_time, 3)
        
    def _apply_zone_constraints(self, 
                              portfolio: List[int], 
                              zone_data: Dict) -> List[int]:
        """
        Apply zone-specific constraints to portfolio
        
        Args:
            portfolio: Current portfolio
            zone_data: Zone-specific constraints
            
        Returns:
            Modified portfolio adhering to zone constraints
        """
        if not zone_data:
            return portfolio
            
        # Extract zone constraints
        zone_strategies = zone_data.get('allowed_strategies', [])
        min_zone_strategies = zone_data.get('min_strategies_per_zone', 0)
        
        if zone_strategies:
            # Filter portfolio to only include allowed strategies
            filtered_portfolio = [s for s in portfolio if s in zone_strategies]
            
            # Ensure minimum strategies from zone
            if len(filtered_portfolio) < min_zone_strategies:
                # Add more strategies from the zone
                available = [s for s in zone_strategies if s not in filtered_portfolio]
                needed = min_zone_strategies - len(filtered_portfolio)
                if available and needed > 0:
                    additional = np.random.choice(available, 
                                                min(needed, len(available)), 
                                                replace=False)
                    filtered_portfolio.extend(additional)
                    
            return filtered_portfolio
        
        return portfolio
        
    def _detect_data_type(self, data: Union[np.ndarray, 'cudf.DataFrame']) -> str:
        """
        Detect the type of input data
        
        Args:
            data: Input data
            
        Returns:
            'numpy' or 'cudf'
        """
        if isinstance(data, np.ndarray):
            return 'numpy'
        elif CUDF_AVAILABLE and isinstance(data, cudf.DataFrame):
            return 'cudf'
        else:
            raise ValueError(f"Unsupported data type: {type(data)}. Expected numpy.ndarray or cudf.DataFrame")
    
    def _get_strategy_list(self, data: Union[np.ndarray, 'cudf.DataFrame']) -> List[Union[int, str]]:
        """
        Get list of strategy identifiers from data
        
        Args:
            data: Input data
            
        Returns:
            List of strategy indices (for numpy) or column names (for cuDF)
        """
        data_type = self._detect_data_type(data)
        
        if data_type == 'numpy':
            return list(range(data.shape[1]))
        else:  # cudf
            # Exclude non-strategy columns like Date
            strategy_cols = [col for col in data.columns if col not in ['Date', 'date', 'DATE']]
            return strategy_cols
    
    def _get_num_strategies(self, data: Union[np.ndarray, 'cudf.DataFrame']) -> int:
        """
        Get number of strategies from data
        
        Args:
            data: Input data
            
        Returns:
            Number of strategies
        """
        strategy_list = self._get_strategy_list(data)
        return len(strategy_list)
    
    def _convert_to_numpy_if_needed(self, data: Union[np.ndarray, 'cudf.DataFrame']) -> np.ndarray:
        """
        Convert cuDF DataFrame to numpy array for legacy compatibility
        
        Args:
            data: Input data
            
        Returns:
            Numpy array
        """
        data_type = self._detect_data_type(data)
        
        if data_type == 'numpy':
            return data
        else:  # cudf
            strategy_cols = self._get_strategy_list(data)
            return data[strategy_cols].to_numpy()

    def validate_inputs(self, 
                       data: Union[np.ndarray, 'cudf.DataFrame'], 
                       portfolio_size: Union[int, Tuple[int, int]]) -> None:
        """
        Validate input parameters (updated for cuDF support)
        
        Args:
            data: Input data (numpy array or cuDF DataFrame)
            portfolio_size: Portfolio size specification
            
        Raises:
            ValueError: If inputs are invalid
        """
        data_type = self._detect_data_type(data)
        
        if data_type == 'numpy':
            if data.ndim != 2:
                raise ValueError(f"numpy array must be 2D, got {data.ndim}D")
        elif data_type == 'cudf':
            if len(data) == 0:
                raise ValueError("cuDF DataFrame is empty")
            
        num_strategies = self._get_num_strategies(data)
        
        if isinstance(portfolio_size, int):
            if portfolio_size <= 0:
                raise ValueError(f"Portfolio size must be positive, got {portfolio_size}")
            if portfolio_size > num_strategies:
                raise ValueError(f"Portfolio size {portfolio_size} exceeds number of strategies {num_strategies}")
        elif isinstance(portfolio_size, tuple):
            if len(portfolio_size) != 2:
                raise ValueError(f"Portfolio size tuple must have 2 elements, got {len(portfolio_size)}")
            min_size, max_size = portfolio_size
            if min_size <= 0 or max_size <= 0:
                raise ValueError(f"Portfolio sizes must be positive, got min={min_size}, max={max_size}")
            if min_size > max_size:
                raise ValueError(f"min_size {min_size} must be <= max_size {max_size}")
            if max_size > num_strategies:
                raise ValueError(f"max_size {max_size} exceeds number of strategies {num_strategies}")
        else:
            raise ValueError(f"Portfolio size must be int or tuple, got {type(portfolio_size)}")
            
    def get_algorithm_info(self) -> Dict:
        """Get algorithm information and configuration"""
        return {
            'name': self.algorithm_name,
            'class': self.__class__.__name__,
            'configuration': self.config,
            'supports_zones': True,
            'supports_variable_portfolio': True,
            'supports_parallel': getattr(self, 'supports_parallel', False),
            'gpu_accelerated': self.use_gpu,
            'cudf_available': CUDF_AVAILABLE,
            'supports_cudf': True,
            'supports_legacy_numpy': True
        }