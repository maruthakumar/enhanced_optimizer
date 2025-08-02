#!/usr/bin/env python3
"""
Base Algorithm Class for Heavy Optimizer Platform

This abstract base class provides common functionality for all optimization algorithms.
Each algorithm must inherit from this class and implement the optimize() method.
"""

import configparser
import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import numpy as np


class BaseOptimizationAlgorithm(ABC):
    """Abstract base class for all optimization algorithms"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the algorithm with configuration
        
        Args:
            config_path: Path to the .ini configuration file
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = {}
        self.algorithm_name = self.__class__.__name__.replace('Algorithm', '')
        
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
                daily_matrix: np.ndarray, 
                portfolio_size: Union[int, Tuple[int, int]], 
                fitness_function: callable,
                zone_data: Optional[Dict] = None) -> Dict:
        """
        Run the optimization algorithm
        
        Args:
            daily_matrix: Matrix of daily returns (days x strategies)
            portfolio_size: Either fixed size (int) or (min_size, max_size) tuple
            fitness_function: Function to evaluate portfolio fitness
            zone_data: Optional zone-specific data for zone-wise optimization
            
        Returns:
            Dictionary containing:
                - best_portfolio: List of strategy indices
                - best_fitness: Final fitness score
                - execution_time: Time taken to run algorithm
                - algorithm_name: Name of the algorithm
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
        
    def validate_inputs(self, 
                       daily_matrix: np.ndarray, 
                       portfolio_size: Union[int, Tuple[int, int]]) -> None:
        """
        Validate input parameters
        
        Args:
            daily_matrix: Input data matrix
            portfolio_size: Portfolio size specification
            
        Raises:
            ValueError: If inputs are invalid
        """
        if not isinstance(daily_matrix, np.ndarray):
            raise ValueError("daily_matrix must be a numpy array")
            
        if daily_matrix.ndim != 2:
            raise ValueError(f"daily_matrix must be 2D, got {daily_matrix.ndim}D")
            
        num_strategies = daily_matrix.shape[1]
        
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
            'supports_parallel': getattr(self, 'supports_parallel', False)
        }