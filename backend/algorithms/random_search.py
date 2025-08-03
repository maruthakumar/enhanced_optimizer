#!/usr/bin/env python3
"""
Random Search Implementation for Heavy Optimizer Platform

Implements RS algorithm with configuration support.

RETROFITTED FOR PARQUET/ARROW/CUDF SUPPORT (Story 1.1R)
Now supports both legacy numpy arrays and GPU-accelerated cuDF DataFrames.
"""

import numpy as np
import random
from typing import Dict, List, Tuple, Optional, Union, Callable
import time
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithms.base_algorithm import BaseOptimizationAlgorithm
from algorithms.fitness_functions import FitnessCalculator
from config.config_manager import get_config_manager

# Try to import cuDF for GPU support
try:
    import cudf
    CUDF_AVAILABLE = True
except (ImportError, RuntimeError):
    CUDF_AVAILABLE = False


class RandomSearch(BaseOptimizationAlgorithm):
    """Random Search implementation with configuration support"""
    
    def __init__(self, config_path: Optional[str] = None, use_gpu: bool = True):
        """
        Initialize RS with configuration
        
        Args:
            config_path: Path to configuration file
            use_gpu: Whether to use GPU acceleration with cuDF
        """
        super().__init__(config_path, use_gpu)
        
        # Get configuration manager
        self.config_manager = get_config_manager()
        
        # Load algorithm-specific parameters
        self.iterations = self.config_manager.getint('ALGORITHM_PARAMETERS', 'rs_iterations', 1000)
        self.random_seed = self.config_manager.getint('ALGORITHM_PARAMETERS', 'rs_random_seed', 42)
        self.sampling_method = self.config_manager.get('ALGORITHM_PARAMETERS', 'rs_sampling_method', 'uniform')
        
        # Set random seed for reproducibility
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        
        # Initialize fitness calculator
        self.fitness_calculator = FitnessCalculator(use_gpu=self.use_gpu)
        
        self.logger.info(f"Initialized RS with iterations={self.iterations}, "
                        f"random_seed={self.random_seed}, GPU mode={self.use_gpu}")
    
    def optimize(self, data: Union[np.ndarray, 'cudf.DataFrame'], 
                portfolio_size: Union[int, Tuple[int, int]], 
                fitness_function: Optional[Callable] = None, 
                zone_data: Optional[Dict] = None) -> Dict:
        """
        Run random search optimization (updated for cuDF support)
        
        Args:
            data: Strategy data - numpy array (legacy) or cuDF DataFrame (new GPU pipeline)
            portfolio_size: Target portfolio size or (min, max) range
            fitness_function: Optional external fitness function (legacy compatibility)
            zone_data: Optional zone constraint data
            
        Returns:
            Dictionary with optimization results
        """
        start_time = time.time()
        
        # Validate inputs using base class method
        self.validate_inputs(data, portfolio_size)
        
        # Detect data type and get strategy information
        data_type = self._detect_data_type(data)
        strategy_list = self._get_strategy_list(data)
        num_strategies = len(strategy_list)
        
        # Create fitness function if not provided
        if fitness_function is None:
            fitness_function = self.fitness_calculator.create_fitness_function(data, data_type)
        
        # Handle portfolio size
        if isinstance(portfolio_size, tuple):
            min_size, max_size = portfolio_size
        else:
            min_size = max_size = portfolio_size
        
        # Track best
        best_solution = None
        best_fitness = -float('inf')
        fitness_history = []
        
        # Random search loop
        for iteration in range(self.iterations):
            # Random portfolio size in range
            if min_size != max_size:
                actual_size = random.randint(min_size, max_size)
            else:
                actual_size = min_size
            
            # Generate random portfolio
            if zone_data and zone_data.get('enabled'):
                portfolio = self._create_zone_constrained_portfolio(
                    strategy_list, actual_size, zone_data
                )
            else:
                portfolio = list(np.random.choice(strategy_list, actual_size, replace=False))
            
            # Evaluate
            fitness = fitness_function(portfolio)
            
            # Update best
            if fitness > best_fitness:
                best_fitness = fitness
                best_solution = portfolio.copy()
            
            fitness_history.append(best_fitness)
            
            # Log progress
            if iteration % 100 == 0:
                self.logger.debug(f"Iteration {iteration}: Best fitness = {best_fitness:.6f}")
        
        execution_time = time.time() - start_time
        
        # Calculate detailed metrics for the best solution
        detailed_metrics = {}
        if best_solution is not None:
            try:
                detailed_metrics = self.fitness_calculator.calculate_detailed_metrics(data, best_solution)
            except Exception as e:
                self.logger.warning(f"Could not calculate detailed metrics: {e}")
        
        return {
            'best_fitness': float(best_fitness),
            'best_portfolio': best_solution if best_solution is not None else [],
            'execution_time': execution_time,
            'iterations': self.iterations,
            'fitness_history': fitness_history,
            'algorithm_name': 'random_search',
            'data_type': data_type,
            'gpu_accelerated': self.use_gpu,
            'detailed_metrics': detailed_metrics,
            'parameters': {
                'random_seed': self.random_seed,
                'sampling_method': self.sampling_method,
                'num_strategies': num_strategies,
                'portfolio_size_range': (min_size, max_size)
            }
        }
    
    def _create_zone_constrained_portfolio(self, strategy_list: List[Union[int, str]], 
                                         portfolio_size: int, zone_data: Dict) -> List[Union[int, str]]:
        """Create portfolio respecting zone constraints"""
        zone_count = zone_data.get('zone_count', 4)
        zone_weights = zone_data.get('zone_weights', [0.25] * zone_count)
        min_per_zone = zone_data.get('min_per_zone', 1)
        
        num_strategies = len(strategy_list)
        strategies_per_zone = num_strategies // zone_count
        portfolio = []
        
        for zone in range(zone_count):
            zone_size = max(min_per_zone, int(portfolio_size * zone_weights[zone]))
            zone_start = zone * strategies_per_zone
            zone_end = min((zone + 1) * strategies_per_zone, num_strategies)
            
            zone_strategies = strategy_list[zone_start:zone_end]
            selected = list(np.random.choice(zone_strategies, 
                                           min(zone_size, len(zone_strategies)), 
                                           replace=False))
            portfolio.extend(selected)
        
        # Ensure exact size
        if len(portfolio) > portfolio_size:
            portfolio = list(np.random.choice(portfolio, portfolio_size, replace=False))
        elif len(portfolio) < portfolio_size:
            remaining = portfolio_size - len(portfolio)
            available = [s for s in strategy_list if s not in portfolio]
            if len(available) > 0:
                additional = list(np.random.choice(available, 
                                                 min(remaining, len(available)), 
                                                 replace=False))
                portfolio.extend(additional)
        
        return portfolio
