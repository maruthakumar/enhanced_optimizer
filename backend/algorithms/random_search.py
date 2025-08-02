#!/usr/bin/env python3
"""
Random Search Implementation for Heavy Optimizer Platform

Implements RS algorithm with configuration support.
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
from config.config_manager import get_config_manager


class RandomSearch(BaseOptimizationAlgorithm):
    """Random Search implementation with configuration support"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize RS with configuration
        
        Args:
            config_path: Path to configuration file
        """
        super().__init__(config_path)
        
        # Get configuration manager
        self.config_manager = get_config_manager()
        
        # Load algorithm-specific parameters
        self.iterations = self.config_manager.getint('ALGORITHM_PARAMETERS', 'rs_iterations', 1000)
        self.random_seed = self.config_manager.getint('ALGORITHM_PARAMETERS', 'rs_random_seed', 42)
        self.sampling_method = self.config_manager.get('ALGORITHM_PARAMETERS', 'rs_sampling_method', 'uniform')
        
        # Set random seed for reproducibility
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        
        self.logger.info(f"Initialized RS with iterations={self.iterations}, "
                        f"random_seed={self.random_seed}")
    
    def optimize(self, daily_matrix: np.ndarray, portfolio_size: Union[int, Tuple[int, int]], 
                fitness_function: Callable, zone_data: Optional[Dict] = None) -> Dict:
        """
        Run random search optimization
        
        Args:
            daily_matrix: Daily returns matrix (days x strategies)
            portfolio_size: Target portfolio size or (min, max) range
            fitness_function: Function to evaluate portfolio fitness
            zone_data: Optional zone constraint data
            
        Returns:
            Dictionary with optimization results
        """
        start_time = time.time()
        
        # Handle portfolio size
        if isinstance(portfolio_size, tuple):
            min_size, max_size = portfolio_size
        else:
            min_size = max_size = portfolio_size
        
        num_strategies = daily_matrix.shape[1]
        
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
                    num_strategies, actual_size, zone_data
                )
            else:
                portfolio = np.random.choice(num_strategies, actual_size, replace=False)
            
            # Evaluate
            fitness = fitness_function(daily_matrix, portfolio)
            
            # Update best
            if fitness > best_fitness:
                best_fitness = fitness
                best_solution = portfolio.copy()
            
            fitness_history.append(best_fitness)
            
            # Log progress
            if iteration % 100 == 0:
                self.logger.debug(f"Iteration {iteration}: Best fitness = {best_fitness:.6f}")
        
        execution_time = time.time() - start_time
        
        return {
            'best_fitness': float(best_fitness),
            'best_portfolio': best_solution.tolist() if best_solution is not None else [],
            'execution_time': execution_time,
            'iterations': self.iterations,
            'fitness_history': fitness_history,
            'algorithm': 'random_search',
            'parameters': {
                'random_seed': self.random_seed,
                'sampling_method': self.sampling_method
            }
        }
    
    def _create_zone_constrained_portfolio(self, num_strategies: int, portfolio_size: int,
                                         zone_data: Dict) -> np.ndarray:
        """Create portfolio respecting zone constraints"""
        zone_count = zone_data.get('zone_count', 4)
        zone_weights = zone_data.get('zone_weights', [0.25] * zone_count)
        min_per_zone = zone_data.get('min_per_zone', 1)
        
        strategies_per_zone = num_strategies // zone_count
        portfolio = []
        
        for zone in range(zone_count):
            zone_size = max(min_per_zone, int(portfolio_size * zone_weights[zone]))
            zone_start = zone * strategies_per_zone
            zone_end = min((zone + 1) * strategies_per_zone, num_strategies)
            
            zone_strategies = np.arange(zone_start, zone_end)
            selected = np.random.choice(zone_strategies, 
                                      min(zone_size, len(zone_strategies)), 
                                      replace=False)
            portfolio.extend(selected)
        
        # Ensure exact size
        portfolio = np.array(portfolio)
        if len(portfolio) > portfolio_size:
            portfolio = np.random.choice(portfolio, portfolio_size, replace=False)
        elif len(portfolio) < portfolio_size:
            remaining = portfolio_size - len(portfolio)
            available = np.setdiff1d(np.arange(num_strategies), portfolio)
            if len(available) > 0:
                additional = np.random.choice(available, 
                                            min(remaining, len(available)), 
                                            replace=False)
                portfolio = np.concatenate([portfolio, additional])
        
        return portfolio
