#!/usr/bin/env python3
"""
Bayesian Optimization Implementation for Heavy Optimizer Platform

Implements BO algorithm with configuration support.
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


class BayesianOptimization(BaseOptimizationAlgorithm):
    """Bayesian Optimization implementation with configuration support"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize BO with configuration
        
        Args:
            config_path: Path to configuration file
        """
        super().__init__(config_path)
        
        # Get configuration manager
        self.config_manager = get_config_manager()
        
        # Load algorithm-specific parameters
        self.acquisition_function = self.config_manager.get('ALGORITHM_PARAMETERS', 'bo_acquisition_function', 'expected_improvement')
        self.n_initial_points = self.config_manager.getint('ALGORITHM_PARAMETERS', 'bo_n_initial_points', 10)
        self.n_calls = self.config_manager.getint('ALGORITHM_PARAMETERS', 'bo_n_calls', 50)
        self.kernel = self.config_manager.get('ALGORITHM_PARAMETERS', 'bo_kernel', 'matern')
        self.random_state = self.config_manager.getint('ALGORITHM_PARAMETERS', 'bo_random_state', 42)
        
        self.logger.info(f"Initialized BO with acquisition_function={self.acquisition_function}, "
                        f"n_calls={self.n_calls}")
    
    def optimize(self, daily_matrix: np.ndarray, portfolio_size: Union[int, Tuple[int, int]], 
                fitness_function: Callable, zone_data: Optional[Dict] = None) -> Dict:
        """
        Run Bayesian optimization (simplified version)
        
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
            actual_size = random.randint(min_size, max_size)
        else:
            actual_size = portfolio_size
        
        num_strategies = daily_matrix.shape[1]
        
        # Initialize with random points
        observed_portfolios = []
        observed_fitness = []
        
        # Initial random sampling
        for _ in range(self.n_initial_points):
            if zone_data and zone_data.get('enabled'):
                portfolio = self._create_zone_constrained_portfolio(
                    num_strategies, actual_size, zone_data
                )
            else:
                portfolio = np.random.choice(num_strategies, actual_size, replace=False)
            
            fitness = fitness_function(daily_matrix, portfolio)
            observed_portfolios.append(portfolio)
            observed_fitness.append(fitness)
        
        # Track best
        best_idx = np.argmax(observed_fitness)
        best_solution = observed_portfolios[best_idx].copy()
        best_fitness = observed_fitness[best_idx]
        fitness_history = [best_fitness]
        
        # Bayesian optimization loop (simplified)
        for iteration in range(self.n_initial_points, self.n_calls):
            # Simple acquisition: exploit best with some exploration
            if random.random() < 0.2:  # 20% exploration
                # Random exploration
                if zone_data and zone_data.get('enabled'):
                    candidate = self._create_zone_constrained_portfolio(
                        num_strategies, actual_size, zone_data
                    )
                else:
                    candidate = np.random.choice(num_strategies, actual_size, replace=False)
            else:
                # Exploit around best solution
                candidate = self._mutate_portfolio(best_solution, num_strategies)
            
            # Evaluate candidate
            fitness = fitness_function(daily_matrix, candidate)
            
            # Update observations
            observed_portfolios.append(candidate)
            observed_fitness.append(fitness)
            
            # Update best
            if fitness > best_fitness:
                best_fitness = fitness
                best_solution = candidate.copy()
            
            fitness_history.append(best_fitness)
            
            # Log progress
            if iteration % 10 == 0:
                self.logger.debug(f"Iteration {iteration}: Best fitness = {best_fitness:.6f}")
        
        execution_time = time.time() - start_time
        
        return {
            'best_fitness': float(best_fitness),
            'best_portfolio': best_solution.tolist() if best_solution is not None else [],
            'execution_time': execution_time,
            'n_calls': self.n_calls,
            'fitness_history': fitness_history,
            'algorithm': 'bayesian_optimization',
            'parameters': {
                'acquisition_function': self.acquisition_function,
                'n_initial_points': self.n_initial_points,
                'kernel': self.kernel
            }
        }
    
    def _mutate_portfolio(self, portfolio: np.ndarray, num_strategies: int) -> np.ndarray:
        """Mutate portfolio for local exploration"""
        mutated = portfolio.copy()
        
        # Replace one strategy
        pos = random.randint(0, len(mutated) - 1)
        available = np.setdiff1d(np.arange(num_strategies), mutated)
        
        if len(available) > 0:
            mutated[pos] = np.random.choice(available)
        
        return mutated
    
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
        
        return np.array(portfolio[:portfolio_size])
