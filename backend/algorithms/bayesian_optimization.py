#!/usr/bin/env python3
"""
Bayesian Optimization Implementation for Heavy Optimizer Platform

Implements BO algorithm with configuration support.

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


class BayesianOptimization(BaseOptimizationAlgorithm):
    """Bayesian Optimization implementation with configuration support"""
    
    def __init__(self, config_path: Optional[str] = None, use_gpu: bool = True):
        """
        Initialize BO with configuration
        
        Args:
            config_path: Path to configuration file
            use_gpu: Whether to use GPU acceleration with cuDF
        """
        super().__init__(config_path, use_gpu)
        
        # Get configuration manager
        self.config_manager = get_config_manager()
        
        # Load algorithm-specific parameters
        self.acquisition_function = self.config_manager.get('ALGORITHM_PARAMETERS', 'bo_acquisition_function', 'expected_improvement')
        self.n_initial_points = self.config_manager.getint('ALGORITHM_PARAMETERS', 'bo_n_initial_points', 10)
        self.n_calls = self.config_manager.getint('ALGORITHM_PARAMETERS', 'bo_n_calls', 50)
        self.kernel = self.config_manager.get('ALGORITHM_PARAMETERS', 'bo_kernel', 'matern')
        self.random_state = self.config_manager.getint('ALGORITHM_PARAMETERS', 'bo_random_state', 42)
        self.exploration_rate = self.config_manager.getfloat('ALGORITHM_PARAMETERS', 'bo_exploration_rate', 0.2)
        
        # Initialize fitness calculator
        self.fitness_calculator = FitnessCalculator(use_gpu=self.use_gpu)
        
        self.logger.info(f"Initialized BO with acquisition_function={self.acquisition_function}, "
                        f"n_calls={self.n_calls}, GPU mode={self.use_gpu}")
    
    def optimize(self, data: Union[np.ndarray, 'cudf.DataFrame'], 
                portfolio_size: Union[int, Tuple[int, int]], 
                fitness_function: Optional[Callable] = None, 
                zone_data: Optional[Dict] = None) -> Dict:
        """
        Run Bayesian optimization (updated for cuDF support)
        
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
            actual_size = random.randint(min_size, max_size)
        else:
            actual_size = portfolio_size
        
        # Initialize with random points
        observed_portfolios = []
        observed_fitness = []
        
        # Initial random sampling
        for _ in range(self.n_initial_points):
            if zone_data and zone_data.get('enabled'):
                portfolio = self._create_zone_constrained_portfolio(
                    strategy_list, actual_size, zone_data
                )
            else:
                portfolio = list(np.random.choice(strategy_list, actual_size, replace=False))
            
            fitness = fitness_function(portfolio)
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
            if random.random() < self.exploration_rate:
                # Random exploration
                if zone_data and zone_data.get('enabled'):
                    candidate = self._create_zone_constrained_portfolio(
                        strategy_list, actual_size, zone_data
                    )
                else:
                    candidate = list(np.random.choice(strategy_list, actual_size, replace=False))
            else:
                # Exploit around best solution
                candidate = self._mutate_portfolio(best_solution, strategy_list)
            
            # Evaluate candidate
            fitness = fitness_function(candidate)
            
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
            'n_calls': self.n_calls,
            'fitness_history': fitness_history,
            'algorithm_name': 'bayesian_optimization',
            'data_type': data_type,
            'gpu_accelerated': self.use_gpu,
            'detailed_metrics': detailed_metrics,
            'parameters': {
                'acquisition_function': self.acquisition_function,
                'n_initial_points': self.n_initial_points,
                'kernel': self.kernel,
                'exploration_rate': self.exploration_rate,
                'num_strategies': num_strategies,
                'portfolio_size': actual_size
            }
        }
    
    def _mutate_portfolio(self, portfolio: List[Union[int, str]], 
                         strategy_list: List[Union[int, str]]) -> List[Union[int, str]]:
        """Mutate portfolio for local exploration"""
        mutated = portfolio.copy()
        
        # Replace one strategy
        pos = random.randint(0, len(mutated) - 1)
        available = [s for s in strategy_list if s not in mutated]
        
        if len(available) > 0:
            mutated[pos] = random.choice(available)
        
        return mutated
    
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
        
        return portfolio[:portfolio_size]
