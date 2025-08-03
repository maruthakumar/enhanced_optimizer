#!/usr/bin/env python3
"""
Hill Climbing Implementation for Heavy Optimizer Platform

Implements HC algorithm with configuration support.

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


class HillClimbing(BaseOptimizationAlgorithm):
    """Hill Climbing implementation with configuration support"""
    
    def __init__(self, config_path: Optional[str] = None, use_gpu: bool = True):
        """
        Initialize HC with configuration
        
        Args:
            config_path: Path to configuration file
            use_gpu: Whether to use GPU acceleration with cuDF
        """
        super().__init__(config_path, use_gpu)
        
        # Get configuration manager
        self.config_manager = get_config_manager()
        
        # Load algorithm-specific parameters
        self.max_iterations = self.config_manager.getint('ALGORITHM_PARAMETERS', 'hc_max_iterations', 100)
        self.neighborhood_size = self.config_manager.getint('ALGORITHM_PARAMETERS', 'hc_neighborhood_size', 10)
        self.restart_threshold = self.config_manager.getint('ALGORITHM_PARAMETERS', 'hc_restart_threshold', 10)
        self.step_size = self.config_manager.getint('ALGORITHM_PARAMETERS', 'hc_step_size', 1)
        
        # Initialize fitness calculator
        self.fitness_calculator = FitnessCalculator(use_gpu=self.use_gpu)
        
        self.logger.info(f"Initialized HC with max_iterations={self.max_iterations}, "
                        f"neighborhood_size={self.neighborhood_size}, GPU mode={self.use_gpu}")
    
    def optimize(self, data: Union[np.ndarray, 'cudf.DataFrame'], 
                portfolio_size: Union[int, Tuple[int, int]], 
                fitness_function: Optional[Callable] = None, 
                zone_data: Optional[Dict] = None) -> Dict:
        """
        Run hill climbing optimization (updated for cuDF support)
        
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
        
        # Initialize solution
        if zone_data and zone_data.get('enabled'):
            current_solution = self._create_zone_constrained_portfolio(
                strategy_list, actual_size, zone_data
            )
        else:
            current_solution = list(np.random.choice(strategy_list, actual_size, replace=False))
        
        current_fitness = fitness_function(current_solution)
        
        # Track best
        best_solution = current_solution.copy()
        best_fitness = current_fitness
        fitness_history = [best_fitness]
        
        # Hill climbing loop
        no_improvement_count = 0
        
        for iteration in range(self.max_iterations):
            # Generate neighborhood
            neighbors = self._generate_neighbors(
                current_solution, strategy_list, zone_data
            )
            
            # Evaluate neighbors
            improved = False
            for neighbor in neighbors:
                neighbor_fitness = fitness_function(neighbor)
                
                if neighbor_fitness > current_fitness:
                    current_solution = neighbor
                    current_fitness = neighbor_fitness
                    improved = True
                    
                    if current_fitness > best_fitness:
                        best_fitness = current_fitness
                        best_solution = current_solution.copy()
                    
                    break  # Take first improvement
            
            fitness_history.append(best_fitness)
            
            if not improved:
                no_improvement_count += 1
            else:
                no_improvement_count = 0
            
            # Random restart if stuck
            if no_improvement_count >= self.restart_threshold:
                if zone_data and zone_data.get('enabled'):
                    current_solution = self._create_zone_constrained_portfolio(
                        strategy_list, actual_size, zone_data
                    )
                else:
                    current_solution = list(np.random.choice(strategy_list, actual_size, replace=False))
                
                current_fitness = fitness_function(current_solution)
                no_improvement_count = 0
                
                self.logger.debug(f"Random restart at iteration {iteration}")
            
            # Log progress
            if iteration % 20 == 0:
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
            'iterations': iteration + 1,
            'fitness_history': fitness_history,
            'algorithm_name': 'hill_climbing',
            'data_type': data_type,
            'gpu_accelerated': self.use_gpu,
            'detailed_metrics': detailed_metrics,
            'parameters': {
                'neighborhood_size': self.neighborhood_size,
                'restart_threshold': self.restart_threshold,
                'step_size': self.step_size,
                'num_strategies': num_strategies,
                'portfolio_size': actual_size
            }
        }
    
    def _generate_neighbors(self, solution: List[Union[int, str]], 
                           strategy_list: List[Union[int, str]],
                           zone_data: Optional[Dict] = None) -> List[List[Union[int, str]]]:
        """Generate neighborhood of solutions"""
        neighbors = []
        
        for _ in range(self.neighborhood_size):
            neighbor = solution.copy()
            
            # Apply step_size mutations
            for _ in range(self.step_size):
                # Random mutation type
                if random.random() < 0.5 and len(neighbor) > 1:
                    # Swap
                    i, j = random.sample(range(len(neighbor)), 2)
                    neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
                else:
                    # Replace
                    pos = random.randint(0, len(neighbor) - 1)
                    available = [s for s in strategy_list if s not in neighbor]
                    if len(available) > 0:
                        neighbor[pos] = random.choice(available)
            
            neighbors.append(neighbor)
        
        return neighbors
    
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
