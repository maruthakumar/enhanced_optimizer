#!/usr/bin/env python3
"""
Ant Colony Optimization Implementation for Heavy Optimizer Platform

Implements ACO algorithm with configuration support.

RETROFITTED FOR PARQUET/ARROW/CUDF SUPPORT (Story 1.1R)
Now supports both legacy numpy arrays and GPU-accelerated cuDF DataFrames.
Fixed negative probability bug in pheromone updates.
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


class AntColonyOptimization(BaseOptimizationAlgorithm):
    """Ant Colony Optimization implementation with configuration support"""
    
    def __init__(self, config_path: Optional[str] = None, use_gpu: bool = True):
        """
        Initialize ACO with configuration
        
        Args:
            config_path: Path to configuration file
            use_gpu: Whether to use GPU acceleration with cuDF
        """
        super().__init__(config_path, use_gpu)
        
        # Get configuration manager
        self.config_manager = get_config_manager()
        
        # Load algorithm-specific parameters
        self.colony_size = self.config_manager.getint('ALGORITHM_PARAMETERS', 'aco_colony_size', 20)
        self.alpha = self.config_manager.getfloat('ALGORITHM_PARAMETERS', 'aco_alpha', 1.0)
        self.beta = self.config_manager.getfloat('ALGORITHM_PARAMETERS', 'aco_beta', 2.0)
        self.evaporation_rate = self.config_manager.getfloat('ALGORITHM_PARAMETERS', 'aco_evaporation_rate', 0.5)
        self.pheromone_deposit = self.config_manager.getfloat('ALGORITHM_PARAMETERS', 'aco_pheromone_deposit', 1.0)
        self.iterations = self.config_manager.getint('ALGORITHM_PARAMETERS', 'aco_iterations', 50)
        self.min_pheromone = self.config_manager.getfloat('ALGORITHM_PARAMETERS', 'aco_min_pheromone', 0.01)
        self.max_pheromone = self.config_manager.getfloat('ALGORITHM_PARAMETERS', 'aco_max_pheromone', 10.0)
        
        # Initialize fitness calculator
        self.fitness_calculator = FitnessCalculator(use_gpu=self.use_gpu)
        
        self.logger.info(f"Initialized ACO with colony_size={self.colony_size}, "
                        f"alpha={self.alpha}, beta={self.beta}, GPU mode={self.use_gpu}")
    
    def optimize(self, data: Union[np.ndarray, 'cudf.DataFrame'], 
                portfolio_size: Union[int, Tuple[int, int]], 
                fitness_function: Optional[Callable] = None, 
                zone_data: Optional[Dict] = None) -> Dict:
        """
        Run ant colony optimization (updated for cuDF support)
        
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
        
        # Initialize pheromone matrix
        pheromone = np.ones((num_strategies, num_strategies)) * 0.1
        
        # Track best solution
        best_solution = None
        best_fitness = -float('inf')
        fitness_history = []
        
        # ACO iterations
        for iteration in range(self.iterations):
            # Generate solutions for all ants
            solutions = []
            fitness_scores = []
            
            for ant in range(self.colony_size):
                # Construct solution
                solution = self._construct_solution(
                    strategy_list, actual_size, pheromone, zone_data
                )
                fitness = fitness_function(solution)
                
                solutions.append(solution)
                fitness_scores.append(fitness)
                
                # Update best
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_solution = solution.copy()
            
            fitness_history.append(best_fitness)
            
            # Update pheromones
            self._update_pheromones(pheromone, solutions, fitness_scores, strategy_list)
            
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
            'iterations': self.iterations,
            'colony_size': self.colony_size,
            'fitness_history': fitness_history,
            'algorithm_name': 'ant_colony_optimization',
            'data_type': data_type,
            'gpu_accelerated': self.use_gpu,
            'detailed_metrics': detailed_metrics,
            'parameters': {
                'alpha': self.alpha,
                'beta': self.beta,
                'evaporation_rate': self.evaporation_rate,
                'pheromone_deposit': self.pheromone_deposit,
                'min_pheromone': self.min_pheromone,
                'max_pheromone': self.max_pheromone,
                'num_strategies': num_strategies,
                'portfolio_size': actual_size
            }
        }
    
    def _construct_solution(self, strategy_list: List[Union[int, str]], portfolio_size: int,
                           pheromone: np.ndarray, zone_data: Optional[Dict] = None) -> List[Union[int, str]]:
        """Construct solution using pheromone trails"""
        solution = []
        available = list(range(len(strategy_list)))
        
        # Start from random strategy
        current = random.choice(available)
        solution.append(current)
        available.remove(current)
        
        # Build rest of portfolio
        while len(solution) < portfolio_size and available:
            # Calculate probabilities
            probabilities = []
            for next_strategy in available:
                # Pheromone influence
                tau = pheromone[current, next_strategy] ** self.alpha
                # Heuristic (inverse of distance/difference)
                eta = 1.0 ** self.beta
                probabilities.append(tau * eta)
            
            # Normalize probabilities
            total = sum(probabilities)
            if total > 0:
                probabilities = [p / total for p in probabilities]
            else:
                probabilities = [1.0 / len(available)] * len(available)
            
            # Select next strategy
            next_idx = np.random.choice(len(available), p=probabilities)
            next_strategy = available[next_idx]
            
            solution.append(next_strategy)
            available.remove(next_strategy)
            current = next_strategy
        
        # Convert indices to actual strategy identifiers
        return [strategy_list[idx] for idx in solution]
    
    def _update_pheromones(self, pheromone: np.ndarray, solutions: List[List[Union[int, str]]],
                          fitness_scores: List[float], strategy_list: List[Union[int, str]]) -> None:
        """Update pheromone trails with fix for negative probability bug"""
        # Evaporation
        pheromone *= (1 - self.evaporation_rate)
        
        # Ensure minimum pheromone level to avoid zero probabilities
        pheromone = np.maximum(pheromone, self.min_pheromone)
        
        # Normalize fitness scores to avoid negative deposits
        min_fitness = min(fitness_scores) if fitness_scores else 0
        normalized_scores = [(f - min_fitness + 0.001) for f in fitness_scores]  # Add small epsilon
        
        # Deposit new pheromones
        for solution, norm_fitness in zip(solutions, normalized_scores):
            # Calculate deposit based on normalized fitness
            deposit = self.pheromone_deposit * norm_fitness
            
            # Convert solution to indices if needed
            indices = [strategy_list.index(s) if isinstance(s, str) else s for s in solution]
            
            # Update pheromone trails
            for i in range(len(indices) - 1):
                pheromone[indices[i], indices[i+1]] += deposit
                pheromone[indices[i+1], indices[i]] += deposit  # Symmetric
        
        # Cap maximum pheromone to avoid overflow
        pheromone = np.minimum(pheromone, self.max_pheromone)
