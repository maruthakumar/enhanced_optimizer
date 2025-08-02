#!/usr/bin/env python3
"""
Ant Colony Optimization Implementation for Heavy Optimizer Platform

Implements ACO algorithm with configuration support.
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


class AntColonyOptimization(BaseOptimizationAlgorithm):
    """Ant Colony Optimization implementation with configuration support"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize ACO with configuration
        
        Args:
            config_path: Path to configuration file
        """
        super().__init__(config_path)
        
        # Get configuration manager
        self.config_manager = get_config_manager()
        
        # Load algorithm-specific parameters
        self.colony_size = self.config_manager.getint('ALGORITHM_PARAMETERS', 'aco_colony_size', 20)
        self.alpha = self.config_manager.getfloat('ALGORITHM_PARAMETERS', 'aco_alpha', 1.0)
        self.beta = self.config_manager.getfloat('ALGORITHM_PARAMETERS', 'aco_beta', 2.0)
        self.evaporation_rate = self.config_manager.getfloat('ALGORITHM_PARAMETERS', 'aco_evaporation_rate', 0.5)
        self.pheromone_deposit = self.config_manager.getfloat('ALGORITHM_PARAMETERS', 'aco_pheromone_deposit', 1.0)
        self.iterations = self.config_manager.getint('ALGORITHM_PARAMETERS', 'aco_iterations', 50)
        
        self.logger.info(f"Initialized ACO with colony_size={self.colony_size}, "
                        f"alpha={self.alpha}, beta={self.beta}")
    
    def optimize(self, daily_matrix: np.ndarray, portfolio_size: Union[int, Tuple[int, int]], 
                fitness_function: Callable, zone_data: Optional[Dict] = None) -> Dict:
        """
        Run ant colony optimization
        
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
                    num_strategies, actual_size, pheromone, zone_data
                )
                fitness = fitness_function(daily_matrix, solution)
                
                solutions.append(solution)
                fitness_scores.append(fitness)
                
                # Update best
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_solution = solution.copy()
            
            fitness_history.append(best_fitness)
            
            # Update pheromones
            self._update_pheromones(pheromone, solutions, fitness_scores)
            
            # Log progress
            if iteration % 10 == 0:
                self.logger.debug(f"Iteration {iteration}: Best fitness = {best_fitness:.6f}")
        
        execution_time = time.time() - start_time
        
        return {
            'best_fitness': float(best_fitness),
            'best_portfolio': best_solution.tolist() if best_solution is not None else [],
            'execution_time': execution_time,
            'iterations': self.iterations,
            'colony_size': self.colony_size,
            'fitness_history': fitness_history,
            'algorithm': 'ant_colony_optimization',
            'parameters': {
                'alpha': self.alpha,
                'beta': self.beta,
                'evaporation_rate': self.evaporation_rate,
                'pheromone_deposit': self.pheromone_deposit
            }
        }
    
    def _construct_solution(self, num_strategies: int, portfolio_size: int,
                           pheromone: np.ndarray, zone_data: Optional[Dict] = None) -> np.ndarray:
        """Construct solution using pheromone trails"""
        solution = []
        available = list(range(num_strategies))
        
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
        
        return np.array(solution)
    
    def _update_pheromones(self, pheromone: np.ndarray, solutions: List[np.ndarray],
                          fitness_scores: List[float]) -> None:
        """Update pheromone trails"""
        # Evaporation
        pheromone *= (1 - self.evaporation_rate)
        
        # Deposit new pheromones
        for solution, fitness in zip(solutions, fitness_scores):
            if fitness > 0:
                deposit = self.pheromone_deposit * fitness
                for i in range(len(solution) - 1):
                    pheromone[solution[i], solution[i+1]] += deposit
                    pheromone[solution[i+1], solution[i]] += deposit  # Symmetric
