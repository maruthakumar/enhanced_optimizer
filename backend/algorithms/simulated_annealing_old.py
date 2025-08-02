#!/usr/bin/env python3
"""
Simulated Annealing Module for Heavy Optimizer Platform

This module implements the Simulated Annealing (SA) algorithm as an 
independent, configurable optimization algorithm following the modular architecture.
"""

import numpy as np
import random
import time
import logging
import math
from typing import Dict, List, Tuple, Optional, Union
from .base_algorithm import BaseOptimizationAlgorithm


class SimulatedAnnealing(BaseOptimizationAlgorithm):
    """Simulated Annealing implementation for portfolio optimization"""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize SA with configuration"""
        super().__init__(config_path)
        
        # Default SA parameters (can be overridden by config)
        self.initial_temperature = self._get_config_value('initial_temperature', 1.0, float)
        self.cooling_rate = self._get_config_value('cooling_rate', 0.95, float)
        self.iterations = self._get_config_value('iterations', 150, int)
        self.neighbor_selection_method = self._get_config_value('neighbor_selection', 'swap', str)
        self.min_temperature = self._get_config_value('min_temperature', 0.001, float)
        
        self.logger.info(f"Initialized SA with initial_temp={self.initial_temperature}, "
                        f"cooling_rate={self.cooling_rate}, iterations={self.iterations}")
    
    def optimize(self, 
                daily_matrix: np.ndarray, 
                portfolio_size: Union[int, Tuple[int, int]], 
                fitness_function: callable,
                zone_data: Optional[Dict] = None) -> Dict:
        """Run Simulated Annealing optimization"""
        start_time = time.time()
        
        # Validate inputs
        self.validate_inputs(daily_matrix, portfolio_size)
        
        # Determine actual portfolio size
        actual_size = self._determine_portfolio_size(portfolio_size)
        num_strategies = daily_matrix.shape[1]
        
        # Initialize solution
        current_solution = self._initialize_solution(num_strategies, actual_size, zone_data)
        current_fitness = fitness_function(daily_matrix, current_solution)
        
        best_solution = current_solution.copy()
        best_fitness = current_fitness
        temperature = self.initial_temperature
        
        iteration_stats = []
        accepted_moves = 0
        total_moves = 0
        
        # SA iterations
        for iteration in range(self.iterations):
            # Generate neighbor solution
            neighbor = self._generate_neighbor(current_solution, num_strategies, zone_data)
            neighbor_fitness = fitness_function(daily_matrix, neighbor)
            
            # Calculate acceptance probability
            delta = neighbor_fitness - current_fitness
            total_moves += 1
            
            if delta > 0 or random.random() < math.exp(delta / temperature):
                # Accept the move
                current_solution = neighbor
                current_fitness = neighbor_fitness
                accepted_moves += 1
                
                if current_fitness > best_fitness:
                    best_fitness = current_fitness
                    best_solution = current_solution.copy()
            
            # Update temperature
            temperature *= self.cooling_rate
            temperature = max(temperature, self.min_temperature)
            
            # Record iteration statistics
            iteration_stats.append({
                'iteration': iteration,
                'temperature': temperature,
                'current_fitness': current_fitness,
                'best_fitness': best_fitness,
                'acceptance_rate': accepted_moves / total_moves if total_moves > 0 else 0
            })
            
            # Log progress periodically
            if iteration % 30 == 0:
                self.logger.debug(f"Iter {iteration}: Temp={temperature:.4f}, "
                                 f"Best={best_fitness:.4f}, Current={current_fitness:.4f}")
        
        # Prepare results
        execution_time = self._calculate_execution_time(start_time)
        
        return {
            'best_portfolio': best_solution.tolist(),
            'best_fitness': float(best_fitness),
            'execution_time': execution_time,
            'algorithm_name': 'SimulatedAnnealing',
            'iterations': self.iterations,
            'initial_temperature': self.initial_temperature,
            'final_temperature': temperature,
            'cooling_rate': self.cooling_rate,
            'iteration_stats': iteration_stats,
            'acceptance_rate': accepted_moves / total_moves if total_moves > 0 else 0
        }
    
    def _initialize_solution(self, num_strategies: int, portfolio_size: int,
                           zone_data: Optional[Dict] = None) -> np.ndarray:
        """Initialize a random solution"""
        if zone_data and 'allowed_strategies' in zone_data:
            allowed = zone_data['allowed_strategies']
            if len(allowed) >= portfolio_size:
                return np.random.choice(allowed, portfolio_size, replace=False)
            else:
                solution = np.array(allowed)
                remaining = portfolio_size - len(solution)
                other = [s for s in range(num_strategies) if s not in allowed]
                if other and remaining > 0:
                    additional = np.random.choice(other, min(remaining, len(other)), replace=False)
                    solution = np.concatenate([solution, additional])
                return solution
        else:
            return np.random.choice(num_strategies, portfolio_size, replace=False)
    
    def _generate_neighbor(self, current_solution: np.ndarray, num_strategies: int,
                         zone_data: Optional[Dict] = None) -> np.ndarray:
        """Generate a neighbor solution using the configured method"""
        neighbor = current_solution.copy()
        
        if self.neighbor_selection_method == 'swap':
            # Swap two strategies
            if len(neighbor) >= 2:
                i, j = random.sample(range(len(neighbor)), 2)
                neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
        else:
            # Default: replace one strategy
            idx = random.randint(0, len(neighbor) - 1)
            available = [s for s in range(num_strategies) if s not in neighbor]
            
            if zone_data and 'allowed_strategies' in zone_data:
                # Prefer zone strategies
                zone_available = [s for s in available if s in zone_data['allowed_strategies']]
                if zone_available:
                    available = zone_available
            
            if available:
                neighbor[idx] = random.choice(available)
        
        return neighbor
