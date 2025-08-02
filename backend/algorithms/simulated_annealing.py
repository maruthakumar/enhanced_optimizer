#!/usr/bin/env python3
"""
Simulated Annealing Implementation for Heavy Optimizer Platform

Implements SA algorithm with configuration support.
"""

import numpy as np
import random
import math
from typing import Dict, List, Tuple, Optional, Union, Callable
import time
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithms.base_algorithm import BaseOptimizationAlgorithm
from config.config_manager import get_config_manager


class SimulatedAnnealing(BaseOptimizationAlgorithm):
    """Simulated Annealing implementation with configuration support"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize SA with configuration
        
        Args:
            config_path: Path to configuration file
        """
        super().__init__(config_path)
        
        # Get configuration manager
        self.config_manager = get_config_manager()
        
        # Load algorithm-specific parameters
        self.initial_temp = self.config_manager.getfloat('ALGORITHM_PARAMETERS', 'sa_initial_temperature', 1000.0)
        self.final_temp = self.config_manager.getfloat('ALGORITHM_PARAMETERS', 'sa_final_temperature', 0.01)
        self.cooling_rate = self.config_manager.getfloat('ALGORITHM_PARAMETERS', 'sa_cooling_rate', 0.95)
        self.max_iterations = self.config_manager.getint('ALGORITHM_PARAMETERS', 'sa_max_iterations', 1000)
        self.acceptance_method = self.config_manager.get('ALGORITHM_PARAMETERS', 'sa_acceptance_probability', 'boltzmann')
        
        self.logger.info(f"Initialized SA with initial_temp={self.initial_temp}, "
                        f"cooling_rate={self.cooling_rate}, max_iterations={self.max_iterations}")
    
    def optimize(self, daily_matrix: np.ndarray, portfolio_size: Union[int, Tuple[int, int]], 
                fitness_function: Callable, zone_data: Optional[Dict] = None) -> Dict:
        """
        Run simulated annealing optimization
        
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
        
        # Initialize solution
        if zone_data and zone_data.get('enabled'):
            current_solution = self._create_zone_constrained_portfolio(
                num_strategies, actual_size, zone_data
            )
        else:
            current_solution = np.random.choice(num_strategies, actual_size, replace=False)
        
        current_fitness = fitness_function(daily_matrix, current_solution)
        
        # Track best solution
        best_solution = current_solution.copy()
        best_fitness = current_fitness
        
        # Temperature and iteration tracking
        temperature = self.initial_temp
        fitness_history = [best_fitness]
        accepted_moves = 0
        rejected_moves = 0
        
        # Annealing loop
        for iteration in range(self.max_iterations):
            # Generate neighbor solution
            neighbor_solution = self._generate_neighbor(
                current_solution, num_strategies, zone_data
            )
            neighbor_fitness = fitness_function(daily_matrix, neighbor_solution)
            
            # Calculate acceptance probability
            delta_fitness = neighbor_fitness - current_fitness
            
            if delta_fitness > 0:
                # Better solution, always accept
                accept = True
            else:
                # Worse solution, accept with probability
                if self.acceptance_method == 'boltzmann':
                    probability = math.exp(delta_fitness / temperature)
                else:
                    # Linear acceptance probability
                    probability = max(0, 1 + delta_fitness / temperature)
                
                accept = random.random() < probability
            
            # Accept or reject move
            if accept:
                current_solution = neighbor_solution
                current_fitness = neighbor_fitness
                accepted_moves += 1
                
                # Update best if needed
                if current_fitness > best_fitness:
                    best_fitness = current_fitness
                    best_solution = current_solution.copy()
            else:
                rejected_moves += 1
            
            # Cool down
            temperature *= self.cooling_rate
            
            # Track fitness
            fitness_history.append(best_fitness)
            
            # Stop if temperature too low
            if temperature < self.final_temp:
                break
            
            # Log progress periodically
            if iteration % 100 == 0:
                self.logger.debug(f"Iteration {iteration}: Best fitness = {best_fitness:.6f}, "
                                f"Temperature = {temperature:.6f}")
        
        execution_time = time.time() - start_time
        
        return {
            'best_fitness': float(best_fitness),
            'best_portfolio': best_solution.tolist() if best_solution is not None else [],
            'execution_time': execution_time,
            'iterations': iteration + 1,
            'final_temperature': temperature,
            'accepted_moves': accepted_moves,
            'rejected_moves': rejected_moves,
            'acceptance_rate': accepted_moves / (accepted_moves + rejected_moves) if (accepted_moves + rejected_moves) > 0 else 0,
            'fitness_history': fitness_history,
            'algorithm': 'simulated_annealing',
            'parameters': {
                'initial_temperature': self.initial_temp,
                'final_temperature': self.final_temp,
                'cooling_rate': self.cooling_rate,
                'acceptance_method': self.acceptance_method
            }
        }
    
    def _generate_neighbor(self, current_solution: np.ndarray, num_strategies: int,
                          zone_data: Optional[Dict] = None) -> np.ndarray:
        """Generate neighbor solution by modifying current solution"""
        neighbor = current_solution.copy()
        
        # Choose modification type
        modification_type = random.choice(['swap', 'replace', 'multi_swap'])
        
        if modification_type == 'swap' and len(neighbor) > 1:
            # Swap two positions
            i, j = random.sample(range(len(neighbor)), 2)
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
            
        elif modification_type == 'replace':
            # Replace one strategy with another
            pos = random.randint(0, len(neighbor) - 1)
            available = np.setdiff1d(np.arange(num_strategies), neighbor)
            
            if len(available) > 0:
                new_strategy = np.random.choice(available)
                neighbor[pos] = new_strategy
                
        elif modification_type == 'multi_swap' and len(neighbor) > 3:
            # Multiple swaps for larger neighborhood
            num_swaps = min(3, len(neighbor) // 2)
            for _ in range(num_swaps):
                i, j = random.sample(range(len(neighbor)), 2)
                neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
        
        # Ensure zone constraints if needed
        if zone_data and zone_data.get('enabled'):
            neighbor = self._enforce_zone_constraints(neighbor, zone_data)
        
        return neighbor
    
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
        
        # Ensure we have exactly portfolio_size strategies
        portfolio = np.array(portfolio)
        if len(portfolio) > portfolio_size:
            portfolio = np.random.choice(portfolio, portfolio_size, replace=False)
        elif len(portfolio) < portfolio_size:
            # Add more strategies randomly
            remaining = portfolio_size - len(portfolio)
            available = np.setdiff1d(np.arange(num_strategies), portfolio)
            if len(available) > 0:
                additional = np.random.choice(available, 
                                            min(remaining, len(available)), 
                                            replace=False)
                portfolio = np.concatenate([portfolio, additional])
        
        return portfolio
    
    def _enforce_zone_constraints(self, portfolio: np.ndarray, zone_data: Dict) -> np.ndarray:
        """Ensure portfolio meets zone constraints"""
        # This is a simplified version - in practice, you might want more sophisticated rebalancing
        return portfolio