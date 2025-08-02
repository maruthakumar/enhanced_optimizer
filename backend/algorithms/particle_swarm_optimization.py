#!/usr/bin/env python3
"""
Particle Swarm Optimization Implementation for Heavy Optimizer Platform

Implements PSO algorithm with configuration support.
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


class ParticleSwarmOptimization(BaseOptimizationAlgorithm):
    """Particle Swarm Optimization implementation with configuration support"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize PSO with configuration
        
        Args:
            config_path: Path to configuration file
        """
        super().__init__(config_path)
        
        # Get configuration manager
        self.config_manager = get_config_manager()
        
        # Load algorithm-specific parameters
        self.swarm_size = self.config_manager.getint('ALGORITHM_PARAMETERS', 'pso_swarm_size', 25)
        self.inertia_weight = self.config_manager.getfloat('ALGORITHM_PARAMETERS', 'pso_inertia_weight', 0.9)
        self.cognitive_coef = self.config_manager.getfloat('ALGORITHM_PARAMETERS', 'pso_cognitive_coefficient', 2.0)
        self.social_coef = self.config_manager.getfloat('ALGORITHM_PARAMETERS', 'pso_social_coefficient', 2.0)
        self.max_velocity = self.config_manager.getfloat('ALGORITHM_PARAMETERS', 'pso_max_velocity', 0.5)
        self.iterations = self.config_manager.getint('ALGORITHM_PARAMETERS', 'pso_iterations', 75)
        self.min_inertia = self.config_manager.getfloat('ALGORITHM_PARAMETERS', 'pso_min_inertia', 0.4)
        self.max_inertia = self.config_manager.getfloat('ALGORITHM_PARAMETERS', 'pso_max_inertia', 0.9)
        
        self.logger.info(f"Initialized PSO with swarm_size={self.swarm_size}, "
                        f"iterations={self.iterations}, inertia={self.inertia_weight}")
    
    def optimize(self, daily_matrix: np.ndarray, portfolio_size: Union[int, Tuple[int, int]], 
                fitness_function: Callable, zone_data: Optional[Dict] = None) -> Dict:
        """
        Run PSO optimization
        
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
        
        # Initialize swarm
        swarm_positions = self._initialize_swarm(num_strategies, actual_size, zone_data)
        swarm_velocities = self._initialize_velocities(self.swarm_size, actual_size, num_strategies)
        
        # Initialize personal and global bests
        personal_best_positions = [pos.copy() for pos in swarm_positions]
        personal_best_fitness = [-float('inf')] * self.swarm_size
        global_best_position = None
        global_best_fitness = -float('inf')
        
        fitness_history = []
        
        # PSO iterations
        for iteration in range(self.iterations):
            # Update inertia weight (linear decrease)
            current_inertia = self.max_inertia - (self.max_inertia - self.min_inertia) * (iteration / self.iterations)
            
            # Evaluate fitness for each particle
            for i, position in enumerate(swarm_positions):
                fitness = fitness_function(daily_matrix, position)
                
                # Update personal best
                if fitness > personal_best_fitness[i]:
                    personal_best_fitness[i] = fitness
                    personal_best_positions[i] = position.copy()
                
                # Update global best
                if fitness > global_best_fitness:
                    global_best_fitness = fitness
                    global_best_position = position.copy()
            
            fitness_history.append(global_best_fitness)
            
            # Update velocities and positions
            for i in range(self.swarm_size):
                # Update velocity
                swarm_velocities[i] = self._update_velocity(
                    swarm_velocities[i],
                    swarm_positions[i],
                    personal_best_positions[i],
                    global_best_position,
                    current_inertia
                )
                
                # Update position
                swarm_positions[i] = self._update_position(
                    swarm_positions[i],
                    swarm_velocities[i],
                    num_strategies,
                    zone_data
                )
            
            # Log progress periodically
            if iteration % 15 == 0:
                self.logger.debug(f"Iteration {iteration}: Best fitness = {global_best_fitness:.6f}")
        
        execution_time = time.time() - start_time
        
        return {
            'best_fitness': float(global_best_fitness),
            'best_portfolio': global_best_position.tolist() if global_best_position is not None else [],
            'execution_time': execution_time,
            'iterations': self.iterations,
            'swarm_size': self.swarm_size,
            'fitness_history': fitness_history,
            'algorithm': 'particle_swarm_optimization',
            'parameters': {
                'inertia_weight': self.inertia_weight,
                'cognitive_coefficient': self.cognitive_coef,
                'social_coefficient': self.social_coef,
                'max_velocity': self.max_velocity
            }
        }
    
    def _initialize_swarm(self, num_strategies: int, portfolio_size: int,
                         zone_data: Optional[Dict] = None) -> List[np.ndarray]:
        """Initialize swarm positions"""
        swarm = []
        
        for _ in range(self.swarm_size):
            if zone_data and zone_data.get('enabled'):
                # Create portfolio respecting zone constraints
                position = self._create_zone_constrained_portfolio(
                    num_strategies, portfolio_size, zone_data
                )
            else:
                # Random portfolio
                position = np.random.choice(num_strategies, portfolio_size, replace=False)
            
            swarm.append(position)
        
        return swarm
    
    def _initialize_velocities(self, swarm_size: int, portfolio_size: int, 
                              num_strategies: int) -> List[np.ndarray]:
        """Initialize particle velocities"""
        velocities = []
        
        for _ in range(swarm_size):
            # Initialize with small random velocities
            velocity = np.random.uniform(-self.max_velocity, self.max_velocity, portfolio_size)
            velocities.append(velocity)
        
        return velocities
    
    def _update_velocity(self, velocity: np.ndarray, position: np.ndarray,
                        personal_best: np.ndarray, global_best: np.ndarray,
                        inertia: float) -> np.ndarray:
        """Update particle velocity using PSO equations"""
        # Random factors
        r1 = np.random.random(len(velocity))
        r2 = np.random.random(len(velocity))
        
        # Calculate position differences (discrete space handling)
        personal_diff = self._calculate_position_difference(personal_best, position)
        global_diff = self._calculate_position_difference(global_best, position)
        
        # Update velocity
        new_velocity = (inertia * velocity + 
                       self.cognitive_coef * r1 * personal_diff +
                       self.social_coef * r2 * global_diff)
        
        # Clamp velocity
        new_velocity = np.clip(new_velocity, -self.max_velocity, self.max_velocity)
        
        return new_velocity
    
    def _calculate_position_difference(self, target: np.ndarray, current: np.ndarray) -> np.ndarray:
        """Calculate difference between positions in discrete space"""
        diff = np.zeros(len(current))
        
        for i in range(len(current)):
            if current[i] != target[i]:
                # Encourage movement towards target
                diff[i] = 1.0 if random.random() < 0.5 else -1.0
        
        return diff
    
    def _update_position(self, position: np.ndarray, velocity: np.ndarray,
                        num_strategies: int, zone_data: Optional[Dict] = None) -> np.ndarray:
        """Update particle position based on velocity"""
        new_position = position.copy()
        
        # Apply velocity probabilistically (since we're in discrete space)
        for i in range(len(position)):
            if abs(velocity[i]) > random.random():
                # Need to change this position
                available = np.setdiff1d(np.arange(num_strategies), new_position)
                
                if len(available) > 0:
                    # Replace with random available strategy
                    new_strategy = np.random.choice(available)
                    new_position[i] = new_strategy
        
        return new_position
    
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