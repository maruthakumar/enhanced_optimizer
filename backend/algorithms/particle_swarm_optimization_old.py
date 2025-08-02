#!/usr/bin/env python3
"""
Particle Swarm Optimization Module for Heavy Optimizer Platform

This module implements the Particle Swarm Optimization (PSO) algorithm as an 
independent, configurable optimization algorithm following the modular architecture.
"""

import numpy as np
import random
import time
import logging
from typing import Dict, List, Tuple, Optional, Union
from .base_algorithm import BaseOptimizationAlgorithm


class ParticleSwarmOptimization(BaseOptimizationAlgorithm):
    """Particle Swarm Optimization implementation for portfolio optimization"""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize PSO with configuration"""
        super().__init__(config_path)
        
        # Default PSO parameters (can be overridden by config)
        self.swarm_size = self._get_config_value('swarm_size', 30, int)
        self.iterations = self._get_config_value('iterations', 100, int)
        self.inertia = self._get_config_value('inertia', 0.7, float)
        self.cognitive_coef = self._get_config_value('acceleration_coef_1', 1.5, float)
        self.social_coef = self._get_config_value('acceleration_coef_2', 1.5, float)
        self.velocity_max = self._get_config_value('velocity_max', 4.0, float)
        
        self.logger.info(f"Initialized PSO with swarm_size={self.swarm_size}, "
                        f"iterations={self.iterations}, inertia={self.inertia}")
    
    def optimize(self, 
                daily_matrix: np.ndarray, 
                portfolio_size: Union[int, Tuple[int, int]], 
                fitness_function: callable,
                zone_data: Optional[Dict] = None) -> Dict:
        """Run Particle Swarm Optimization"""
        start_time = time.time()
        
        # Validate inputs
        self.validate_inputs(daily_matrix, portfolio_size)
        
        # Determine actual portfolio size
        actual_size = self._determine_portfolio_size(portfolio_size)
        num_strategies = daily_matrix.shape[1]
        
        # Initialize swarm
        particles, velocities = self._initialize_swarm(num_strategies, actual_size, zone_data)
        personal_best_positions = [p.copy() for p in particles]
        personal_best_scores = [-np.inf] * self.swarm_size
        
        # Track global best
        global_best_position = None
        global_best_score = -np.inf
        
        iteration_stats = []
        
        # PSO iterations
        for iteration in range(self.iterations):
            # Evaluate all particles
            scores = []
            for i, particle in enumerate(particles):
                score = fitness_function(daily_matrix, particle)
                scores.append(score)
                
                # Update personal best
                if score > personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = particle.copy()
                
                # Update global best
                if score > global_best_score:
                    global_best_score = score
                    global_best_position = particle.copy()
            
            # Record iteration statistics
            avg_score = np.mean(scores)
            iteration_stats.append({
                'iteration': iteration,
                'best_score': global_best_score,
                'avg_score': avg_score,
                'std_score': np.std(scores)
            })
            
            # Log progress periodically
            if iteration % 20 == 0:
                self.logger.debug(f"Iter {iteration}: Best={global_best_score:.4f}, "
                                 f"Avg={avg_score:.4f}")
            
            # Update velocities and positions (skip on last iteration)
            if iteration < self.iterations - 1:
                particles, velocities = self._update_swarm(
                    particles, velocities, personal_best_positions, 
                    global_best_position, num_strategies, zone_data
                )
        
        # Prepare results
        execution_time = self._calculate_execution_time(start_time)
        
        return {
            'best_portfolio': global_best_position.tolist(),
            'best_fitness': float(global_best_score),
            'execution_time': execution_time,
            'algorithm_name': 'ParticleSwarmOptimization',
            'iterations': self.iterations,
            'swarm_size': self.swarm_size,
            'iteration_stats': iteration_stats,
            'final_avg_fitness': iteration_stats[-1]['avg_score']
        }
    
    def _initialize_swarm(self, num_strategies: int, portfolio_size: int,
                         zone_data: Optional[Dict] = None) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Initialize particle positions and velocities"""
        particles = []
        velocities = []
        
        for _ in range(self.swarm_size):
            # Initialize position (portfolio)
            if zone_data and 'allowed_strategies' in zone_data:
                allowed = zone_data['allowed_strategies']
                if len(allowed) >= portfolio_size:
                    position = np.random.choice(allowed, portfolio_size, replace=False)
                else:
                    position = np.array(allowed)
                    remaining = portfolio_size - len(position)
                    other = [s for s in range(num_strategies) if s not in allowed]
                    if other and remaining > 0:
                        additional = np.random.choice(other, min(remaining, len(other)), replace=False)
                        position = np.concatenate([position, additional])
            else:
                position = np.random.choice(num_strategies, portfolio_size, replace=False)
            
            # Initialize velocity (continuous values for each position)
            velocity = np.random.uniform(-self.velocity_max, self.velocity_max, portfolio_size)
            
            particles.append(position)
            velocities.append(velocity)
        
        return particles, velocities
    
    def _update_swarm(self, particles: List[np.ndarray], velocities: List[np.ndarray],
                     personal_best_positions: List[np.ndarray], global_best_position: np.ndarray,
                     num_strategies: int, zone_data: Optional[Dict] = None) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Update particle velocities and positions"""
        new_particles = []
        new_velocities = []
        
        for i in range(self.swarm_size):
            # Update velocity using PSO equations
            r1 = np.random.random(len(particles[i]))
            r2 = np.random.random(len(particles[i]))
            
            # Velocity update equation
            cognitive = self.cognitive_coef * r1 * (personal_best_positions[i] - particles[i])
            social = self.social_coef * r2 * (global_best_position - particles[i])
            new_velocity = self.inertia * velocities[i] + cognitive + social
            
            # Clamp velocity
            new_velocity = np.clip(new_velocity, -self.velocity_max, self.velocity_max)
            new_velocities.append(new_velocity)
            
            # Update position using discrete PSO approach
            new_position = self._update_discrete_position(
                particles[i], new_velocity, num_strategies
            )
            
            # Apply zone constraints if needed
            if zone_data:
                new_position = self._apply_zone_constraints(new_position.tolist(), zone_data)
                new_position = np.array(new_position)
            
            new_particles.append(new_position)
        
        return new_particles, new_velocities
    
    def _update_discrete_position(self, position: np.ndarray, velocity: np.ndarray,
                                 num_strategies: int) -> np.ndarray:
        """Update discrete position based on velocity"""
        new_position = position.copy()
        
        # Use velocity magnitudes as probabilities for position updates
        update_probs = 1 / (1 + np.exp(-velocity))  # Sigmoid function
        
        for i in range(len(position)):
            if random.random() < update_probs[i]:
                # Replace this strategy with a new one
                available = [s for s in range(num_strategies) if s not in new_position]
                if available:
                    new_position[i] = random.choice(available)
        
        return new_position
                portfolio = np.random.choice(num_strategies, portfolio_size, replace=False)
                fitness = fitness_function(daily_matrix, portfolio)
                
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_portfolio = portfolio
            
            execution_time = time.time() - start_time
            self.success_count += 1
            self.total_execution_time += execution_time
            
            return {
                'best_fitness': float(best_fitness),
                'best_portfolio': best_portfolio.tolist() if best_portfolio is not None else [],
                'execution_time': execution_time,
                'iterations': iterations,
                'algorithm': self.algorithm_name
            }
            
        except Exception as e:
            self.error_count += 1
            execution_time = time.time() - start_time
            
            self.logger.error(f"❌ PSO failed: {str(e)}")
            
            return {
                'best_fitness': 0.0,
                'best_portfolio': [],
                'execution_time': execution_time,
                'algorithm': self.algorithm_name,
                'error': str(e),
                'status': 'FAILED'
            }
    
    def get_algorithm_info(self) -> Dict[str, Any]:
        """Get algorithm information"""
        avg_execution_time = (self.total_execution_time / self.execution_count) if self.execution_count > 0 else 0.0
        success_rate = (self.success_count / self.execution_count * 100) if self.execution_count > 0 else 0.0
        
        return {
            'algorithm_name': self.algorithm_name,
            'algorithm_description': self.algorithm_description,
            'execution_count': self.execution_count,
            'success_count': self.success_count,
            'error_count': self.error_count,
            'success_rate': success_rate,
            'total_execution_time': self.total_execution_time,
            'average_execution_time': avg_execution_time,
            'status': 'PRODUCTION_READY'
        }
