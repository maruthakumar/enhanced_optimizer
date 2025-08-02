#!/usr/bin/env python3
"""
Genetic Algorithm Module for Heavy Optimizer Platform

This module implements the Genetic Algorithm (GA) as an independent, 
configurable optimization algorithm following the modular architecture.
"""

import numpy as np
import random
import time
import logging
from typing import List, Dict, Tuple, Optional, Union
from .base_algorithm import BaseOptimizationAlgorithm


class GeneticAlgorithm(BaseOptimizationAlgorithm):
    """Genetic Algorithm implementation for portfolio optimization"""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize Genetic Algorithm with configuration"""
        super().__init__(config_path)
        
        # Default GA parameters (can be overridden by config)
        self.population_size = self._get_config_value('population_size', 30, int)
        self.generations = self._get_config_value('generations', 50, int)
        self.mutation_rate = self._get_config_value('mutation_rate', 0.1, float)
        self.crossover_rate = self._get_config_value('crossover_rate', 0.8, float)
        self.tournament_size = self._get_config_value('tournament_size', 3, int)
        self.elitism_count = self._get_config_value('elitism_count', 1, int)
        
        self.logger.info(f"Initialized GA with population_size={self.population_size}, "
                        f"generations={self.generations}, mutation_rate={self.mutation_rate}")
        
    def optimize(self, 
                daily_matrix: np.ndarray, 
                portfolio_size: Union[int, Tuple[int, int]], 
                fitness_function: callable,
                zone_data: Optional[Dict] = None) -> Dict:
        """Run Genetic Algorithm optimization"""
        start_time = time.time()
        
        # Validate inputs
        self.validate_inputs(daily_matrix, portfolio_size)
        
        # Determine actual portfolio size
        actual_size = self._determine_portfolio_size(portfolio_size)
        num_strategies = daily_matrix.shape[1]
        
        # Initialize population
        population = self._initialize_population(num_strategies, actual_size, zone_data)
        
        best_individual = None
        best_fitness = -np.inf
        generation_stats = []
        
        # Evolution loop
        for generation in range(self.generations):
            # Evaluate fitness for all individuals
            fitness_scores = []
            for individual in population:
                fitness = fitness_function(daily_matrix, individual)
                fitness_scores.append(fitness)
                
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_individual = individual.copy()
            
            # Record generation statistics
            avg_fitness = np.mean(fitness_scores)
            generation_stats.append({
                'generation': generation,
                'best_fitness': best_fitness,
                'avg_fitness': avg_fitness,
                'std_fitness': np.std(fitness_scores)
            })
            
            # Log progress periodically
            if generation % 10 == 0:
                self.logger.debug(f"Gen {generation}: Best={best_fitness:.4f}, "
                                 f"Avg={avg_fitness:.4f}")
            
            # Create next generation (skip on last iteration)
            if generation < self.generations - 1:
                population = self._create_next_generation(
                    population, fitness_scores, num_strategies, zone_data
                )
        
        # Prepare results
        execution_time = self._calculate_execution_time(start_time)
        
        return {
            'best_portfolio': best_individual.tolist(),
            'best_fitness': float(best_fitness),
            'execution_time': execution_time,
            'algorithm_name': 'GeneticAlgorithm',
            'generations': self.generations,
            'population_size': self.population_size,
            'generation_stats': generation_stats,
            'final_avg_fitness': generation_stats[-1]['avg_fitness']
        }
    
    def _initialize_population(self, num_strategies: int, portfolio_size: int, 
                             zone_data: Optional[Dict] = None) -> List[np.ndarray]:
        """Initialize random population"""
        population = []
        
        for _ in range(self.population_size):
            if zone_data and 'allowed_strategies' in zone_data:
                # Initialize from zone-specific strategies
                allowed = zone_data['allowed_strategies']
                if len(allowed) >= portfolio_size:
                    individual = np.random.choice(allowed, portfolio_size, replace=False)
                else:
                    # If not enough allowed strategies, use what's available
                    individual = np.array(allowed)
                    # Fill remaining with random strategies
                    remaining = portfolio_size - len(individual)
                    other_strategies = [s for s in range(num_strategies) if s not in allowed]
                    if other_strategies and remaining > 0:
                        additional = np.random.choice(other_strategies, 
                                                    min(remaining, len(other_strategies)), 
                                                    replace=False)
                        individual = np.concatenate([individual, additional])
            else:
                # Standard random initialization
                individual = np.random.choice(num_strategies, portfolio_size, replace=False)
            
            population.append(individual)
        
        return population
    
    def _create_next_generation(self, population: List[np.ndarray], 
                              fitness_scores: List[float], 
                              num_strategies: int,
                              zone_data: Optional[Dict] = None) -> List[np.ndarray]:
        """Create next generation using selection, crossover, and mutation"""
        new_population = []
        
        # Elitism: keep best individuals
        if self.elitism_count > 0:
            elite_indices = np.argsort(fitness_scores)[-self.elitism_count:]
            for idx in elite_indices:
                new_population.append(population[idx].copy())
        
        # Fill rest of population
        while len(new_population) < self.population_size:
            # Tournament selection for parent 1
            parent1 = self._tournament_selection(population, fitness_scores)
            
            # Tournament selection for parent 2
            parent2 = self._tournament_selection(population, fitness_scores)
            
            # Crossover
            if random.random() < self.crossover_rate:
                child = self._crossover(parent1, parent2, num_strategies)
            else:
                child = parent1.copy()
            
            # Mutation
            if random.random() < self.mutation_rate:
                child = self._mutate(child, num_strategies)
            
            # Apply zone constraints if needed
            if zone_data:
                child = self._apply_zone_constraints(child.tolist(), zone_data)
                child = np.array(child)
            
            new_population.append(child)
        
        return new_population[:self.population_size]
    
    def _tournament_selection(self, population: List[np.ndarray], 
                            fitness_scores: List[float]) -> np.ndarray:
        """Tournament selection"""
        tournament_indices = random.sample(range(len(population)), self.tournament_size)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        return population[winner_idx].copy()
    
    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray, 
                  num_strategies: int) -> np.ndarray:
        """Uniform crossover ensuring no duplicates"""
        child = []
        used = set()
        
        # Randomly inherit from parents
        for i in range(len(parent1)):
            if random.random() < 0.5:
                if parent1[i] not in used:
                    child.append(parent1[i])
                    used.add(parent1[i])
                elif parent2[i] not in used:
                    child.append(parent2[i])
                    used.add(parent2[i])
            else:
                if parent2[i] not in used:
                    child.append(parent2[i])
                    used.add(parent2[i])
                elif parent1[i] not in used:
                    child.append(parent1[i])
                    used.add(parent1[i])
        
        # Fill remaining slots with random strategies
        while len(child) < len(parent1):
            strategy = random.randint(0, num_strategies - 1)
            if strategy not in used:
                child.append(strategy)
                used.add(strategy)
        
        return np.array(child)
    
    def _mutate(self, individual: np.ndarray, num_strategies: int) -> np.ndarray:
        """Mutation operator - randomly replace strategies"""
        mutated = individual.copy()
        
        for i in range(len(mutated)):
            if random.random() < self.mutation_rate:
                # Find strategies not in current portfolio
                available = [s for s in range(num_strategies) if s not in mutated]
                if available:
                    mutated[i] = random.choice(available)
        
        return mutated