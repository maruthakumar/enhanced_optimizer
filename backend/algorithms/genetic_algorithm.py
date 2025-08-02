#!/usr/bin/env python3
"""
Genetic Algorithm Implementation for Heavy Optimizer Platform

Implements genetic algorithm with configuration support.
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


class GeneticAlgorithm(BaseOptimizationAlgorithm):
    """Genetic Algorithm implementation with configuration support"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize Genetic Algorithm with configuration
        
        Args:
            config_path: Path to configuration file
        """
        super().__init__(config_path)
        
        # Get configuration manager
        self.config_manager = get_config_manager()
        
        # Load algorithm-specific parameters
        self.population_size = self.config_manager.getint('ALGORITHM_PARAMETERS', 'ga_population_size', 30)
        self.mutation_rate = self.config_manager.getfloat('ALGORITHM_PARAMETERS', 'ga_mutation_rate', 0.1)
        self.crossover_rate = self.config_manager.getfloat('ALGORITHM_PARAMETERS', 'ga_crossover_rate', 0.8)
        self.generations = self.config_manager.getint('ALGORITHM_PARAMETERS', 'ga_generations', 100)
        self.selection_method = self.config_manager.get('ALGORITHM_PARAMETERS', 'ga_selection_method', 'tournament')
        self.elitism_rate = self.config_manager.getfloat('ALGORITHM_PARAMETERS', 'ga_elitism_rate', 0.1)
        self.tournament_size = self.config_manager.getint('ALGORITHM_PARAMETERS', 'ga_tournament_size', 3)
        
        self.logger.info(f"Initialized GA with population_size={self.population_size}, "
                        f"generations={self.generations}, mutation_rate={self.mutation_rate}")
    
    def optimize(self, daily_matrix: np.ndarray, portfolio_size: Union[int, Tuple[int, int]], 
                fitness_function: Callable, zone_data: Optional[Dict] = None) -> Dict:
        """
        Run genetic algorithm optimization
        
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
        
        # Initialize population
        population = self._initialize_population(num_strategies, actual_size, zone_data)
        
        # Evolution tracking
        best_individual = None
        best_fitness = -float('inf')
        fitness_history = []
        
        # Evolution loop
        for generation in range(self.generations):
            # Evaluate fitness for all individuals
            fitness_scores = []
            for individual in population:
                fitness = fitness_function(daily_matrix, individual)
                fitness_scores.append(fitness)
                
                # Track best
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_individual = individual.copy()
            
            fitness_history.append(best_fitness)
            
            # Create new generation
            new_population = []
            
            # Elitism - keep best individuals
            elite_count = int(self.population_size * self.elitism_rate)
            if elite_count > 0:
                elite_indices = np.argsort(fitness_scores)[-elite_count:]
                for idx in elite_indices:
                    new_population.append(population[idx].copy())
            
            # Generate rest of population
            while len(new_population) < self.population_size:
                # Selection
                parent1 = self._select_parent(population, fitness_scores)
                parent2 = self._select_parent(population, fitness_scores)
                
                # Crossover
                if random.random() < self.crossover_rate:
                    child1, child2 = self._crossover(parent1, parent2, zone_data)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()
                
                # Mutation
                if random.random() < self.mutation_rate:
                    child1 = self._mutate(child1, num_strategies, zone_data)
                if random.random() < self.mutation_rate:
                    child2 = self._mutate(child2, num_strategies, zone_data)
                
                # Add to new population
                new_population.extend([child1, child2])
            
            # Trim to exact population size
            population = new_population[:self.population_size]
            
            # Log progress periodically
            if generation % 20 == 0:
                self.logger.debug(f"Generation {generation}: Best fitness = {best_fitness:.6f}")
        
        execution_time = time.time() - start_time
        
        return {
            'best_fitness': float(best_fitness),
            'best_portfolio': best_individual.tolist() if best_individual is not None else [],
            'execution_time': execution_time,
            'generations': self.generations,
            'population_size': self.population_size,
            'fitness_history': fitness_history,
            'algorithm': 'genetic_algorithm',
            'parameters': {
                'mutation_rate': self.mutation_rate,
                'crossover_rate': self.crossover_rate,
                'elitism_rate': self.elitism_rate,
                'selection_method': self.selection_method
            }
        }
    
    def _initialize_population(self, num_strategies: int, portfolio_size: int,
                              zone_data: Optional[Dict] = None) -> List[np.ndarray]:
        """Initialize population with valid portfolios"""
        population = []
        
        for _ in range(self.population_size):
            if zone_data and zone_data.get('enabled'):
                # Create portfolio respecting zone constraints
                individual = self._create_zone_constrained_portfolio(
                    num_strategies, portfolio_size, zone_data
                )
            else:
                # Random portfolio
                individual = np.random.choice(num_strategies, portfolio_size, replace=False)
            
            population.append(individual)
        
        return population
    
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
    
    def _select_parent(self, population: List[np.ndarray], 
                      fitness_scores: List[float]) -> np.ndarray:
        """Select parent using configured selection method"""
        if self.selection_method == 'tournament':
            # Tournament selection
            tournament_indices = np.random.choice(len(population), 
                                                self.tournament_size, 
                                                replace=False)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            return population[winner_idx].copy()
        
        elif self.selection_method == 'roulette':
            # Roulette wheel selection
            # Shift fitness to be positive
            min_fitness = min(fitness_scores)
            shifted_fitness = [f - min_fitness + 1 for f in fitness_scores]
            total_fitness = sum(shifted_fitness)
            
            if total_fitness == 0:
                # All equal fitness, random selection
                return population[random.randint(0, len(population) - 1)].copy()
            
            probabilities = [f / total_fitness for f in shifted_fitness]
            selected_idx = np.random.choice(len(population), p=probabilities)
            return population[selected_idx].copy()
        
        else:
            # Default to random selection
            return population[random.randint(0, len(population) - 1)].copy()
    
    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray,
                   zone_data: Optional[Dict] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Perform crossover between two parents"""
        size = len(parent1)
        
        # Order crossover (OX) - preserves no duplicates
        # Select crossover points
        point1 = random.randint(0, size - 1)
        point2 = random.randint(point1 + 1, size)
        
        # Create children
        child1 = np.full(size, -1, dtype=int)
        child2 = np.full(size, -1, dtype=int)
        
        # Copy segment from parents
        child1[point1:point2] = parent1[point1:point2]
        child2[point1:point2] = parent2[point1:point2]
        
        # Fill remaining positions
        self._fill_child(child1, parent2)
        self._fill_child(child2, parent1)
        
        return child1, child2
    
    def _fill_child(self, child: np.ndarray, parent: np.ndarray) -> None:
        """Fill remaining positions in child from parent"""
        pos = 0
        for gene in parent:
            if gene not in child:
                while pos < len(child) and child[pos] != -1:
                    pos += 1
                if pos < len(child):
                    child[pos] = gene
    
    def _mutate(self, individual: np.ndarray, num_strategies: int,
                zone_data: Optional[Dict] = None) -> np.ndarray:
        """Mutate individual by swapping with random strategy"""
        mutated = individual.copy()
        
        # Select position to mutate
        pos = random.randint(0, len(mutated) - 1)
        
        # Find strategies not in portfolio
        available = np.setdiff1d(np.arange(num_strategies), mutated)
        
        if len(available) > 0:
            # Swap with random available strategy
            new_strategy = np.random.choice(available)
            mutated[pos] = new_strategy
        
        return mutated