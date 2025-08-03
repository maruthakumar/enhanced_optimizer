#!/usr/bin/env python3
"""
Genetic Algorithm Implementation for Heavy Optimizer Platform

Implements genetic algorithm with configuration support.

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


class GeneticAlgorithm(BaseOptimizationAlgorithm):
    """Genetic Algorithm implementation with configuration support"""
    
    def __init__(self, config_path: Optional[str] = None, use_gpu: bool = True):
        """
        Initialize Genetic Algorithm with configuration
        
        Args:
            config_path: Path to configuration file
            use_gpu: Whether to use GPU acceleration with cuDF
        """
        super().__init__(config_path, use_gpu)
        
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
        
        # Initialize fitness calculator
        self.fitness_calculator = FitnessCalculator(use_gpu=self.use_gpu)
        
        self.logger.info(f"Initialized GA with population_size={self.population_size}, "
                        f"generations={self.generations}, mutation_rate={self.mutation_rate}, "
                        f"GPU mode={self.use_gpu}")
    
    def optimize(self, data: Union[np.ndarray, 'cudf.DataFrame'], 
                portfolio_size: Union[int, Tuple[int, int]], 
                fitness_function: Optional[Callable] = None, 
                zone_data: Optional[Dict] = None) -> Dict:
        """
        Run genetic algorithm optimization (updated for cuDF support)
        
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
        
        # Initialize population
        population = self._initialize_population(strategy_list, actual_size, zone_data)
        
        # Evolution tracking
        best_individual = None
        best_fitness = -float('inf')
        fitness_history = []
        
        # Evolution loop
        for generation in range(self.generations):
            # Evaluate fitness for all individuals
            fitness_scores = []
            for individual in population:
                fitness = fitness_function(individual)
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
                    child1 = self._mutate(child1, strategy_list, zone_data)
                if random.random() < self.mutation_rate:
                    child2 = self._mutate(child2, strategy_list, zone_data)
                
                # Add to new population
                new_population.extend([child1, child2])
            
            # Trim to exact population size
            population = new_population[:self.population_size]
            
            # Log progress periodically
            if generation % 20 == 0:
                self.logger.debug(f"Generation {generation}: Best fitness = {best_fitness:.6f}")
        
        execution_time = time.time() - start_time
        
        # Calculate detailed metrics for the best solution
        detailed_metrics = {}
        if best_individual is not None:
            try:
                detailed_metrics = self.fitness_calculator.calculate_detailed_metrics(data, best_individual)
            except Exception as e:
                self.logger.warning(f"Could not calculate detailed metrics: {e}")
        
        return {
            'best_fitness': float(best_fitness),
            'best_portfolio': best_individual if best_individual is not None else [],
            'execution_time': execution_time,
            'generations': self.generations,
            'population_size': self.population_size,
            'fitness_history': fitness_history,
            'algorithm_name': 'genetic_algorithm',
            'data_type': data_type,
            'gpu_accelerated': self.use_gpu,
            'detailed_metrics': detailed_metrics,
            'parameters': {
                'mutation_rate': self.mutation_rate,
                'crossover_rate': self.crossover_rate,
                'elitism_rate': self.elitism_rate,
                'selection_method': self.selection_method,
                'num_strategies': num_strategies,
                'portfolio_size': actual_size
            }
        }
    
    def _initialize_population(self, strategy_list: List[Union[int, str]], portfolio_size: int,
                              zone_data: Optional[Dict] = None) -> List[List[Union[int, str]]]:
        """Initialize population with valid portfolios"""
        population = []
        num_strategies = len(strategy_list)
        
        for _ in range(self.population_size):
            if zone_data and zone_data.get('enabled'):
                # Create portfolio respecting zone constraints
                individual = self._create_zone_constrained_portfolio(
                    strategy_list, portfolio_size, zone_data
                )
            else:
                # Random portfolio
                individual = list(np.random.choice(strategy_list, portfolio_size, replace=False))
            
            population.append(individual)
        
        return population
    
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
        
        # Ensure we have exactly portfolio_size strategies
        if len(portfolio) > portfolio_size:
            portfolio = list(np.random.choice(portfolio, portfolio_size, replace=False))
        elif len(portfolio) < portfolio_size:
            # Add more strategies randomly
            remaining = portfolio_size - len(portfolio)
            available = [s for s in strategy_list if s not in portfolio]
            if len(available) > 0:
                additional = list(np.random.choice(available, 
                                                 min(remaining, len(available)), 
                                                 replace=False))
                portfolio.extend(additional)
        
        return portfolio
    
    def _select_parent(self, population: List[List[Union[int, str]]], 
                      fitness_scores: List[float]) -> List[Union[int, str]]:
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
    
    def _crossover(self, parent1: List[Union[int, str]], parent2: List[Union[int, str]],
                   zone_data: Optional[Dict] = None) -> Tuple[List[Union[int, str]], List[Union[int, str]]]:
        """Perform crossover between two parents"""
        size = len(parent1)
        
        # Order crossover (OX) - preserves no duplicates
        # Select crossover points
        point1 = random.randint(0, size - 1)
        point2 = random.randint(point1 + 1, size)
        
        # Create children with None placeholders
        child1 = [None] * size
        child2 = [None] * size
        
        # Copy segment from parents
        child1[point1:point2] = parent1[point1:point2]
        child2[point1:point2] = parent2[point1:point2]
        
        # Fill remaining positions
        self._fill_child(child1, parent2)
        self._fill_child(child2, parent1)
        
        return child1, child2
    
    def _fill_child(self, child: List[Union[int, str, None]], 
                   parent: List[Union[int, str]]) -> None:
        """Fill remaining positions in child from parent"""
        pos = 0
        for gene in parent:
            if gene not in child:
                while pos < len(child) and child[pos] is not None:
                    pos += 1
                if pos < len(child):
                    child[pos] = gene
    
    def _mutate(self, individual: List[Union[int, str]], 
               strategy_list: List[Union[int, str]],
               zone_data: Optional[Dict] = None) -> List[Union[int, str]]:
        """Mutate individual by swapping with random strategy"""
        mutated = individual.copy()
        
        # Select position to mutate
        pos = random.randint(0, len(mutated) - 1)
        
        # Find strategies not in portfolio
        available = [s for s in strategy_list if s not in mutated]
        
        if len(available) > 0:
            # Swap with random available strategy
            new_strategy = random.choice(available)
            mutated[pos] = new_strategy
        
        return mutated