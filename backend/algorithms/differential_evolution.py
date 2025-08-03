#!/usr/bin/env python3
"""
Differential Evolution Implementation for Heavy Optimizer Platform

Implements DE algorithm with configuration support.

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


class DifferentialEvolution(BaseOptimizationAlgorithm):
    """Differential Evolution implementation with configuration support"""
    
    def __init__(self, config_path: Optional[str] = None, use_gpu: bool = True):
        """
        Initialize DE with configuration
        
        Args:
            config_path: Path to configuration file
            use_gpu: Whether to use GPU acceleration with cuDF
        """
        super().__init__(config_path, use_gpu)
        
        # Get configuration manager
        self.config_manager = get_config_manager()
        
        # Load algorithm-specific parameters
        self.population_size = self.config_manager.getint('ALGORITHM_PARAMETERS', 'de_population_size', 30)
        self.mutation_factor = self.config_manager.getfloat('ALGORITHM_PARAMETERS', 'de_mutation_factor', 0.8)
        self.crossover_prob = self.config_manager.getfloat('ALGORITHM_PARAMETERS', 'de_crossover_probability', 0.7)
        self.strategy = self.config_manager.get('ALGORITHM_PARAMETERS', 'de_strategy', 'best_1_bin')
        self.generations = self.config_manager.getint('ALGORITHM_PARAMETERS', 'de_generations', 100)
        
        # Initialize fitness calculator
        self.fitness_calculator = FitnessCalculator(use_gpu=self.use_gpu)
        
        self.logger.info(f"Initialized DE with population_size={self.population_size}, "
                        f"mutation_factor={self.mutation_factor}, strategy={self.strategy}, "
                        f"GPU mode={self.use_gpu}")
    
    def optimize(self, data: Union[np.ndarray, 'cudf.DataFrame'], 
                portfolio_size: Union[int, Tuple[int, int]], 
                fitness_function: Optional[Callable] = None, 
                zone_data: Optional[Dict] = None) -> Dict:
        """
        Run differential evolution optimization (updated for cuDF support)
        
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
        fitness_scores = [fitness_function(ind) for ind in population]
        
        # Track best
        best_idx = np.argmax(fitness_scores)
        best_individual = population[best_idx].copy()
        best_fitness = fitness_scores[best_idx]
        fitness_history = [best_fitness]
        
        # Evolution loop
        for generation in range(self.generations):
            new_population = []
            new_fitness_scores = []
            
            for i, target in enumerate(population):
                # Create mutant vector
                if self.strategy == 'best_1_bin':
                    mutant = self._mutate_best_1(population, best_individual, i)
                elif self.strategy == 'rand_1_bin':
                    mutant = self._mutate_rand_1(population, i)
                else:
                    mutant = self._mutate_rand_1(population, i)  # Default
                
                # Crossover
                trial = self._crossover(target, mutant, num_strategies)
                
                # Selection
                trial_fitness = fitness_function(trial)
                
                if trial_fitness > fitness_scores[i]:
                    new_population.append(trial)
                    new_fitness_scores.append(trial_fitness)
                    
                    # Update best
                    if trial_fitness > best_fitness:
                        best_fitness = trial_fitness
                        best_individual = trial.copy()
                else:
                    new_population.append(target)
                    new_fitness_scores.append(fitness_scores[i])
            
            population = new_population
            fitness_scores = new_fitness_scores
            fitness_history.append(best_fitness)
            
            # Update best index for next generation
            best_idx = np.argmax(fitness_scores)
            
            # Log progress
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
            'algorithm_name': 'differential_evolution',
            'data_type': data_type,
            'gpu_accelerated': self.use_gpu,
            'detailed_metrics': detailed_metrics,
            'parameters': {
                'mutation_factor': self.mutation_factor,
                'crossover_probability': self.crossover_prob,
                'strategy': self.strategy,
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
                individual = self._create_zone_constrained_portfolio(
                    strategy_list, portfolio_size, zone_data
                )
            else:
                individual = list(np.random.choice(strategy_list, portfolio_size, replace=False))
            
            population.append(individual)
        
        return population
    
    def _mutate_best_1(self, population: List[List[Union[int, str]]], 
                       best: List[Union[int, str]], 
                       current_idx: int) -> List[Union[int, str]]:
        """DE/best/1 mutation strategy"""
        # Select two random individuals different from current
        indices = list(range(len(population)))
        indices.remove(current_idx)
        r1, r2 = random.sample(indices, 2)
        
        # Create mutant (in discrete space, we interpret this as influence)
        mutant = best.copy()
        
        # Apply mutation by swapping some elements based on mutation factor
        num_mutations = int(len(mutant) * self.mutation_factor)
        for _ in range(num_mutations):
            # Influenced by difference between r1 and r2
            if random.random() < 0.5:
                # Take element from r1
                pos = random.randint(0, len(mutant) - 1)
                if population[r1][pos] not in mutant:
                    # Find element to replace
                    replace_pos = random.randint(0, len(mutant) - 1)
                    mutant[replace_pos] = population[r1][pos]
        
        return mutant
    
    def _mutate_rand_1(self, population: List[List[Union[int, str]]], 
                       current_idx: int) -> List[Union[int, str]]:
        """DE/rand/1 mutation strategy"""
        # Select three random individuals different from current
        indices = list(range(len(population)))
        indices.remove(current_idx)
        r1, r2, r3 = random.sample(indices, 3)
        
        # Create mutant based on r1
        mutant = population[r1].copy()
        
        # Apply mutation influenced by difference between r2 and r3
        num_mutations = int(len(mutant) * self.mutation_factor)
        for _ in range(num_mutations):
            if random.random() < 0.5:
                # Take element from r2
                pos = random.randint(0, len(mutant) - 1)
                if population[r2][pos] not in mutant:
                    replace_pos = random.randint(0, len(mutant) - 1)
                    mutant[replace_pos] = population[r2][pos]
        
        return mutant
    
    def _crossover(self, target: List[Union[int, str]], mutant: List[Union[int, str]], 
                   num_strategies: int) -> List[Union[int, str]]:
        """Perform crossover between target and mutant"""
        trial = target.copy()
        
        # Ensure at least one position is from mutant
        force_pos = random.randint(0, len(trial) - 1)
        
        for i in range(len(trial)):
            if i == force_pos or random.random() < self.crossover_prob:
                # Take from mutant if not duplicate
                if mutant[i] not in trial or trial[i] == mutant[i]:
                    trial[i] = mutant[i]
                else:
                    # Find suitable replacement
                    available = [s for s in mutant if s not in trial]
                    if len(available) > 0:
                        trial[i] = random.choice(available)
        
        return trial
    
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
        
        # Ensure exact size
        if len(portfolio) > portfolio_size:
            portfolio = list(np.random.choice(portfolio, portfolio_size, replace=False))
        elif len(portfolio) < portfolio_size:
            remaining = portfolio_size - len(portfolio)
            available = [s for s in strategy_list if s not in portfolio]
            if len(available) > 0:
                additional = list(np.random.choice(available, 
                                                 min(remaining, len(available)), 
                                                 replace=False))
                portfolio.extend(additional)
        
        return portfolio