#!/usr/bin/env python3
"""
Algorithm Factory for Heavy Optimizer Platform

This factory provides centralized algorithm instantiation with configuration support,
enabling the modular architecture requirements.
"""

import logging
from typing import Dict, Optional, Union, Type, Tuple
from pathlib import Path

from .base_algorithm import BaseOptimizationAlgorithm
from .genetic_algorithm import GeneticAlgorithm
from .particle_swarm_optimization import ParticleSwarmOptimization
from .simulated_annealing import SimulatedAnnealing
from .differential_evolution import DifferentialEvolution
from .ant_colony_optimization import AntColonyOptimization
from .hill_climbing import HillClimbing
from .bayesian_optimization import BayesianOptimization
from .random_search import RandomSearch


class AlgorithmFactory:
    """Factory for creating optimization algorithm instances"""
    
    # Algorithm registry mapping names to classes
    ALGORITHM_REGISTRY = {
        # Full names
        'genetic_algorithm': GeneticAlgorithm,
        'particle_swarm_optimization': ParticleSwarmOptimization,
        'simulated_annealing': SimulatedAnnealing,
        'differential_evolution': DifferentialEvolution,
        'ant_colony_optimization': AntColonyOptimization,
        'hill_climbing': HillClimbing,
        'bayesian_optimization': BayesianOptimization,
        'random_search': RandomSearch,
        
        # Short names
        'ga': GeneticAlgorithm,
        'pso': ParticleSwarmOptimization,
        'sa': SimulatedAnnealing,
        'de': DifferentialEvolution,
        'aco': AntColonyOptimization,
        'hc': HillClimbing,
        'bo': BayesianOptimization,
        'rs': RandomSearch,
        
        # Uppercase variants
        'GA': GeneticAlgorithm,
        'PSO': ParticleSwarmOptimization,
        'SA': SimulatedAnnealing,
        'DE': DifferentialEvolution,
        'ACO': AntColonyOptimization,
        'HC': HillClimbing,
        'BO': BayesianOptimization,
        'RS': RandomSearch
    }
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize factory with optional configuration file
        
        Args:
            config_path: Path to .ini configuration file
        """
        self.config_path = config_path
        self.logger = logging.getLogger(__name__)
        
        # Validate config file if provided
        if config_path and not Path(config_path).exists():
            self.logger.warning(f"Configuration file not found: {config_path}")
            self.config_path = None
    
    def create_algorithm(self, algorithm_name: str) -> BaseOptimizationAlgorithm:
        """
        Create an algorithm instance
        
        Args:
            algorithm_name: Name of the algorithm (case-insensitive)
            
        Returns:
            Algorithm instance
            
        Raises:
            ValueError: If algorithm name is not recognized
        """
        # Normalize algorithm name
        normalized_name = algorithm_name.lower().replace('-', '_').replace(' ', '_')
        
        if normalized_name not in self.ALGORITHM_REGISTRY:
            available = self.get_available_algorithms()
            raise ValueError(
                f"Unknown algorithm: {algorithm_name}. "
                f"Available algorithms: {', '.join(available)}"
            )
        
        # Get algorithm class
        algorithm_class = self.ALGORITHM_REGISTRY[normalized_name]
        
        # Create instance with configuration
        instance = algorithm_class(self.config_path)
        
        self.logger.info(f"Created {algorithm_class.__name__} instance")
        return instance
    
    def create_all_algorithms(self) -> Dict[str, BaseOptimizationAlgorithm]:
        """
        Create instances of all available algorithms
        
        Returns:
            Dictionary mapping algorithm names to instances
        """
        algorithms = {}
        
        # Use canonical names only
        canonical_names = [
            'genetic_algorithm',
            'particle_swarm_optimization',
            'simulated_annealing',
            'differential_evolution',
            'ant_colony_optimization',
            'hill_climbing',
            'bayesian_optimization',
            'random_search'
        ]
        
        for name in canonical_names:
            try:
                algorithms[name] = self.create_algorithm(name)
                self.logger.info(f"Created {name}")
            except Exception as e:
                self.logger.error(f"Failed to create {name}: {e}")
        
        return algorithms
    
    def get_available_algorithms(self) -> list:
        """Get list of available algorithm names"""
        # Return canonical names
        return [
            'genetic_algorithm (GA)',
            'particle_swarm_optimization (PSO)',
            'simulated_annealing (SA)',
            'differential_evolution (DE)',
            'ant_colony_optimization (ACO)',
            'hill_climbing (HC)',
            'bayesian_optimization (BO)',
            'random_search (RS)'
        ]
    
    def get_algorithm_info(self, algorithm_name: str) -> Dict:
        """
        Get information about an algorithm
        
        Args:
            algorithm_name: Name of the algorithm
            
        Returns:
            Dictionary with algorithm information
        """
        try:
            algorithm = self.create_algorithm(algorithm_name)
            return algorithm.get_algorithm_info()
        except ValueError as e:
            return {'error': str(e)}
    
    @classmethod
    def verify_all_algorithms(cls) -> Tuple[bool, str]:
        """
        Verify all 8 algorithms are available
        
        Returns:
            Tuple of (success, message)
        """
        expected = {'ga', 'pso', 'sa', 'de', 'aco', 'hc', 'bo', 'rs'}
        available = set()
        
        for name in cls.ALGORITHM_REGISTRY:
            if len(name) <= 3:  # Short names only
                available.add(name.lower())
        
        if expected == available:
            return True, f"✅ All 8 algorithms available: {sorted(available)}"
        else:
            missing = expected - available
            extra = available - expected
            msg = ""
            if missing:
                msg += f"❌ Missing algorithms: {missing} "
            if extra:
                msg += f"Extra algorithms: {extra}"
            return False, msg


def create_algorithm(algorithm_name: str, 
                    config_path: Optional[str] = None) -> BaseOptimizationAlgorithm:
    """
    Convenience function to create an algorithm instance
    
    Args:
        algorithm_name: Name of the algorithm
        config_path: Optional path to configuration file
        
    Returns:
        Algorithm instance
    """
    factory = AlgorithmFactory(config_path)
    return factory.create_algorithm(algorithm_name)


def create_all_algorithms(config_path: Optional[str] = None) -> Dict[str, BaseOptimizationAlgorithm]:
    """
    Convenience function to create all algorithm instances
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        Dictionary mapping algorithm names to instances
    """
    factory = AlgorithmFactory(config_path)
    return factory.create_all_algorithms()