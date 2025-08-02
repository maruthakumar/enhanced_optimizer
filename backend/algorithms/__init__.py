"""
Modular Algorithm Suite for Heavy Optimizer Platform

This module provides all 8 optimization algorithms as independent,
configurable modules following the modular architecture requirements.
"""

from .base_algorithm import BaseOptimizationAlgorithm
from .genetic_algorithm import GeneticAlgorithm
from .particle_swarm_optimization import ParticleSwarmOptimization
from .simulated_annealing import SimulatedAnnealing
from .differential_evolution import DifferentialEvolution
from .ant_colony_optimization import AntColonyOptimization
from .hill_climbing import HillClimbing
from .bayesian_optimization import BayesianOptimization
from .random_search import RandomSearch
from .algorithm_factory import AlgorithmFactory, create_algorithm, create_all_algorithms

__all__ = [
    'BaseOptimizationAlgorithm',
    'GeneticAlgorithm',
    'ParticleSwarmOptimization',
    'SimulatedAnnealing',
    'DifferentialEvolution',
    'AntColonyOptimization',
    'HillClimbing',
    'BayesianOptimization',
    'RandomSearch',
    'AlgorithmFactory',
    'create_algorithm',
    'create_all_algorithms'
]

# Algorithm registry for production system
ALGORITHM_REGISTRY = {
    'GA': GeneticAlgorithm,
    'PSO': ParticleSwarmOptimization,
    'SA': SimulatedAnnealing,
    'DE': DifferentialEvolution,
    'ACO': AntColonyOptimization,
    'HC': HillClimbing,
    'BO': BayesianOptimization,
    'RS': RandomSearch
}

def get_all_algorithms():
    """Get all 8 algorithms for complete production system"""
    return ALGORITHM_REGISTRY

def verify_algorithm_completeness():
    """Verify all 8 algorithms are present"""
    return AlgorithmFactory.verify_all_algorithms()