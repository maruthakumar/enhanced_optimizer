#!/usr/bin/env python3
"""
Workflow Adapter for Heavy Optimizer Platform

Provides backward compatibility for existing workflows to use the new modular algorithms.
"""

import numpy as np
import time
from typing import Dict, Any, Optional, Callable
import logging
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithms import AlgorithmFactory
from config.config_manager import get_config_manager


class WorkflowAlgorithmAdapter:
    """Adapter to use modular algorithms in existing workflows"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize adapter with configuration
        
        Args:
            config_path: Path to configuration file
        """
        self.logger = logging.getLogger(__name__)
        self.config_manager = get_config_manager()
        self.algorithm_factory = AlgorithmFactory(config_path)
        
        # Cache for algorithm instances
        self._algorithm_cache = {}
        
        # Get zone configuration
        self.zone_config = self.config_manager.get_zone_config()
        
        self.logger.info("Initialized workflow algorithm adapter")
    
    def run_genetic_algorithm(self, daily_matrix: np.ndarray, 
                            portfolio_size: int) -> Dict:
        """Run Genetic Algorithm (backward compatible interface)"""
        return self._run_algorithm('ga', daily_matrix, portfolio_size)
    
    def run_particle_swarm_optimization(self, daily_matrix: np.ndarray,
                                      portfolio_size: int) -> Dict:
        """Run Particle Swarm Optimization (backward compatible interface)"""
        return self._run_algorithm('pso', daily_matrix, portfolio_size)
    
    def run_simulated_annealing(self, daily_matrix: np.ndarray,
                               portfolio_size: int) -> Dict:
        """Run Simulated Annealing (backward compatible interface)"""
        return self._run_algorithm('sa', daily_matrix, portfolio_size)
    
    def run_differential_evolution(self, daily_matrix: np.ndarray,
                                 portfolio_size: int) -> Dict:
        """Run Differential Evolution (backward compatible interface)"""
        return self._run_algorithm('de', daily_matrix, portfolio_size)
    
    def run_ant_colony_optimization(self, daily_matrix: np.ndarray,
                                  portfolio_size: int) -> Dict:
        """Run Ant Colony Optimization (backward compatible interface)"""
        return self._run_algorithm('aco', daily_matrix, portfolio_size)
    
    def run_hill_climbing(self, daily_matrix: np.ndarray,
                         portfolio_size: int) -> Dict:
        """Run Hill Climbing (backward compatible interface)"""
        return self._run_algorithm('hc', daily_matrix, portfolio_size)
    
    def run_bayesian_optimization(self, daily_matrix: np.ndarray,
                                portfolio_size: int) -> Dict:
        """Run Bayesian Optimization (backward compatible interface)"""
        return self._run_algorithm('bo', daily_matrix, portfolio_size)
    
    def run_random_search(self, daily_matrix: np.ndarray,
                         portfolio_size: int) -> Dict:
        """Run Random Search (backward compatible interface)"""
        return self._run_algorithm('rs', daily_matrix, portfolio_size)
    
    def _run_algorithm(self, algorithm_name: str, daily_matrix: np.ndarray,
                      portfolio_size: int) -> Dict:
        """
        Run specified algorithm using modular implementation
        
        Args:
            algorithm_name: Short name of algorithm (e.g., 'ga', 'pso')
            daily_matrix: Daily returns matrix
            portfolio_size: Target portfolio size
            
        Returns:
            Dictionary with optimization results
        """
        try:
            # Get or create algorithm instance
            if algorithm_name not in self._algorithm_cache:
                self._algorithm_cache[algorithm_name] = self.algorithm_factory.create_algorithm(algorithm_name)
            
            algorithm = self._algorithm_cache[algorithm_name]
            
            # Create fitness function
            fitness_function = self._create_fitness_function()
            
            # Get zone data if enabled
            zone_data = self.zone_config if self.zone_config['enabled'] else None
            
            # Run algorithm
            result = algorithm.optimize(
                daily_matrix=daily_matrix,
                portfolio_size=portfolio_size,
                fitness_function=fitness_function,
                zone_data=zone_data
            )
            
            # Ensure backward compatibility of result format
            if 'execution_time' not in result:
                result['execution_time'] = 0.0
            if 'algorithm' not in result:
                result['algorithm'] = algorithm_name
            
            return result
            
        except Exception as e:
            self.logger.error(f"Algorithm {algorithm_name} failed: {e}")
            # Return failure result for backward compatibility
            return {
                'best_fitness': -float('inf'),
                'best_portfolio': [],
                'execution_time': 0.0,
                'algorithm': algorithm_name,
                'error': str(e)
            }
    
    def _create_fitness_function(self) -> Callable:
        """Create fitness function for algorithms"""
        # Get correlation configuration
        correlation_penalty = self.config_manager.getfloat('CORRELATION', 'penalty_weight', 0.1)
        
        def fitness_function(data: np.ndarray, portfolio: np.ndarray) -> float:
            """Calculate fitness for a portfolio"""
            # Extract portfolio returns
            portfolio_returns = data[:, portfolio].mean(axis=1)
            
            # Calculate cumulative returns
            cumulative_returns = np.cumsum(portfolio_returns)
            
            # Calculate ROI
            roi = cumulative_returns[-1]
            
            # Calculate maximum drawdown
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdowns = running_max - cumulative_returns
            max_drawdown = np.max(drawdowns) if np.max(drawdowns) > 0 else 1e-6
            
            # Base fitness (ROI/Drawdown ratio)
            base_fitness = roi / max_drawdown if max_drawdown > 1e-6 else roi * 100
            
            # Apply correlation penalty if configured
            penalty = 0
            if correlation_penalty > 0 and len(portfolio) > 1:
                # Simple correlation penalty (full implementation would use correlation matrix)
                penalty = correlation_penalty * 0.5  # Placeholder
            
            return base_fitness - penalty
        
        return fitness_function
    
    def run_all_algorithms(self, daily_matrix: np.ndarray, 
                          portfolio_size: int) -> Dict[str, Dict]:
        """
        Run all algorithms and return results
        
        Args:
            daily_matrix: Daily returns matrix
            portfolio_size: Target portfolio size
            
        Returns:
            Dictionary mapping algorithm names to results
        """
        algorithms = ['ga', 'pso', 'sa', 'de', 'aco', 'hc', 'bo', 'rs']
        results = {}
        
        for algo in algorithms:
            self.logger.info(f"Running {algo.upper()}...")
            results[algo] = self._run_algorithm(algo, daily_matrix, portfolio_size)
        
        return results
    
    def get_best_algorithm(self, results: Dict[str, Dict]) -> str:
        """
        Determine best algorithm from results
        
        Args:
            results: Dictionary of algorithm results
            
        Returns:
            Name of best algorithm
        """
        best_algo = None
        best_fitness = -float('inf')
        
        for algo, result in results.items():
            if result.get('best_fitness', -float('inf')) > best_fitness:
                best_fitness = result['best_fitness']
                best_algo = algo
        
        return best_algo


def create_workflow_adapter(config_path: Optional[str] = None) -> WorkflowAlgorithmAdapter:
    """
    Create workflow adapter instance
    
    Args:
        config_path: Optional configuration file path
        
    Returns:
        WorkflowAlgorithmAdapter instance
    """
    return WorkflowAlgorithmAdapter(config_path)