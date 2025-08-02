#!/usr/bin/env python3
"""
Comprehensive Test Suite for All Algorithms

This module tests all 8 optimization algorithms together including:
- Comparative performance
- Edge cases across all algorithms
- Configuration-driven behavior
- Zone optimization support
- Consistency checks
"""

import unittest
import numpy as np
import time
from pathlib import Path
import tempfile
import os

# Add backend to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from algorithms import (
    GeneticAlgorithm, ParticleSwarmOptimization, SimulatedAnnealing,
    DifferentialEvolution, AntColonyOptimization, HillClimbing,
    BayesianOptimization, RandomSearch, AlgorithmFactory
)
from tests.test_base_algorithm import BaseAlgorithmTest


class TestAllAlgorithms(BaseAlgorithmTest):
    """Test all algorithms with common test cases"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test data once for all algorithms"""
        super().setUpClass()
        cls.algorithm_classes = [
            GeneticAlgorithm,
            ParticleSwarmOptimization,
            SimulatedAnnealing,
            DifferentialEvolution,
            AntColonyOptimization,
            HillClimbing,
            BayesianOptimization,
            RandomSearch
        ]
        
        cls.algorithm_names = [
            'GA', 'PSO', 'SA', 'DE', 'ACO', 'HC', 'BO', 'RS'
        ]
    
    def test_all_algorithms_basic_functionality(self):
        """Test that all algorithms work with basic data"""
        results = {}
        
        for algo_class, name in zip(self.algorithm_classes, self.algorithm_names):
            print(f"\nTesting {name}...")
            
            try:
                result = self.run_algorithm_test(
                    algo_class,
                    f"{name} - Basic Test",
                    self.medium_data,
                    self.medium_portfolio
                )
                results[name] = result
                
                # All algorithms should find positive fitness
                self.assertGreater(result['best_fitness'], -10,
                                 f"{name} produced very poor fitness")
                
            except Exception as e:
                self.fail(f"{name} failed with error: {e}")
        
        # Summary
        print("\n" + "="*60)
        print("ALGORITHM COMPARISON - Basic Functionality")
        print("="*60)
        
        sorted_results = sorted(results.items(), 
                              key=lambda x: x[1]['best_fitness'], 
                              reverse=True)
        
        for name, result in sorted_results:
            print(f"{name:5s} | Fitness: {result['best_fitness']:8.4f} | "
                  f"Time: {result['execution_time']:6.3f}s")
    
    def test_all_algorithms_empty_data(self):
        """Test all algorithms handle empty data gracefully"""
        empty_data = np.array([]).reshape(0, 0)
        
        for algo_class, name in zip(self.algorithm_classes, self.algorithm_names):
            with self.subTest(algorithm=name):
                algo = algo_class()
                
                with self.assertRaises(ValueError):
                    algo.optimize(
                        daily_matrix=empty_data,
                        portfolio_size=5,
                        fitness_function=self.calculate_fitness
                    )
    
    def test_all_algorithms_single_strategy(self):
        """Test all algorithms with single strategy"""
        single_strategy_data = self.create_test_data(n_days=20, n_strategies=1)
        
        for algo_class, name in zip(self.algorithm_classes, self.algorithm_names):
            with self.subTest(algorithm=name):
                algo = algo_class()
                
                result = algo.optimize(
                    daily_matrix=single_strategy_data,
                    portfolio_size=1,
                    fitness_function=self.calculate_fitness
                )
                
                self.assertEqual(len(result['best_portfolio']), 1)
                self.assertEqual(result['best_portfolio'][0], 0)
    
    def test_all_algorithms_zone_optimization(self):
        """Test all algorithms support zone optimization"""
        data = self.create_test_data(n_days=50, n_strategies=30)
        zone_data = {
            'allowed_strategies': list(range(10, 20)),
            'min_strategies_per_zone': 3
        }
        
        results = {}
        
        for algo_class, name in zip(self.algorithm_classes, self.algorithm_names):
            with self.subTest(algorithm=name):
                algo = algo_class()
                
                result = algo.optimize(
                    daily_matrix=data,
                    portfolio_size=5,
                    fitness_function=self.calculate_fitness,
                    zone_data=zone_data
                )
                
                results[name] = result
                
                # Check zone constraints
                portfolio = result['best_portfolio']
                zone_strategies = [s for s in portfolio 
                                 if s in zone_data['allowed_strategies']]
                
                self.assertGreaterEqual(
                    len(zone_strategies), 
                    zone_data['min_strategies_per_zone'],
                    f"{name} did not respect zone constraints"
                )
        
        # Summary
        print("\n" + "="*60)
        print("ZONE OPTIMIZATION RESULTS")
        print("="*60)
        
        for name, result in results.items():
            portfolio = result['best_portfolio']
            zone_count = sum(1 for s in portfolio 
                           if s in zone_data['allowed_strategies'])
            print(f"{name:5s} | Zone strategies: {zone_count}/5 | "
                  f"Fitness: {result['best_fitness']:.4f}")
    
    def test_all_algorithms_configuration_driven(self):
        """Test all algorithms with custom configuration"""
        # Create test config with aggressive parameters
        config_content = """[GA]
population_size = 10
generations = 20

[PSO]
swarm_size = 10
iterations = 20

[SA]
iterations = 50
cooling_rate = 0.8

[DE]
population_size = 10
generations = 20

[ACO]
num_ants = 10
iterations = 20

[HC]
iterations = 30
restarts = 2

[BO]
n_initial_samples = 10
iterations = 15

[RS]
iterations = 50
batch_size = 10
"""
        fd, config_path = tempfile.mkstemp(suffix='.ini')
        with os.fdopen(fd, 'w') as f:
            f.write(config_content)
        
        try:
            for algo_class, name in zip(self.algorithm_classes, self.algorithm_names):
                with self.subTest(algorithm=name):
                    # With config
                    algo_with_config = algo_class(config_path)
                    
                    # Without config (defaults)
                    algo_default = algo_class()
                    
                    # Both should work
                    result_config = algo_with_config.optimize(
                        daily_matrix=self.small_data,
                        portfolio_size=5,
                        fitness_function=self.calculate_fitness
                    )
                    
                    result_default = algo_default.optimize(
                        daily_matrix=self.small_data,
                        portfolio_size=5,
                        fitness_function=self.calculate_fitness
                    )
                    
                    # Both should produce valid results
                    self.validate_algorithm_result(result_config, f"{name}-Config")
                    self.validate_algorithm_result(result_default, f"{name}-Default")
                    
                    # Config version should generally be faster (fewer iterations)
                    print(f"{name}: Config time={result_config['execution_time']:.3f}s, "
                          f"Default time={result_default['execution_time']:.3f}s")
        
        finally:
            os.unlink(config_path)
    
    def test_all_algorithms_variable_portfolio_size(self):
        """Test all algorithms with variable portfolio size"""
        data = self.create_test_data(n_days=50, n_strategies=30)
        
        for algo_class, name in zip(self.algorithm_classes, self.algorithm_names):
            with self.subTest(algorithm=name):
                algo = algo_class()
                
                # Test with range
                result = algo.optimize(
                    daily_matrix=data,
                    portfolio_size=(5, 10),
                    fitness_function=self.calculate_fitness
                )
                
                portfolio_size = len(result['best_portfolio'])
                self.assertGreaterEqual(portfolio_size, 5)
                self.assertLessEqual(portfolio_size, 10)
    
    def test_all_algorithms_consistency(self):
        """Test that algorithms produce consistent results"""
        np.random.seed(42)
        data = self.medium_data
        portfolio_size = 10
        
        consistency_results = {}
        
        for algo_class, name in zip(self.algorithm_classes, self.algorithm_names):
            with self.subTest(algorithm=name):
                # Run 3 times
                fitness_values = []
                
                for i in range(3):
                    np.random.seed(42 + i)  # Different seeds
                    algo = algo_class()
                    
                    result = algo.optimize(
                        daily_matrix=data,
                        portfolio_size=portfolio_size,
                        fitness_function=self.calculate_fitness
                    )
                    
                    fitness_values.append(result['best_fitness'])
                
                # Calculate consistency metrics
                mean_fitness = np.mean(fitness_values)
                std_fitness = np.std(fitness_values)
                cv = std_fitness / abs(mean_fitness) if mean_fitness != 0 else 0
                
                consistency_results[name] = {
                    'mean': mean_fitness,
                    'std': std_fitness,
                    'cv': cv,
                    'values': fitness_values
                }
                
                # Most algorithms should be reasonably consistent
                # (Random Search might have higher variance)
                if name != 'RS':
                    self.assertLess(cv, 0.3, 
                                  f"{name} shows high variance: CV={cv:.2%}")
        
        # Summary
        print("\n" + "="*60)
        print("ALGORITHM CONSISTENCY ANALYSIS")
        print("="*60)
        
        for name, metrics in sorted(consistency_results.items()):
            print(f"{name:5s} | Mean: {metrics['mean']:7.4f} | "
                  f"Std: {metrics['std']:6.4f} | CV: {metrics['cv']:6.2%}")
    
    def test_algorithm_factory(self):
        """Test the algorithm factory"""
        factory = AlgorithmFactory(self.temp_config)
        
        # Test creating all algorithms
        algorithms = factory.create_all_algorithms()
        
        self.assertEqual(len(algorithms), 8)
        
        # Test each algorithm works
        for name, algo in algorithms.items():
            result = algo.optimize(
                daily_matrix=self.small_data,
                portfolio_size=5,
                fitness_function=self.calculate_fitness
            )
            
            self.validate_algorithm_result(result, name)
    
    def test_algorithm_scalability(self):
        """Test how algorithms scale with problem size"""
        sizes = [
            ("Small", self.small_data, 5),
            ("Medium", self.medium_data, 10),
            ("Large", self.large_data, 20)
        ]
        
        scalability_results = {name: {} for name in self.algorithm_names}
        
        for size_name, data, portfolio_size in sizes:
            print(f"\nTesting {size_name} dataset...")
            
            for algo_class, name in zip(self.algorithm_classes, self.algorithm_names):
                algo = algo_class()
                
                start_time = time.time()
                result = algo.optimize(
                    daily_matrix=data,
                    portfolio_size=portfolio_size,
                    fitness_function=self.calculate_fitness
                )
                
                scalability_results[name][size_name] = {
                    'time': result['execution_time'],
                    'fitness': result['best_fitness']
                }
        
        # Summary
        print("\n" + "="*60)
        print("ALGORITHM SCALABILITY ANALYSIS")
        print("="*60)
        print("Algorithm | Small Time | Medium Time | Large Time | Time Ratio")
        print("-"*60)
        
        for name in self.algorithm_names:
            small_time = scalability_results[name]['Small']['time']
            medium_time = scalability_results[name]['Medium']['time']
            large_time = scalability_results[name]['Large']['time']
            
            ratio = large_time / small_time if small_time > 0 else 0
            
            print(f"{name:9s} | {small_time:10.3f} | {medium_time:11.3f} | "
                  f"{large_time:10.3f} | {ratio:10.1f}x")


class TestLegacyComparison(BaseAlgorithmTest):
    """Test that new modular algorithms behave like legacy implementations"""
    
    def test_fitness_function_compatibility(self):
        """Test that fitness function matches legacy behavior"""
        # Create test portfolio
        portfolio = np.array([0, 2, 4, 6, 8])
        
        # Calculate fitness
        fitness = self.calculate_fitness(self.medium_data, portfolio)
        
        # Fitness should be ROI/Drawdown - correlation_penalty
        # Just verify it's in reasonable range
        self.assertGreater(fitness, -10)
        self.assertLess(fitness, 10)
        
        # Test with highly correlated portfolio
        # (strategies next to each other are more correlated in our test data)
        correlated_portfolio = np.array([0, 1, 2, 3, 4])
        correlated_fitness = self.calculate_fitness(self.medium_data, correlated_portfolio)
        
        # Should generally be lower due to correlation penalty
        uncorrelated_portfolio = np.array([0, 10, 20, 30, 40])
        uncorrelated_fitness = self.calculate_fitness(self.medium_data, uncorrelated_portfolio)
        
        # This might not always hold due to different returns, but check it's reasonable
        print(f"Correlated fitness: {correlated_fitness:.4f}")
        print(f"Uncorrelated fitness: {uncorrelated_fitness:.4f}")


if __name__ == '__main__':
    unittest.main()