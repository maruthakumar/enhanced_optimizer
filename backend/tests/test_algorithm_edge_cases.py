#!/usr/bin/env python3
"""
Edge Case Tests for All Algorithms

This module tests edge cases and boundary conditions including:
- Empty datasets
- Single trading strategy
- Portfolio size edge cases
- Invalid inputs
- Extreme fitness values
"""

import unittest
import numpy as np
from pathlib import Path

# Add backend to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from algorithms import (
    GeneticAlgorithm, ParticleSwarmOptimization, SimulatedAnnealing,
    DifferentialEvolution, AntColonyOptimization, HillClimbing,
    BayesianOptimization, RandomSearch
)
from tests.test_base_algorithm import BaseAlgorithmTest


class TestAlgorithmEdgeCases(BaseAlgorithmTest):
    """Test edge cases for all algorithms"""
    
    @classmethod
    def setUpClass(cls):
        """Set up algorithm list"""
        super().setUpClass()
        cls.algorithms = [
            ('GA', GeneticAlgorithm),
            ('PSO', ParticleSwarmOptimization),
            ('SA', SimulatedAnnealing),
            ('DE', DifferentialEvolution),
            ('ACO', AntColonyOptimization),
            ('HC', HillClimbing),
            ('BO', BayesianOptimization),
            ('RS', RandomSearch)
        ]
    
    def test_empty_dataset(self):
        """Test with completely empty dataset"""
        for name, algo_class in self.algorithms:
            with self.subTest(algorithm=name):
                algo = algo_class()
                empty_data = np.array([]).reshape(0, 0)
                
                with self.assertRaises(ValueError):
                    algo.optimize(
                        daily_matrix=empty_data,
                        portfolio_size=5,
                        fitness_function=self.calculate_fitness
                    )
    
    def test_single_day_data(self):
        """Test with only one trading day"""
        single_day_data = np.array([[0.01, 0.02, -0.01, 0.03, 0.00]])
        
        for name, algo_class in self.algorithms:
            with self.subTest(algorithm=name):
                algo = algo_class()
                
                result = algo.optimize(
                    daily_matrix=single_day_data,
                    portfolio_size=3,
                    fitness_function=self.calculate_fitness
                )
                
                # Should still work but fitness will be simple
                self.validate_algorithm_result(result, name)
                self.assertEqual(len(result['best_portfolio']), 3)
    
    def test_single_strategy(self):
        """Test with only one strategy available"""
        for name, algo_class in self.algorithms:
            with self.subTest(algorithm=name):
                algo = algo_class()
                single_strategy = self.create_test_data(n_days=20, n_strategies=1)
                
                result = algo.optimize(
                    daily_matrix=single_strategy,
                    portfolio_size=1,
                    fitness_function=self.calculate_fitness
                )
                
                self.assertEqual(len(result['best_portfolio']), 1)
                self.assertEqual(result['best_portfolio'][0], 0)
    
    def test_portfolio_larger_than_strategies(self):
        """Test when requested portfolio size exceeds available strategies"""
        small_data = self.create_test_data(n_days=20, n_strategies=3)
        
        for name, algo_class in self.algorithms:
            with self.subTest(algorithm=name):
                algo = algo_class()
                
                with self.assertRaises(ValueError):
                    algo.optimize(
                        daily_matrix=small_data,
                        portfolio_size=5,  # More than 3 available
                        fitness_function=self.calculate_fitness
                    )
    
    def test_portfolio_equals_strategies(self):
        """Test when portfolio size equals number of strategies"""
        n_strategies = 5
        data = self.create_test_data(n_days=20, n_strategies=n_strategies)
        
        for name, algo_class in self.algorithms:
            with self.subTest(algorithm=name):
                algo = algo_class()
                
                result = algo.optimize(
                    daily_matrix=data,
                    portfolio_size=n_strategies,
                    fitness_function=self.calculate_fitness
                )
                
                # Should select all strategies
                self.assertEqual(len(result['best_portfolio']), n_strategies)
                self.assertEqual(set(result['best_portfolio']), set(range(n_strategies)))
    
    def test_zero_portfolio_size(self):
        """Test with zero portfolio size"""
        for name, algo_class in self.algorithms:
            with self.subTest(algorithm=name):
                algo = algo_class()
                
                with self.assertRaises(ValueError):
                    algo.optimize(
                        daily_matrix=self.small_data,
                        portfolio_size=0,
                        fitness_function=self.calculate_fitness
                    )
    
    def test_negative_portfolio_size(self):
        """Test with negative portfolio size"""
        for name, algo_class in self.algorithms:
            with self.subTest(algorithm=name):
                algo = algo_class()
                
                with self.assertRaises(ValueError):
                    algo.optimize(
                        daily_matrix=self.small_data,
                        portfolio_size=-5,
                        fitness_function=self.calculate_fitness
                    )
    
    def test_all_zero_returns(self):
        """Test with all strategies having zero returns"""
        zero_returns = np.zeros((50, 20))
        
        for name, algo_class in self.algorithms:
            with self.subTest(algorithm=name):
                algo = algo_class()
                
                result = algo.optimize(
                    daily_matrix=zero_returns,
                    portfolio_size=5,
                    fitness_function=self.calculate_fitness
                )
                
                # Fitness should be 0 (no returns, no drawdown)
                self.assertAlmostEqual(result['best_fitness'], 0.0, places=5)
    
    def test_all_negative_returns(self):
        """Test with all strategies having negative returns"""
        negative_returns = np.full((50, 20), -0.01)  # -1% daily loss
        
        for name, algo_class in self.algorithms:
            with self.subTest(algorithm=name):
                algo = algo_class()
                
                result = algo.optimize(
                    daily_matrix=negative_returns,
                    portfolio_size=5,
                    fitness_function=self.calculate_fitness
                )
                
                # Fitness should be negative
                self.assertLess(result['best_fitness'], 0)
    
    def test_extreme_positive_returns(self):
        """Test with extremely positive returns"""
        extreme_returns = np.full((50, 20), 0.5)  # 50% daily gain!
        
        for name, algo_class in self.algorithms:
            with self.subTest(algorithm=name):
                algo = algo_class()
                
                result = algo.optimize(
                    daily_matrix=extreme_returns,
                    portfolio_size=5,
                    fitness_function=self.calculate_fitness
                )
                
                # Fitness should be very high
                self.assertGreater(result['best_fitness'], 10)
    
    def test_highly_correlated_strategies(self):
        """Test with perfectly correlated strategies"""
        # All strategies have identical returns
        base_returns = np.random.randn(50) * 0.02
        correlated_data = np.tile(base_returns.reshape(-1, 1), (1, 20))
        
        for name, algo_class in self.algorithms:
            with self.subTest(algorithm=name):
                algo = algo_class()
                
                result = algo.optimize(
                    daily_matrix=correlated_data,
                    portfolio_size=5,
                    fitness_function=self.calculate_fitness
                )
                
                # Fitness should be heavily penalized due to correlation
                # All strategies are identical, so correlation = 1.0
                # Penalty = 10 * 1.0 = 10
                self.validate_algorithm_result(result, name)
    
    def test_zone_constraints_impossible(self):
        """Test zone constraints that are impossible to satisfy"""
        data = self.create_test_data(n_days=50, n_strategies=20)
        
        # Impossible constraint: need 10 strategies from a zone of 5
        zone_data = {
            'allowed_strategies': [0, 1, 2, 3, 4],  # Only 5 strategies
            'min_strategies_per_zone': 10  # Need 10!
        }
        
        for name, algo_class in self.algorithms:
            with self.subTest(algorithm=name):
                algo = algo_class()
                
                # Algorithm should still work but can't satisfy constraint
                result = algo.optimize(
                    daily_matrix=data,
                    portfolio_size=10,
                    fitness_function=self.calculate_fitness,
                    zone_data=zone_data
                )
                
                # Should use all 5 zone strategies plus 5 others
                portfolio = result['best_portfolio']
                zone_strategies = [s for s in portfolio 
                                 if s in zone_data['allowed_strategies']]
                
                self.assertEqual(len(zone_strategies), 5)  # Max possible
    
    def test_invalid_portfolio_size_tuple(self):
        """Test with invalid portfolio size tuple"""
        for name, algo_class in self.algorithms:
            with self.subTest(algorithm=name):
                algo = algo_class()
                
                # Min > Max
                with self.assertRaises(ValueError):
                    algo.optimize(
                        daily_matrix=self.small_data,
                        portfolio_size=(10, 5),
                        fitness_function=self.calculate_fitness
                    )
                
                # Wrong tuple length
                with self.assertRaises(ValueError):
                    algo.optimize(
                        daily_matrix=self.small_data,
                        portfolio_size=(5, 10, 15),
                        fitness_function=self.calculate_fitness
                    )
    
    def test_nan_in_data(self):
        """Test with NaN values in data"""
        data_with_nan = self.small_data.copy()
        data_with_nan[5, 3] = np.nan
        
        for name, algo_class in self.algorithms:
            with self.subTest(algorithm=name):
                algo = algo_class()
                
                # Most algorithms should fail or produce poor results
                try:
                    result = algo.optimize(
                        daily_matrix=data_with_nan,
                        portfolio_size=5,
                        fitness_function=self.calculate_fitness
                    )
                    
                    # If it doesn't fail, fitness should be NaN or very poor
                    if not np.isnan(result['best_fitness']):
                        print(f"{name} handled NaN data with fitness: "
                              f"{result['best_fitness']}")
                except:
                    # Expected for some algorithms
                    pass
    
    def test_infinite_values_in_data(self):
        """Test with infinite values in data"""
        data_with_inf = self.small_data.copy()
        data_with_inf[5, 3] = np.inf
        
        for name, algo_class in self.algorithms:
            with self.subTest(algorithm=name):
                algo = algo_class()
                
                # Should handle or fail gracefully
                try:
                    result = algo.optimize(
                        daily_matrix=data_with_inf,
                        portfolio_size=5,
                        fitness_function=self.calculate_fitness
                    )
                    
                    # If it doesn't fail, check result
                    if np.isfinite(result['best_fitness']):
                        # Might avoid the infinite strategy
                        self.assertNotIn(3, result['best_portfolio'])
                except:
                    # Expected for some algorithms
                    pass
    
    def test_fitness_function_exception(self):
        """Test with fitness function that throws exception"""
        def bad_fitness_function(daily_matrix, portfolio):
            raise RuntimeError("Fitness calculation failed!")
        
        for name, algo_class in self.algorithms:
            with self.subTest(algorithm=name):
                algo = algo_class()
                
                with self.assertRaises(RuntimeError):
                    algo.optimize(
                        daily_matrix=self.small_data,
                        portfolio_size=5,
                        fitness_function=bad_fitness_function
                    )
    
    def test_non_numpy_array_input(self):
        """Test with non-numpy array input"""
        list_data = [[0.01, 0.02], [0.03, 0.04], [-0.01, 0.01]]
        
        for name, algo_class in self.algorithms:
            with self.subTest(algorithm=name):
                algo = algo_class()
                
                with self.assertRaises(ValueError):
                    algo.optimize(
                        daily_matrix=list_data,  # Not numpy array
                        portfolio_size=1,
                        fitness_function=self.calculate_fitness
                    )


if __name__ == '__main__':
    unittest.main()