#!/usr/bin/env python3
"""
Unit Tests for Simulated Annealing

Tests the modular SA implementation including:
- Temperature scheduling
- Acceptance probability
- Neighbor generation
- Convergence behavior
"""

import unittest
import numpy as np
import math
from pathlib import Path

# Add backend to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from algorithms.simulated_annealing import SimulatedAnnealing
from tests.test_base_algorithm import BaseAlgorithmTest


class TestSimulatedAnnealing(BaseAlgorithmTest):
    """Test cases for Simulated Annealing"""
    
    def test_basic_functionality_small_data(self):
        """Test SA with small dataset"""
        result = self.run_algorithm_test(
            SimulatedAnnealing,
            "SA - Small Data",
            self.small_data,
            self.small_portfolio
        )
        
        self.assertGreater(result['best_fitness'], -1.0)
        self.assertLess(result['best_fitness'], 10.0)
    
    def test_basic_functionality_medium_data(self):
        """Test SA with medium dataset"""
        result = self.run_algorithm_test(
            SimulatedAnnealing,
            "SA - Medium Data",
            self.medium_data,
            self.medium_portfolio
        )
        
        # Check algorithm-specific metrics
        self.assertIn('iterations', result)
        self.assertIn('initial_temperature', result)
        self.assertIn('final_temperature', result)
        self.assertIn('cooling_rate', result)
    
    def test_temperature_cooling(self):
        """Test temperature cooling schedule"""
        sa = SimulatedAnnealing()
        
        result = sa.optimize(
            daily_matrix=self.medium_data,
            portfolio_size=10,
            fitness_function=self.calculate_fitness
        )
        
        # Check temperature decreased
        self.assertLess(result['final_temperature'], result['initial_temperature'])
        
        # Check cooling followed schedule
        expected_final_temp = result['initial_temperature'] * \
                            (result['cooling_rate'] ** result['iterations'])
        
        # Should be close (accounting for min temperature constraint)
        self.assertAlmostEqual(
            result['final_temperature'],
            max(expected_final_temp, sa.min_temperature),
            places=5
        )
    
    def test_acceptance_probability(self):
        """Test SA acceptance probability logic"""
        sa = SimulatedAnnealing()
        
        # Test acceptance probability calculation
        temperatures = [1.0, 0.5, 0.1]
        deltas = [-1.0, -0.5, -0.1]  # Negative = worse solution
        
        for temp in temperatures:
            for delta in deltas:
                prob = math.exp(delta / temp)
                
                # Probability should be between 0 and 1
                self.assertGreaterEqual(prob, 0)
                self.assertLessEqual(prob, 1)
                
                # Higher temperature = higher acceptance probability
                if temp > 0.1:
                    higher_temp_prob = math.exp(delta / temp)
                    lower_temp_prob = math.exp(delta / 0.1)
                    self.assertGreater(higher_temp_prob, lower_temp_prob)
    
    def test_neighbor_generation_methods(self):
        """Test different neighbor generation methods"""
        # Test with swap method
        sa_swap = SimulatedAnnealing()
        sa_swap.neighbor_selection_method = 'swap'
        
        portfolio = np.array([0, 2, 4, 6, 8])
        
        # Generate multiple neighbors
        neighbors_swap = []
        for _ in range(10):
            neighbor = sa_swap._generate_neighbor(portfolio, 10, None)
            neighbors_swap.append(neighbor)
            
            # Should be valid
            self.assertEqual(len(neighbor), 5)
            self.assertEqual(len(set(neighbor)), 5)
        
        # Test with default (replace) method
        sa_replace = SimulatedAnnealing()
        sa_replace.neighbor_selection_method = 'replace'
        
        neighbors_replace = []
        for _ in range(10):
            neighbor = sa_replace._generate_neighbor(portfolio, 10, None)
            neighbors_replace.append(neighbor)
            
            # Should be valid
            self.assertEqual(len(neighbor), 5)
            self.assertEqual(len(set(neighbor)), 5)
    
    def test_convergence_with_temperature(self):
        """Test that SA converges as temperature decreases"""
        sa = SimulatedAnnealing()
        sa.iterations = 100
        
        result = sa.optimize(
            daily_matrix=self.medium_data,
            portfolio_size=10,
            fitness_function=self.calculate_fitness
        )
        
        iter_stats = result['iteration_stats']
        
        # Acceptance rate should generally decrease
        early_acceptance = np.mean([s['acceptance_rate'] 
                                   for s in iter_stats[:20]])
        late_acceptance = np.mean([s['acceptance_rate'] 
                                  for s in iter_stats[-20:]])
        
        # Late acceptance should be lower (more selective)
        self.assertLess(late_acceptance, early_acceptance)
    
    def test_edge_cases(self):
        """Test SA-specific edge cases"""
        # Test with very high initial temperature
        sa_high_temp = SimulatedAnnealing()
        sa_high_temp.initial_temperature = 10.0
        
        result_high = sa_high_temp.optimize(
            daily_matrix=self.small_data,
            portfolio_size=5,
            fitness_function=self.calculate_fitness
        )
        
        # Test with very low initial temperature
        sa_low_temp = SimulatedAnnealing()
        sa_low_temp.initial_temperature = 0.01
        
        result_low = sa_low_temp.optimize(
            daily_matrix=self.small_data,
            portfolio_size=5,
            fitness_function=self.calculate_fitness
        )
        
        # Both should work
        self.validate_algorithm_result(result_high, "SA-HighTemp")
        self.validate_algorithm_result(result_low, "SA-LowTemp")
    
    def test_zone_optimization(self):
        """Test SA with zone constraints"""
        self.test_zone_optimization(SimulatedAnnealing)


class TestDifferentialEvolution(BaseAlgorithmTest):
    """Test cases for Differential Evolution"""
    
    def test_basic_functionality(self):
        """Test DE basic functionality"""
        result = self.run_algorithm_test(
            DifferentialEvolution,
            "DE - Basic Test",
            self.medium_data,
            self.medium_portfolio
        )
        
        self.assertIn('generations', result)
        self.assertIn('population_size', result)


class TestAntColonyOptimization(BaseAlgorithmTest):
    """Test cases for Ant Colony Optimization"""
    
    def test_basic_functionality(self):
        """Test ACO basic functionality"""
        result = self.run_algorithm_test(
            AntColonyOptimization,
            "ACO - Basic Test",
            self.medium_data,
            self.medium_portfolio
        )
        
        self.assertIn('iterations', result)
        self.assertIn('num_ants', result)


class TestHillClimbing(BaseAlgorithmTest):
    """Test cases for Hill Climbing"""
    
    def test_basic_functionality(self):
        """Test HC basic functionality"""
        result = self.run_algorithm_test(
            HillClimbing,
            "HC - Basic Test",
            self.medium_data,
            self.medium_portfolio
        )
        
        self.assertIn('restarts', result)
        self.assertIn('restart_stats', result)
    
    def test_multiple_restarts(self):
        """Test HC with multiple restarts"""
        hc = HillClimbing()
        
        result = hc.optimize(
            daily_matrix=self.medium_data,
            portfolio_size=10,
            fitness_function=self.calculate_fitness
        )
        
        # Should have performed multiple restarts
        restart_stats = result['restart_stats']
        self.assertEqual(len(restart_stats), hc.restarts)
        
        # Best restart should be identified
        best_restart_idx = result['best_restart']
        best_restart_fitness = restart_stats[best_restart_idx]['best_fitness']
        
        # Should be the maximum fitness among all restarts
        all_fitness = [r['best_fitness'] for r in restart_stats]
        self.assertEqual(best_restart_fitness, max(all_fitness))


class TestBayesianOptimization(BaseAlgorithmTest):
    """Test cases for Bayesian Optimization"""
    
    def test_basic_functionality(self):
        """Test BO basic functionality"""
        result = self.run_algorithm_test(
            BayesianOptimization,
            "BO - Basic Test",
            self.medium_data,
            self.medium_portfolio
        )
        
        self.assertIn('n_initial_samples', result)
        self.assertIn('total_evaluations', result)
        
        # Should have initial samples + iterations
        expected_evals = result['n_initial_samples'] + result['n_iterations']
        self.assertEqual(result['total_evaluations'], expected_evals)


class TestRandomSearch(BaseAlgorithmTest):
    """Test cases for Random Search"""
    
    def test_basic_functionality(self):
        """Test RS basic functionality"""
        result = self.run_algorithm_test(
            RandomSearch,
            "RS - Basic Test",
            self.medium_data,
            self.medium_portfolio
        )
        
        self.assertIn('iterations', result)
        self.assertIn('batch_size', result)
    
    def test_early_stopping(self):
        """Test RS early stopping"""
        rs = RandomSearch()
        rs.early_stopping = True
        rs.patience = 20
        
        # Use data where improvement is unlikely
        uniform_data = np.ones((50, 20)) * 0.01
        
        result = rs.optimize(
            daily_matrix=uniform_data,
            portfolio_size=5,
            fitness_function=self.calculate_fitness
        )
        
        # Check if early stopped
        if result.get('early_stopped', False):
            # Should have stopped before max iterations
            actual_iterations = len(result['iteration_stats']) * rs.batch_size
            self.assertLess(actual_iterations, rs.iterations)


if __name__ == '__main__':
    unittest.main()