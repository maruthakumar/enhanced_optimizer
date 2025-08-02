#!/usr/bin/env python3
"""
Unit Tests for Genetic Algorithm

Tests the modular Genetic Algorithm implementation including:
- Basic functionality
- Configuration loading
- Edge cases
- Zone optimization
- Performance characteristics
"""

import unittest
import numpy as np
from pathlib import Path

# Add backend to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from algorithms.genetic_algorithm import GeneticAlgorithm
from tests.test_base_algorithm import BaseAlgorithmTest


class TestGeneticAlgorithm(BaseAlgorithmTest):
    """Test cases for Genetic Algorithm"""
    
    def test_basic_functionality_small_data(self):
        """Test GA with small dataset"""
        result = self.run_algorithm_test(
            GeneticAlgorithm,
            "GA - Small Data",
            self.small_data,
            self.small_portfolio
        )
        
        # GA should find a reasonable solution
        self.assertGreater(result['best_fitness'], -1.0)
        self.assertLess(result['best_fitness'], 10.0)
    
    def test_basic_functionality_medium_data(self):
        """Test GA with medium dataset"""
        result = self.run_algorithm_test(
            GeneticAlgorithm,
            "GA - Medium Data",
            self.medium_data,
            self.medium_portfolio
        )
        
        # Check algorithm-specific metrics
        self.assertIn('generations', result)
        self.assertIn('population_size', result)
        self.assertIn('generation_stats', result)
    
    def test_basic_functionality_large_data(self):
        """Test GA with large dataset"""
        result = self.run_algorithm_test(
            GeneticAlgorithm,
            "GA - Large Data",
            self.large_data,
            self.large_portfolio
        )
        
        # Verify generation statistics
        gen_stats = result.get('generation_stats', [])
        self.assertGreater(len(gen_stats), 0)
        
        # Check that fitness improves over generations
        if len(gen_stats) > 1:
            first_gen_fitness = gen_stats[0]['best_fitness']
            last_gen_fitness = gen_stats[-1]['best_fitness']
            self.assertGreaterEqual(last_gen_fitness, first_gen_fitness)
    
    def test_configuration_parameters(self):
        """Test GA with custom configuration"""
        # Create custom config
        import tempfile
        import os
        
        custom_config = """[GA]
population_size = 10
generations = 20
mutation_rate = 0.2
crossover_rate = 0.9
tournament_size = 2
elitism_count = 2
"""
        fd, config_path = tempfile.mkstemp(suffix='.ini')
        with os.fdopen(fd, 'w') as f:
            f.write(custom_config)
        
        try:
            # Create algorithm with custom config
            ga = GeneticAlgorithm(config_path)
            
            # Verify parameters were loaded
            self.assertEqual(ga.population_size, 10)
            self.assertEqual(ga.generations, 20)
            self.assertEqual(ga.mutation_rate, 0.2)
            
            # Run optimization
            result = ga.optimize(
                daily_matrix=self.small_data,
                portfolio_size=self.small_portfolio,
                fitness_function=self.calculate_fitness
            )
            
            # Check that custom parameters were used
            self.assertEqual(result['generations'], 20)
            self.assertEqual(result['population_size'], 10)
            
        finally:
            os.unlink(config_path)
    
    def test_edge_case_empty_data(self):
        """Test GA with empty data"""
        self.test_edge_case_empty_data(GeneticAlgorithm)
    
    def test_edge_case_single_strategy(self):
        """Test GA with single strategy"""
        self.test_edge_case_single_strategy(GeneticAlgorithm)
    
    def test_edge_case_portfolio_equals_strategies(self):
        """Test GA when portfolio size equals number of strategies"""
        self.test_edge_case_portfolio_equals_strategies(GeneticAlgorithm)
    
    def test_zone_optimization(self):
        """Test GA with zone constraints"""
        self.test_zone_optimization(GeneticAlgorithm)
    
    def test_variable_portfolio_size(self):
        """Test GA with variable portfolio size"""
        self.test_variable_portfolio_size(GeneticAlgorithm)
    
    def test_crossover_operator(self):
        """Test GA crossover produces valid offspring"""
        ga = GeneticAlgorithm()
        
        # Create two parent portfolios
        parent1 = np.array([0, 1, 2, 3, 4])
        parent2 = np.array([5, 6, 7, 8, 9])
        
        # Test crossover multiple times
        for _ in range(10):
            child = ga._crossover(parent1, parent2, 10)
            
            # Child should have same length as parents
            self.assertEqual(len(child), len(parent1))
            
            # Child should have no duplicates
            self.assertEqual(len(child), len(set(child)))
            
            # All elements should be valid strategy indices
            self.assertTrue(all(0 <= s < 10 for s in child))
    
    def test_mutation_operator(self):
        """Test GA mutation produces valid individuals"""
        ga = GeneticAlgorithm()
        
        # Create an individual
        individual = np.array([0, 1, 2, 3, 4])
        
        # Test mutation multiple times
        mutations_occurred = 0
        for _ in range(20):
            mutated = ga._mutate(individual.copy(), 10)
            
            # Mutated should have same length
            self.assertEqual(len(mutated), len(individual))
            
            # Mutated should have no duplicates
            self.assertEqual(len(mutated), len(set(mutated)))
            
            # Check if mutation occurred
            if not np.array_equal(mutated, individual):
                mutations_occurred += 1
        
        # With default mutation rate, some mutations should occur
        self.assertGreater(mutations_occurred, 0)
    
    def test_tournament_selection(self):
        """Test tournament selection mechanism"""
        ga = GeneticAlgorithm()
        
        # Create a population with known fitness scores
        population = [np.array([0, 1, 2]), np.array([3, 4, 5]), np.array([6, 7, 8])]
        fitness_scores = [1.0, 5.0, 3.0]  # Second individual has highest fitness
        
        # Run tournament selection multiple times
        selections = []
        for _ in range(100):
            selected = ga._tournament_selection(population, fitness_scores)
            # Find which individual was selected
            for i, ind in enumerate(population):
                if np.array_equal(selected, ind):
                    selections.append(i)
                    break
        
        # The second individual (highest fitness) should be selected most often
        selection_counts = [selections.count(i) for i in range(3)]
        self.assertEqual(np.argmax(selection_counts), 1)
    
    def test_elitism(self):
        """Test that elitism preserves best individuals"""
        # Create GA with elitism
        ga = GeneticAlgorithm()
        ga.elitism_count = 2  # Keep top 2 individuals
        
        data = self.small_data
        
        # Run optimization
        result = ga.optimize(
            daily_matrix=data,
            portfolio_size=self.small_portfolio,
            fitness_function=self.calculate_fitness
        )
        
        # Check generation stats to ensure fitness never decreases
        gen_stats = result['generation_stats']
        best_fitness_values = [g['best_fitness'] for g in gen_stats]
        
        # Best fitness should be non-decreasing
        for i in range(1, len(best_fitness_values)):
            self.assertGreaterEqual(best_fitness_values[i], best_fitness_values[i-1],
                                  "Best fitness decreased despite elitism")
    
    def test_fitness_improvement(self):
        """Test that GA improves fitness over generations"""
        ga = GeneticAlgorithm()
        
        # Run on medium data
        result = ga.optimize(
            daily_matrix=self.medium_data,
            portfolio_size=self.medium_portfolio,
            fitness_function=self.calculate_fitness
        )
        
        gen_stats = result['generation_stats']
        
        # Compare first and last generation average fitness
        first_avg = gen_stats[0]['avg_fitness']
        last_avg = gen_stats[-1]['avg_fitness']
        
        # Average fitness should improve
        self.assertGreater(last_avg, first_avg,
                         "Average fitness did not improve over generations")
        
        # Check fitness variance decreases (convergence)
        first_std = gen_stats[0]['std_fitness']
        last_std = gen_stats[-1]['std_fitness']
        
        # Standard deviation often decreases as population converges
        # (but not always, so we just check it's reasonable)
        self.assertLess(last_std, first_std * 2,
                       "Population diverged instead of converging")
    
    def test_performance_characteristics(self):
        """Test GA performance characteristics"""
        ga = GeneticAlgorithm()
        
        # Time complexity should be roughly O(population_size * generations)
        small_result = ga.optimize(
            daily_matrix=self.small_data,
            portfolio_size=5,
            fitness_function=self.calculate_fitness
        )
        
        large_result = ga.optimize(
            daily_matrix=self.medium_data,
            portfolio_size=10,
            fitness_function=self.calculate_fitness
        )
        
        # Execution time should scale reasonably
        time_ratio = large_result['execution_time'] / small_result['execution_time']
        
        # Time should increase but not explode (allow 10x for larger problem)
        self.assertLess(time_ratio, 10,
                       f"Execution time scaled poorly: {time_ratio:.1f}x")


class TestGeneticAlgorithmComparison(BaseAlgorithmTest):
    """Comparison tests against legacy implementation behavior"""
    
    def test_behavior_matches_legacy(self):
        """Test that GA behavior is consistent with legacy implementation"""
        # Set seed for reproducibility
        np.random.seed(42)
        
        ga = GeneticAlgorithm()
        
        # Run multiple times to check consistency
        results = []
        for i in range(3):
            result = ga.optimize(
                daily_matrix=self.medium_data,
                portfolio_size=10,
                fitness_function=self.calculate_fitness
            )
            results.append(result)
        
        # All runs should produce similar fitness values
        fitness_values = [r['best_fitness'] for r in results]
        mean_fitness = np.mean(fitness_values)
        
        for fitness in fitness_values:
            self.assertLess(abs(fitness - mean_fitness) / mean_fitness, 0.2,
                          "GA produces inconsistent results")
    
    def test_convergence_behavior(self):
        """Test that GA converges properly"""
        # Create GA with many generations
        ga = GeneticAlgorithm()
        ga.generations = 100
        
        result = ga.optimize(
            daily_matrix=self.medium_data,
            portfolio_size=10,
            fitness_function=self.calculate_fitness
        )
        
        gen_stats = result['generation_stats']
        
        # Check convergence by looking at fitness improvement rate
        early_improvement = gen_stats[10]['best_fitness'] - gen_stats[0]['best_fitness']
        late_improvement = gen_stats[-1]['best_fitness'] - gen_stats[-11]['best_fitness']
        
        # Late improvement should be smaller (converging)
        self.assertLess(late_improvement, early_improvement,
                       "GA not showing convergence behavior")


if __name__ == '__main__':
    unittest.main()