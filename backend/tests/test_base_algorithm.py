#!/usr/bin/env python3
"""
Base Test Class for Algorithm Testing

Provides common functionality for testing all optimization algorithms
including test data generation, fitness calculation, and validation methods.
"""

import unittest
import numpy as np
import tempfile
import os
from typing import Dict, List, Tuple, Optional
from pathlib import Path

# Add backend to path
import sys
sys.path.append(str(Path(__file__).parent.parent))


class BaseAlgorithmTest(unittest.TestCase):
    """Base class for algorithm unit tests"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create test data
        self.small_data = self.create_test_data(n_days=20, n_strategies=10)
        self.medium_data = self.create_test_data(n_days=50, n_strategies=50)
        self.large_data = self.create_test_data(n_days=100, n_strategies=100)
        
        # Create temporary config file
        self.temp_config = self.create_test_config()
        
        # Standard portfolio sizes
        self.small_portfolio = 5
        self.medium_portfolio = 10
        self.large_portfolio = 20
        
    def tearDown(self):
        """Clean up test fixtures"""
        # Remove temporary config file
        if hasattr(self, 'temp_config') and os.path.exists(self.temp_config):
            os.unlink(self.temp_config)
    
    @staticmethod
    def create_test_data(n_days: int = 100, n_strategies: int = 50, seed: int = 42) -> np.ndarray:
        """
        Create synthetic daily returns data for testing
        
        Args:
            n_days: Number of trading days
            n_strategies: Number of strategies
            seed: Random seed for reproducibility
            
        Returns:
            Matrix of daily returns (days x strategies)
        """
        np.random.seed(seed)
        
        # Create base returns with some correlation structure
        n_base = min(10, n_strategies)
        base_returns = np.random.randn(n_days, n_base) * 0.02
        
        # Expand to more strategies with correlation
        returns = np.zeros((n_days, n_strategies))
        for i in range(n_strategies):
            source_idx = i % n_base
            noise = np.random.randn(n_days) * 0.01
            correlation = 0.7 if i < n_strategies // 2 else 0.3
            returns[:, i] = correlation * base_returns[:, source_idx] + (1 - correlation) * noise
        
        return returns
    
    @staticmethod
    def calculate_fitness(daily_matrix: np.ndarray, portfolio: np.ndarray) -> float:
        """
        Calculate fitness for a portfolio (ROI/Drawdown ratio with correlation penalty)
        
        This is the standard fitness function used across all tests.
        
        Args:
            daily_matrix: Daily returns matrix
            portfolio: Array of strategy indices
            
        Returns:
            Fitness score
        """
        # Extract portfolio returns
        portfolio_returns = daily_matrix[:, portfolio].mean(axis=1)
        
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
        
        # Calculate correlation penalty
        if len(portfolio) > 1:
            portfolio_data = daily_matrix[:, portfolio]
            corr_matrix = np.corrcoef(portfolio_data.T)
            
            # Average pairwise correlation (excluding diagonal)
            n = len(portfolio)
            mask = np.ones_like(corr_matrix, dtype=bool)
            np.fill_diagonal(mask, False)
            avg_correlation = corr_matrix[mask].mean()
            
            # Apply penalty (standard weight from legacy implementation)
            penalty_weight = 10.0
            penalty = penalty_weight * avg_correlation
        else:
            penalty = 0
        
        return base_fitness - penalty
    
    def create_test_config(self) -> str:
        """Create a temporary test configuration file"""
        config_content = """[GA]
population_size = 20
generations = 30
mutation_rate = 0.15
crossover_rate = 0.85

[PSO]
swarm_size = 20
iterations = 30
inertia = 0.8

[SA]
initial_temperature = 2.0
cooling_rate = 0.9
iterations = 50

[DE]
population_size = 15
generations = 25

[ACO]
num_ants = 15
iterations = 25

[HC]
iterations = 50
restarts = 3

[BO]
n_initial_samples = 10
iterations = 20

[RS]
iterations = 100
batch_size = 20
"""
        # Create temporary file
        fd, path = tempfile.mkstemp(suffix='.ini')
        with os.fdopen(fd, 'w') as f:
            f.write(config_content)
        
        return path
    
    def validate_algorithm_result(self, result: Dict, algorithm_name: str) -> None:
        """
        Validate that algorithm result has all required fields
        
        Args:
            result: Algorithm result dictionary
            algorithm_name: Name of the algorithm
        """
        # Required fields
        required_fields = ['best_portfolio', 'best_fitness', 'execution_time', 'algorithm_name']
        
        for field in required_fields:
            self.assertIn(field, result, f"{algorithm_name} missing required field: {field}")
        
        # Type checks
        self.assertIsInstance(result['best_portfolio'], list, 
                            f"{algorithm_name} best_portfolio should be a list")
        self.assertIsInstance(result['best_fitness'], (int, float), 
                            f"{algorithm_name} best_fitness should be numeric")
        self.assertIsInstance(result['execution_time'], (int, float), 
                            f"{algorithm_name} execution_time should be numeric")
        
        # Value checks
        self.assertGreater(len(result['best_portfolio']), 0, 
                         f"{algorithm_name} returned empty portfolio")
        self.assertGreaterEqual(result['execution_time'], 0, 
                              f"{algorithm_name} execution_time should be non-negative")
        
        # Check portfolio validity
        portfolio = result['best_portfolio']
        self.assertEqual(len(portfolio), len(set(portfolio)), 
                        f"{algorithm_name} portfolio contains duplicates")
    
    def compare_fitness_values(self, fitness1: float, fitness2: float, 
                             tolerance: float = 0.1) -> bool:
        """
        Compare two fitness values within tolerance
        
        Args:
            fitness1: First fitness value
            fitness2: Second fitness value
            tolerance: Relative tolerance for comparison
            
        Returns:
            True if values are within tolerance
        """
        if fitness1 == 0 and fitness2 == 0:
            return True
        
        if fitness1 == 0 or fitness2 == 0:
            return abs(fitness1 - fitness2) < tolerance
        
        relative_diff = abs(fitness1 - fitness2) / max(abs(fitness1), abs(fitness2))
        return relative_diff <= tolerance
    
    def run_algorithm_test(self, algorithm_class, test_name: str, 
                         data: np.ndarray, portfolio_size: int,
                         config_path: Optional[str] = None) -> Dict:
        """
        Run a standard test for an algorithm
        
        Args:
            algorithm_class: Algorithm class to test
            test_name: Name of the test for logging
            data: Test data
            portfolio_size: Portfolio size
            config_path: Optional config file path
            
        Returns:
            Algorithm result
        """
        print(f"\nRunning {test_name}...")
        
        # Create algorithm instance
        algorithm = algorithm_class(config_path)
        
        # Run optimization
        result = algorithm.optimize(
            daily_matrix=data,
            portfolio_size=portfolio_size,
            fitness_function=self.calculate_fitness
        )
        
        # Validate result
        self.validate_algorithm_result(result, algorithm_class.__name__)
        
        print(f"  Portfolio size: {len(result['best_portfolio'])}")
        print(f"  Best fitness: {result['best_fitness']:.4f}")
        print(f"  Execution time: {result['execution_time']:.3f}s")
        
        return result
    
    def test_edge_case_empty_data(self, algorithm_class):
        """Test algorithm with empty data"""
        empty_data = np.array([]).reshape(0, 0)
        algorithm = algorithm_class()
        
        with self.assertRaises(ValueError):
            algorithm.optimize(
                daily_matrix=empty_data,
                portfolio_size=5,
                fitness_function=self.calculate_fitness
            )
    
    def test_edge_case_single_strategy(self, algorithm_class):
        """Test algorithm with single strategy"""
        single_strategy_data = self.create_test_data(n_days=20, n_strategies=1)
        algorithm = algorithm_class()
        
        result = algorithm.optimize(
            daily_matrix=single_strategy_data,
            portfolio_size=1,
            fitness_function=self.calculate_fitness
        )
        
        self.assertEqual(len(result['best_portfolio']), 1)
        self.assertEqual(result['best_portfolio'][0], 0)
    
    def test_edge_case_portfolio_equals_strategies(self, algorithm_class):
        """Test when portfolio size equals number of strategies"""
        n_strategies = 5
        data = self.create_test_data(n_days=20, n_strategies=n_strategies)
        algorithm = algorithm_class()
        
        result = algorithm.optimize(
            daily_matrix=data,
            portfolio_size=n_strategies,
            fitness_function=self.calculate_fitness
        )
        
        self.assertEqual(len(result['best_portfolio']), n_strategies)
        self.assertEqual(set(result['best_portfolio']), set(range(n_strategies)))
    
    def test_zone_optimization(self, algorithm_class):
        """Test zone-wise optimization support"""
        data = self.create_test_data(n_days=50, n_strategies=30)
        algorithm = algorithm_class()
        
        # Define zone constraints
        zone_data = {
            'allowed_strategies': list(range(10, 20)),  # Strategies 10-19
            'min_strategies_per_zone': 3
        }
        
        result = algorithm.optimize(
            daily_matrix=data,
            portfolio_size=5,
            fitness_function=self.calculate_fitness,
            zone_data=zone_data
        )
        
        # Check that portfolio respects zone constraints
        portfolio = result['best_portfolio']
        zone_strategies = [s for s in portfolio if s in zone_data['allowed_strategies']]
        
        self.assertGreaterEqual(len(zone_strategies), zone_data['min_strategies_per_zone'],
                              f"Portfolio should have at least {zone_data['min_strategies_per_zone']} "
                              f"strategies from zone, got {len(zone_strategies)}")
    
    def test_variable_portfolio_size(self, algorithm_class):
        """Test variable portfolio size support"""
        data = self.create_test_data(n_days=50, n_strategies=30)
        algorithm = algorithm_class()
        
        # Test with size range
        result = algorithm.optimize(
            daily_matrix=data,
            portfolio_size=(5, 10),  # Range from 5 to 10
            fitness_function=self.calculate_fitness
        )
        
        # Currently algorithms use min_size, but this may change
        portfolio_size = len(result['best_portfolio'])
        self.assertGreaterEqual(portfolio_size, 5)
        self.assertLessEqual(portfolio_size, 10)
    
    def test_configuration_loading(self, algorithm_class):
        """Test that algorithm loads configuration correctly"""
        # Test with config
        algorithm_with_config = algorithm_class(self.temp_config)
        
        # Test without config (should use defaults)
        algorithm_without_config = algorithm_class()
        
        # Both should work
        data = self.small_data
        portfolio_size = self.small_portfolio
        
        result_with_config = algorithm_with_config.optimize(
            daily_matrix=data,
            portfolio_size=portfolio_size,
            fitness_function=self.calculate_fitness
        )
        
        result_without_config = algorithm_without_config.optimize(
            daily_matrix=data,
            portfolio_size=portfolio_size,
            fitness_function=self.calculate_fitness
        )
        
        # Both should produce valid results
        self.validate_algorithm_result(result_with_config, algorithm_class.__name__)
        self.validate_algorithm_result(result_without_config, algorithm_class.__name__)
    
    def test_deterministic_behavior(self, algorithm_class, n_runs: int = 3):
        """Test that algorithm behaves consistently across runs"""
        data = self.small_data
        portfolio_size = self.small_portfolio
        
        results = []
        for i in range(n_runs):
            # Set seed for reproducibility
            np.random.seed(42)
            
            algorithm = algorithm_class()
            result = algorithm.optimize(
                daily_matrix=data,
                portfolio_size=portfolio_size,
                fitness_function=self.calculate_fitness
            )
            results.append(result)
        
        # Check that results are similar (not necessarily identical due to randomness)
        fitness_values = [r['best_fitness'] for r in results]
        
        # All fitness values should be within reasonable range
        min_fitness = min(fitness_values)
        max_fitness = max(fitness_values)
        
        if min_fitness != 0:
            variation = (max_fitness - min_fitness) / abs(min_fitness)
            self.assertLess(variation, 0.5, 
                          f"Algorithm shows too much variation: {variation:.2%}")


if __name__ == '__main__':
    unittest.main()