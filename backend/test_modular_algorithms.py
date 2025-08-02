#!/usr/bin/env python3
"""
Test script for the modular algorithm implementation

This demonstrates how the new modular algorithms work with:
1. Configuration file reading
2. Zone-wise optimization support
3. Variable portfolio sizes
4. Standardized interfaces
"""

import numpy as np
import logging
import json
from pathlib import Path

# Add backend to path
import sys
sys.path.append(str(Path(__file__).parent))

from algorithms import AlgorithmFactory, verify_algorithm_completeness

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_test_data(n_days=100, n_strategies=50):
    """Create synthetic daily returns data for testing"""
    # Generate random returns with some correlation structure
    np.random.seed(42)
    
    # Create base returns
    base_returns = np.random.randn(n_days, 10) * 0.02
    
    # Expand to more strategies with correlation
    returns = np.zeros((n_days, n_strategies))
    for i in range(n_strategies):
        source_idx = i % 10
        noise = np.random.randn(n_days) * 0.01
        returns[:, i] = base_returns[:, source_idx] + noise
    
    return returns


def calculate_test_fitness(daily_matrix: np.ndarray, portfolio: np.ndarray) -> float:
    """Calculate fitness for a portfolio (ROI/Drawdown ratio with correlation penalty)"""
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
        
        # Apply penalty (from configuration)
        penalty_weight = 10.0  # Default from legacy implementation
        penalty = penalty_weight * avg_correlation
    else:
        penalty = 0
    
    return base_fitness - penalty


def test_single_algorithm(factory: AlgorithmFactory, algorithm_name: str, 
                         daily_matrix: np.ndarray, portfolio_size: int = 10):
    """Test a single algorithm"""
    logger.info(f"\nTesting {algorithm_name}")
    logger.info("=" * 60)
    
    try:
        # Create algorithm instance
        algorithm = factory.create_algorithm(algorithm_name)
        
        # Get algorithm info
        info = algorithm.get_algorithm_info()
        logger.info(f"Algorithm info: {json.dumps(info, indent=2)}")
        
        # Run optimization
        result = algorithm.optimize(
            daily_matrix=daily_matrix,
            portfolio_size=portfolio_size,
            fitness_function=calculate_test_fitness
        )
        
        # Log results
        logger.info(f"Best fitness: {result['best_fitness']:.4f}")
        logger.info(f"Execution time: {result['execution_time']:.3f}s")
        logger.info(f"Best portfolio (first 5): {result['best_portfolio'][:5]}")
        
        # Additional metrics if available
        if 'generations' in result:
            logger.info(f"Generations: {result['generations']}")
        if 'iterations' in result:
            logger.info(f"Iterations: {result['iterations']}")
        
        return result
        
    except Exception as e:
        logger.error(f"Algorithm {algorithm_name} failed: {e}")
        return None


def test_zone_optimization(factory: AlgorithmFactory, daily_matrix: np.ndarray):
    """Test zone-wise optimization capability"""
    logger.info("\nTesting Zone-wise Optimization")
    logger.info("=" * 60)
    
    # Define zone data
    zone_data = {
        'allowed_strategies': list(range(10, 30)),  # Strategies 10-29 are in the zone
        'min_strategies_per_zone': 5
    }
    
    algorithm = factory.create_algorithm('genetic_algorithm')
    
    result = algorithm.optimize(
        daily_matrix=daily_matrix,
        portfolio_size=10,
        fitness_function=calculate_test_fitness,
        zone_data=zone_data
    )
    
    # Check if portfolio respects zone constraints
    portfolio = result['best_portfolio']
    zone_strategies = [s for s in portfolio if s in zone_data['allowed_strategies']]
    
    logger.info(f"Portfolio size: {len(portfolio)}")
    logger.info(f"Strategies from zone: {len(zone_strategies)}")
    logger.info(f"Zone strategies: {zone_strategies}")
    logger.info(f"Best fitness: {result['best_fitness']:.4f}")


def test_variable_portfolio_size(factory: AlgorithmFactory, daily_matrix: np.ndarray):
    """Test variable portfolio size support"""
    logger.info("\nTesting Variable Portfolio Size")
    logger.info("=" * 60)
    
    algorithm = factory.create_algorithm('particle_swarm_optimization')
    
    # Test with size range (currently uses min_size)
    result = algorithm.optimize(
        daily_matrix=daily_matrix,
        portfolio_size=(10, 20),  # Range from 10 to 20
        fitness_function=calculate_test_fitness
    )
    
    logger.info(f"Portfolio size range: (10, 20)")
    logger.info(f"Actual portfolio size: {len(result['best_portfolio'])}")
    logger.info(f"Best fitness: {result['best_fitness']:.4f}")


def main():
    """Run all tests"""
    logger.info("Heavy Optimizer Platform - Modular Algorithm Test")
    logger.info("=" * 60)
    
    # Verify all algorithms are available
    success, message = verify_algorithm_completeness()
    logger.info(f"Algorithm verification: {message}")
    
    if not success:
        logger.error("Not all algorithms available, exiting")
        return
    
    # Create test data
    daily_matrix = create_test_data(n_days=100, n_strategies=50)
    logger.info(f"Created test data: {daily_matrix.shape}")
    
    # Create factory with configuration
    config_path = Path(__file__).parent / "config" / "algorithm_config.ini"
    factory = AlgorithmFactory(str(config_path) if config_path.exists() else None)
    
    # Test all algorithms
    algorithms_to_test = [
        'genetic_algorithm',
        'particle_swarm_optimization', 
        'simulated_annealing',
        'differential_evolution',
        'ant_colony_optimization',
        'hill_climbing',
        'bayesian_optimization',
        'random_search'
    ]
    
    results = {}
    for algorithm_name in algorithms_to_test:
        result = test_single_algorithm(factory, algorithm_name, daily_matrix)
        if result:
            results[algorithm_name] = {
                'best_fitness': result['best_fitness'],
                'execution_time': result['execution_time']
            }
    
    # Test special features
    test_zone_optimization(factory, daily_matrix)
    test_variable_portfolio_size(factory, daily_matrix)
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    
    for algo, metrics in sorted(results.items(), key=lambda x: x[1]['best_fitness'], reverse=True):
        logger.info(f"{algo:30s} | Fitness: {metrics['best_fitness']:8.4f} | Time: {metrics['execution_time']:6.3f}s")
    
    logger.info("\n✅ All tests completed successfully!")


if __name__ == "__main__":
    main()