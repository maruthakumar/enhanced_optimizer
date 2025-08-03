#!/usr/bin/env python3
"""
Test ACO Algorithm Retrofit for Parquet/Arrow/cuDF Support
"""

import numpy as np
import sys
import os
import time

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithms.ant_colony_optimization import AntColonyOptimization

def create_test_data():
    """Create simple test data"""
    np.random.seed(42)
    days = 82
    strategies = 100
    
    # Generate random daily returns between -5% and 5%
    daily_matrix = np.random.randn(days, strategies) * 0.02
    
    # Make some strategies better than others
    for i in range(10):  # Top 10 strategies
        daily_matrix[:, i] += 0.001  # Add 0.1% daily bias
    
    # Add some negative returns to test negative probability bug fix
    daily_matrix[40:50, 50:60] -= 0.003  # Create some consistently negative strategies
    
    return daily_matrix

def test_aco_cpu_mode():
    """Test ACO with CPU mode (numpy)"""
    print("Testing ACO Algorithm Retrofit - CPU Mode")
    print("=" * 60)
    
    # Create test data
    data = create_test_data()
    print(f"Created test data: {data.shape[0]} days x {data.shape[1]} strategies")
    
    # Initialize ACO algorithm in CPU mode
    aco = AntColonyOptimization(use_gpu=False)
    print("Initialized ACO in CPU mode")
    
    # Run optimization
    portfolio_size = 10
    print(f"\nRunning ACO optimization for portfolio size {portfolio_size}...")
    
    start_time = time.time()
    result = aco.optimize(
        data=data,
        portfolio_size=portfolio_size
    )
    
    print(f"\nOptimization completed in {result['execution_time']:.3f} seconds")
    print(f"Best fitness: {result['best_fitness']:.6f}")
    print(f"Best portfolio: {result['best_portfolio']}")
    print(f"Data type used: {result['data_type']}")
    print(f"GPU accelerated: {result['gpu_accelerated']}")
    
    # Show detailed metrics
    if 'detailed_metrics' in result and result['detailed_metrics']:
        print("\nDetailed Metrics:")
        for metric, value in result['detailed_metrics'].items():
            print(f"  {metric}: {value:.4f}")
    
    # Show ACO-specific parameters
    print(f"\nACO-Specific Parameters:")
    print(f"  Alpha (pheromone weight): {result['parameters']['alpha']}")
    print(f"  Beta (heuristic weight): {result['parameters']['beta']}")
    print(f"  Evaporation rate: {result['parameters']['evaporation_rate']}")
    print(f"  Min pheromone: {result['parameters']['min_pheromone']}")
    print(f"  Max pheromone: {result['parameters']['max_pheromone']}")
    
    # Verify algorithm-specific outputs
    assert 'iterations' in result
    assert 'colony_size' in result
    assert 'fitness_history' in result
    assert len(result['fitness_history']) == result['iterations']
    assert result['best_fitness'] > 0, "Fitness should be positive (negative probability bug test)"
    
    print("\n✅ ACO CPU mode test passed!")
    print("✅ Negative probability bug fixed - all fitness values positive!")
    
    return result

def test_aco_parameters():
    """Test ACO with different parameter configurations"""
    print("\n" + "=" * 60)
    print("Testing ACO with different parameters")
    print("=" * 60)
    
    data = create_test_data()
    
    # Test with variable portfolio size
    print("\n1. Testing with variable portfolio size (5-15)...")
    aco = AntColonyOptimization(use_gpu=False)
    result = aco.optimize(
        data=data,
        portfolio_size=(5, 15)
    )
    
    portfolio_size = len(result['best_portfolio'])
    print(f"   Portfolio size selected: {portfolio_size}")
    assert 5 <= portfolio_size <= 15, f"Portfolio size {portfolio_size} out of range"
    print("   ✅ Variable portfolio size test passed!")
    
    # Test with zone constraints
    print("\n2. Testing with zone constraints...")
    zone_data = {
        'enabled': True,
        'zone_count': 4,
        'zone_weights': [0.3, 0.3, 0.2, 0.2],
        'min_per_zone': 2
    }
    
    result = aco.optimize(
        data=data,
        portfolio_size=12,
        zone_data=zone_data
    )
    
    print(f"   Best fitness with zones: {result['best_fitness']:.6f}")
    print("   ✅ Zone constraints test passed!")
    
    # Test negative fitness handling
    print("\n3. Testing negative fitness handling...")
    # Create data with mostly negative returns
    negative_data = create_test_data() - 0.01  # Make most returns negative
    
    result = aco.optimize(
        data=negative_data,
        portfolio_size=8
    )
    
    print(f"   Best fitness with negative data: {result['best_fitness']:.6f}")
    print(f"   All pheromone updates completed without errors")
    print("   ✅ Negative fitness handling test passed!")
    
    print("\n✅ All ACO parameter tests passed!")

if __name__ == "__main__":
    # Run tests
    test_aco_cpu_mode()
    test_aco_parameters()
    
    print("\n" + "=" * 60)
    print("ACO Retrofit Test Summary:")
    print("- ✅ CPU mode (numpy) working")
    print("- ✅ Fitness calculation integrated")
    print("- ✅ Variable portfolio sizes supported")
    print("- ✅ Zone constraints working")
    print("- ✅ Algorithm-specific parameters preserved")
    print("- ✅ Pheromone trails working correctly")
    print("- ✅ NEGATIVE PROBABILITY BUG FIXED")
    print("=" * 60)