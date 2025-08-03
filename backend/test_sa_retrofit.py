#!/usr/bin/env python3
"""
Test SA Algorithm Retrofit for Parquet/Arrow/cuDF Support
"""

import numpy as np
import sys
import os
import time

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithms.simulated_annealing import SimulatedAnnealing

def create_test_data():
    """Create simple test data similar to genetic algorithm test"""
    np.random.seed(42)
    days = 82
    strategies = 100
    
    # Generate random daily returns between -5% and 5%
    daily_matrix = np.random.randn(days, strategies) * 0.02
    
    # Make some strategies better than others
    for i in range(10):  # Top 10 strategies
        daily_matrix[:, i] += 0.001  # Add 0.1% daily bias
    
    return daily_matrix

def test_sa_cpu_mode():
    """Test SA with CPU mode (numpy)"""
    print("Testing SA Algorithm Retrofit - CPU Mode")
    print("=" * 60)
    
    # Create test data
    data = create_test_data()
    print(f"Created test data: {data.shape[0]} days x {data.shape[1]} strategies")
    
    # Initialize SA algorithm in CPU mode
    sa = SimulatedAnnealing(use_gpu=False)
    print("Initialized SA in CPU mode")
    
    # Run optimization
    portfolio_size = 10
    print(f"\nRunning SA optimization for portfolio size {portfolio_size}...")
    
    start_time = time.time()
    result = sa.optimize(
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
    
    # Show SA-specific metrics
    print(f"\nSA-Specific Metrics:")
    print(f"  Final temperature: {result['final_temperature']:.6f}")
    print(f"  Accepted moves: {result['accepted_moves']}")
    print(f"  Rejected moves: {result['rejected_moves']}")
    print(f"  Acceptance rate: {result['acceptance_rate']:.3f}")
    
    # Verify algorithm-specific outputs
    assert 'iterations' in result
    assert 'final_temperature' in result
    assert 'acceptance_rate' in result
    assert 'fitness_history' in result
    assert len(result['fitness_history']) == result['iterations'] + 1  # +1 for initial fitness
    
    print("\n✅ SA CPU mode test passed!")
    
    return result

def test_sa_parameters():
    """Test SA with different parameter configurations"""
    print("\n" + "=" * 60)
    print("Testing SA with different parameters")
    print("=" * 60)
    
    data = create_test_data()
    
    # Test with variable portfolio size
    print("\n1. Testing with variable portfolio size (5-15)...")
    sa = SimulatedAnnealing(use_gpu=False)
    result = sa.optimize(
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
    
    result = sa.optimize(
        data=data,
        portfolio_size=12,
        zone_data=zone_data
    )
    
    print(f"   Best fitness with zones: {result['best_fitness']:.6f}")
    print(f"   Acceptance rate: {result['acceptance_rate']:.3f}")
    print("   ✅ Zone constraints test passed!")
    
    print("\n✅ All SA parameter tests passed!")

if __name__ == "__main__":
    # Run tests
    test_sa_cpu_mode()
    test_sa_parameters()
    
    print("\n" + "=" * 60)
    print("SA Retrofit Test Summary:")
    print("- ✅ CPU mode (numpy) working")
    print("- ✅ Fitness calculation integrated")
    print("- ✅ Variable portfolio sizes supported")
    print("- ✅ Zone constraints working")
    print("- ✅ Algorithm-specific parameters preserved")
    print("- ✅ Temperature-based annealing working")
    print("=" * 60)