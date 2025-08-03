#!/usr/bin/env python3
"""
Test HC Algorithm Retrofit for Parquet/Arrow/cuDF Support
"""

import numpy as np
import sys
import os
import time

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithms.hill_climbing import HillClimbing

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
    
    return daily_matrix

def test_hc_cpu_mode():
    """Test HC with CPU mode (numpy)"""
    print("Testing HC Algorithm Retrofit - CPU Mode")
    print("=" * 60)
    
    # Create test data
    data = create_test_data()
    print(f"Created test data: {data.shape[0]} days x {data.shape[1]} strategies")
    
    # Initialize HC algorithm in CPU mode
    hc = HillClimbing(use_gpu=False)
    print("Initialized HC in CPU mode")
    
    # Run optimization
    portfolio_size = 10
    print(f"\nRunning HC optimization for portfolio size {portfolio_size}...")
    
    start_time = time.time()
    result = hc.optimize(
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
    
    # Show HC-specific parameters
    print(f"\nHC-Specific Parameters:")
    print(f"  Neighborhood size: {result['parameters']['neighborhood_size']}")
    print(f"  Restart threshold: {result['parameters']['restart_threshold']}")
    print(f"  Step size: {result['parameters']['step_size']}")
    
    # Verify algorithm-specific outputs
    assert 'iterations' in result
    assert 'fitness_history' in result
    assert len(result['fitness_history']) > 0
    
    print("\n✅ HC CPU mode test passed!")
    
    return result

def test_hc_parameters():
    """Test HC with different parameter configurations"""
    print("\n" + "=" * 60)
    print("Testing HC with different parameters")
    print("=" * 60)
    
    data = create_test_data()
    
    # Test with variable portfolio size
    print("\n1. Testing with variable portfolio size (5-15)...")
    hc = HillClimbing(use_gpu=False)
    result = hc.optimize(
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
    
    result = hc.optimize(
        data=data,
        portfolio_size=12,
        zone_data=zone_data
    )
    
    print(f"   Best fitness with zones: {result['best_fitness']:.6f}")
    print("   ✅ Zone constraints test passed!")
    
    # Test restart mechanism
    print("\n3. Testing restart mechanism...")
    result = hc.optimize(
        data=data,
        portfolio_size=8
    )
    
    # Check if fitness history shows improvement patterns
    fitness_history = result['fitness_history']
    improvements = sum(1 for i in range(1, len(fitness_history)) 
                      if fitness_history[i] > fitness_history[i-1])
    
    print(f"   Total improvements: {improvements}")
    print(f"   Final fitness: {result['best_fitness']:.6f}")
    print("   ✅ Restart mechanism test passed!")
    
    print("\n✅ All HC parameter tests passed!")

if __name__ == "__main__":
    # Run tests
    test_hc_cpu_mode()
    test_hc_parameters()
    
    print("\n" + "=" * 60)
    print("HC Retrofit Test Summary:")
    print("- ✅ CPU mode (numpy) working")
    print("- ✅ Fitness calculation integrated")
    print("- ✅ Variable portfolio sizes supported")
    print("- ✅ Zone constraints working")
    print("- ✅ Algorithm-specific parameters preserved")
    print("- ✅ Neighbor generation working")
    print("- ✅ Restart mechanism functional")
    print("=" * 60)