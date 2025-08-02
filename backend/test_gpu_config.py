#!/usr/bin/env python3
"""
Test GPU Configuration Settings
"""

import os
import sys

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lib.config_reader import GPUConfigReader, get_gpu_config, is_gpu_required, should_use_gpu


def test_config():
    """Test GPU configuration reading"""
    print("🔧 GPU Configuration Test")
    print("="*60)
    
    # Read configuration
    config = get_gpu_config()
    
    print("\n📋 Current Configuration:")
    print(f"  GPU Enabled: {config['enabled']}")
    print(f"  Acceleration Mode: {config['acceleration']}")
    print(f"  CPU Fallback Allowed: {config['cpu_fallback_allowed']}")
    print(f"  Force GPU Mode: {config['force_gpu_mode']}")
    print(f"  Min Strategies for GPU: {config['min_strategies_for_gpu']}")
    
    print(f"\n🚦 Configuration Results:")
    print(f"  Is GPU Required (no fallback): {is_gpu_required()}")
    print(f"  Should use GPU for 5 strategies: {should_use_gpu(5)}")
    print(f"  Should use GPU for 10 strategies: {should_use_gpu(10)}")
    print(f"  Should use GPU for 100 strategies: {should_use_gpu(100)}")
    
    # Test environment variable override
    print("\n🔄 Testing Environment Variable Override:")
    os.environ['GPU_FALLBACK_ALLOWED'] = 'true'
    os.environ['FORCE_GPU_MODE'] = 'false'
    
    # Re-read config (need new instance)
    reader = GPUConfigReader()
    config2 = reader.get_gpu_config()
    
    print(f"  After setting GPU_FALLBACK_ALLOWED=true:")
    print(f"    CPU Fallback Allowed: {config2['cpu_fallback_allowed']}")
    print(f"    Is GPU Required: {reader.is_gpu_required()}")
    
    # Clean up env vars
    del os.environ['GPU_FALLBACK_ALLOWED']
    del os.environ['FORCE_GPU_MODE']


def test_workflow_integration():
    """Test workflow with GPU configuration"""
    print("\n\n🚀 Testing Workflow Integration")
    print("="*60)
    
    import pandas as pd
    import numpy as np
    from csv_only_heavydb_workflow import CSVOnlyHeavyDBOptimizer
    
    # Create tiny test data (below GPU threshold)
    n_strategies = 5  # Below threshold of 10
    n_days = 20
    
    dates = pd.date_range('2024-01-01', periods=n_days)
    data = {'Date': dates, 'trading_day': range(n_days)}
    
    for i in range(n_strategies):
        data[f"SENSEX_{i}"] = np.random.randn(n_days) * 100
    
    df = pd.DataFrame(data)
    test_file = '/tmp/gpu_config_test.csv'
    df.to_csv(test_file, index=False)
    
    print(f"Created test data: {n_strategies} strategies (below GPU threshold)")
    
    # Test with current config (GPU required, no fallback)
    optimizer = CSVOnlyHeavyDBOptimizer()
    
    try:
        print("\nTesting with GPU required and small dataset...")
        optimizer.selected_algorithms = ['GA']
        result = optimizer.run_optimization(test_file, portfolio_size=3)
        print("❌ Should have failed but didn't!")
    except Exception as e:
        print(f"✅ Correctly rejected: {e}")
    
    # Clean up
    os.remove(test_file)
    
    # Test with larger dataset
    n_strategies = 15  # Above threshold
    data = {'Date': dates, 'trading_day': range(n_days)}
    
    for i in range(n_strategies):
        data[f"SENSEX_{i}"] = np.random.randn(n_days) * 100
    
    df = pd.DataFrame(data)
    df.to_csv(test_file, index=False)
    
    print(f"\nCreated test data: {n_strategies} strategies (above GPU threshold)")
    
    try:
        print("Testing with GPU required and sufficient data...")
        result = optimizer.run_optimization(test_file, portfolio_size=5)
        print("✅ GPU processing successful!")
    except Exception as e:
        print(f"❌ Failed: {e}")
    
    # Clean up
    if os.path.exists(test_file):
        os.remove(test_file)


if __name__ == "__main__":
    test_config()
    test_workflow_integration()
    print("\n✅ GPU Configuration Tests Complete!")