#!/usr/bin/env python3
"""
Test with real production data in CPU mode
Temporarily override GPU requirements for testing
"""

import os
import sys
import time
import configparser
import shutil

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Real production data file
REAL_DATA_FILE = "/mnt/optimizer_share/input/Python_Multi_Consolidated_20250726_161921.csv"

def create_test_config():
    """Create temporary config that allows CPU fallback"""
    original_config = "/mnt/optimizer_share/config/production_config.ini"
    test_config = "/mnt/optimizer_share/config/test_config.ini"
    
    # Copy original config
    shutil.copy2(original_config, test_config)
    
    # Modify GPU settings
    config = configparser.ConfigParser()
    config.read(test_config)
    
    if 'GPU' not in config:
        config.add_section('GPU')
    
    config['GPU']['gpu_acceleration'] = 'optional'
    config['GPU']['cpu_fallback_allowed'] = 'true'
    config['GPU']['force_gpu_mode'] = 'false'
    config['GPU']['heavydb_enabled'] = 'false'  # Disable HeavyDB for this test
    
    # Write modified config
    with open(test_config, 'w') as f:
        config.write(f)
    
    return test_config

def test_with_cpu_mode():
    """Test workflow with CPU mode"""
    print("🧪 Testing Real Data with CPU Mode")
    print("=" * 60)
    
    # Create test config
    test_config = create_test_config()
    print(f"✅ Created test config: {test_config}")
    
    # Set environment variable to use test config
    os.environ['CONFIG_FILE'] = test_config
    
    try:
        # Import after setting config
        from csv_only_heavydb_workflow import CSVOnlyHeavyDBOptimizer
        
        optimizer = CSVOnlyHeavyDBOptimizer()
        
        print(f"\n📁 Input: {REAL_DATA_FILE}")
        print(f"📊 Portfolio size: 35 strategies")
        print("🖥️ Mode: CPU (GPU disabled for testing)")
        
        start_time = time.time()
        
        # Run optimization
        success = optimizer.run_optimization(REAL_DATA_FILE, 35)
        
        total_time = time.time() - start_time
        
        if success:
            print(f"\n✅ Optimization completed in {total_time:.1f}s")
        else:
            print(f"\n❌ Optimization failed after {total_time:.1f}s")
        
        return success, total_time
        
    finally:
        # Clean up test config
        if os.path.exists(test_config):
            os.remove(test_config)
        
        # Remove env variable
        if 'CONFIG_FILE' in os.environ:
            del os.environ['CONFIG_FILE']

def test_correlation_only():
    """Test just correlation calculation with real data"""
    print("\n🔬 Testing Correlation Calculation Only")
    print("=" * 60)
    
    import pandas as pd
    import numpy as np
    from lib.correlation_optimizer import calculate_correlation_cpu_fallback
    
    # Load data
    print("📊 Loading real data...")
    df = pd.read_csv(REAL_DATA_FILE)
    
    # Get strategy columns
    strategy_cols = [col for col in df.columns if col.startswith('SENSEX')]
    n_strategies = len(strategy_cols)
    
    print(f"✅ Found {n_strategies:,} strategies")
    
    # Test with different subset sizes
    test_sizes = [100, 500, 1000, 2000]
    
    for size in test_sizes:
        if size > n_strategies:
            break
            
        print(f"\n📊 Testing {size}x{size} correlation matrix...")
        subset_cols = strategy_cols[:size]
        
        start_time = time.time()
        corr_matrix = calculate_correlation_cpu_fallback(df, subset_cols)
        elapsed = time.time() - start_time
        
        if corr_matrix is not None:
            print(f"  ✅ Completed in {elapsed:.2f}s")
            print(f"  - Rate: {(size * size) / elapsed:,.0f} correlations/second")
            
            # Estimate time for full matrix
            full_estimate = elapsed * (n_strategies / size) ** 2
            print(f"  - Estimated for {n_strategies:,}x{n_strategies:,}: {full_estimate/60:.1f} minutes")
        else:
            print(f"  ❌ Failed")

def main():
    """Run CPU mode tests"""
    print("🚀 Real Data Testing - CPU Mode")
    print("=" * 80)
    print(f"File: {REAL_DATA_FILE}")
    print("Note: Temporarily disabling GPU requirements for testing")
    print("=" * 80)
    
    # Check file exists
    if not os.path.exists(REAL_DATA_FILE):
        print(f"❌ Real data file not found: {REAL_DATA_FILE}")
        return
    
    # Test correlation calculation
    test_correlation_only()
    
    # Test full workflow
    success, time = test_with_cpu_mode()
    
    print("\n" + "="*60)
    print("📊 Summary")
    print("="*60)
    print(f"Workflow test: {'✅ PASSED' if success else '❌ FAILED'}")
    print(f"Time: {time:.1f}s")

if __name__ == "__main__":
    main()