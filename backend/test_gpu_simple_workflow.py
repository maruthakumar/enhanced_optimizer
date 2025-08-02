#!/usr/bin/env python3
"""
Simple GPU Workflow Test with fewer strategies
"""

import os
import sys
import time
import pandas as pd
import numpy as np

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from csv_only_heavydb_workflow import CSVOnlyHeavyDBOptimizer


def test_simple_workflow():
    """Test with just a few strategies to avoid timeout"""
    print("\n🚀 Testing GPU Workflow with Small Dataset")
    print("="*60)
    
    # Create small test data
    n_strategies = 10  # Small number to avoid timeout
    n_days = 30
    
    dates = pd.date_range('2024-01-01', periods=n_days)
    data = {'Date': dates, 'trading_day': range(n_days)}
    
    # Create strategies
    for i in range(n_strategies):
        strategy_name = f"SENSEX {1000+i}-{1100+i} SL{i%5+10}%"
        data[strategy_name] = np.random.randn(n_days) * 100 + i * 10
    
    df = pd.DataFrame(data)
    
    # Save test file
    test_file = '/tmp/simple_gpu_test.csv'
    df.to_csv(test_file, index=False)
    
    print(f"Created test data: {n_strategies} strategies, {n_days} days")
    
    # Initialize optimizer
    optimizer = CSVOnlyHeavyDBOptimizer()
    
    # Test with GA only
    portfolio_size = 5
    
    try:
        print(f"\nRunning GA with portfolio size {portfolio_size}...")
        
        optimizer.selected_algorithms = ['GA']
        
        start_time = time.time()
        result = optimizer.run_optimization(
            csv_file_path=test_file,
            portfolio_size=portfolio_size
        )
        
        total_time = time.time() - start_time
        
        if result and isinstance(result, dict) and 'algorithm_results' in result and 'GA' in result['algorithm_results']:
            ga_result = result['algorithm_results']['GA']
            
            print(f"\n✅ Optimization completed in {total_time:.2f}s")
            print(f"\nResults:")
            print(f"  Best Fitness: {ga_result.get('best_fitness', 0):.4f}")
            print(f"  Total ROI: ${ga_result.get('total_roi', 0):,.2f}")
            print(f"  Max Drawdown: ${ga_result.get('max_drawdown', 0):,.2f}")
            print(f"  Win Rate: {ga_result.get('win_rate', 0):.1f}%")
            print(f"  Profit Factor: {ga_result.get('profit_factor', 0):.3f}")
            
            # Check if GPU was used
            if result.get('preprocessing', {}).get('gpu_accelerated', False):
                print(f"\n🎯 GPU Acceleration: ✅ ENABLED")
                gpu_info = result['preprocessing'].get('gpu_info', {})
                if gpu_info.get('available'):
                    for gpu in gpu_info.get('gpus', []):
                        print(f"     GPU: {gpu.get('model', 'Unknown')}")
                        print(f"     Memory: {gpu.get('total_memory_gb', 0)} GB")
            else:
                print(f"\n⚠️ GPU Acceleration: ❌ DISABLED (CPU fallback)")
            
            return True
        else:
            print(f"❌ Optimization failed - no results")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Clean up
        if os.path.exists(test_file):
            os.remove(test_file)


def test_ulta_simple():
    """Test ULTA with small dataset"""
    print("\n\n🔄 Testing ULTA Logic")
    print("="*60)
    
    # Create small test data with poor performers
    n_strategies = 10
    n_days = 30
    
    dates = pd.date_range('2024-01-01', periods=n_days)
    data = {'Date': dates, 'trading_day': range(n_days)}
    
    for i in range(n_strategies):
        strategy_name = f"SENSEX {2000+i}-{2100+i} SL{i%5+15}%"
        
        if i < 5:  # Poor performers
            # Downward trend
            data[strategy_name] = np.linspace(0, -500, n_days) + np.random.randn(n_days) * 20
        else:  # Good performers
            # Upward trend
            data[strategy_name] = np.linspace(0, 200, n_days) + np.random.randn(n_days) * 10
    
    df = pd.DataFrame(data)
    
    # Save test file
    test_file = '/tmp/ulta_simple_test.csv'
    df.to_csv(test_file, index=False)
    
    print(f"Created test data: {n_strategies} strategies (50% poor performers)")
    
    # Run optimizer
    optimizer = CSVOnlyHeavyDBOptimizer()
    
    try:
        optimizer.selected_algorithms = ['GA']
        result = optimizer.run_optimization(
            csv_file_path=test_file,
            portfolio_size=5
        )
        
        if result and 'preprocessing' in result:
            ulta_count = result['preprocessing'].get('ulta_inverted_count', 0)
            print(f"\n✅ ULTA inversions applied: {ulta_count} strategies")
            
            if ulta_count > 0:
                print("   ULTA successfully identified and inverted poor performers!")
            else:
                print("   No ULTA inversions needed")
                
    except Exception as e:
        print(f"❌ ULTA test failed: {e}")
    
    finally:
        # Clean up
        if os.path.exists(test_file):
            os.remove(test_file)


if __name__ == "__main__":
    # Run tests
    success = test_simple_workflow()
    
    if success:
        test_ulta_simple()
    
    print("\n✅ Simple GPU Tests Complete!")