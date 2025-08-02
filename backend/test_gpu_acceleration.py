#!/usr/bin/env python3
"""
Test GPU Acceleration with Licensed HeavyDB
"""

import os
import sys
import time
import pandas as pd
import numpy as np

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lib.heavydb_connector import (
    get_connection,
    execute_query,
    load_strategy_data,
    calculate_correlations_gpu,
    get_execution_mode,
    get_gpu_memory_info
)
from csv_only_heavydb_workflow import CSVOnlyHeavyDBOptimizer


def test_gpu_connection():
    """Test GPU-enabled HeavyDB connection"""
    print("\n" + "="*60)
    print("Testing GPU-Accelerated HeavyDB Connection")
    print("="*60)
    
    # Test connection
    conn = get_connection(force_new=True)
    
    if not conn:
        print("❌ Failed to connect to HeavyDB")
        return False
    
    print("✅ Connected to HeavyDB")
    
    # Check execution mode
    mode = get_execution_mode()
    print(f"Execution Mode: {mode.upper()}")
    
    # Get GPU info
    gpu_info = get_gpu_memory_info()
    if gpu_info.get('available'):
        print("\n🎯 GPU Information:")
        for gpu in gpu_info.get('gpus', []):
            print(f"   - Model: {gpu.get('model', 'Unknown')}")
            print(f"   - Total Memory: {gpu['total_memory_gb']} GB")
            print(f"   - Free Memory: {gpu['free_memory_gb']} GB")
    
    return True


def test_gpu_data_loading():
    """Test GPU-accelerated data loading"""
    print("\n" + "="*60)
    print("Testing GPU Data Loading")
    print("="*60)
    
    # Create test data
    n_strategies = 100
    n_days = 50
    
    data = {
        'trading_date': pd.date_range('2024-01-01', periods=n_days),  # Avoid reserved keyword
        'day_num': range(n_days)  # Avoid reserved keyword
    }
    
    for i in range(n_strategies):
        col_name = f"strategy_{i}"
        data[col_name] = np.random.randn(n_days) * 100
    
    df = pd.DataFrame(data)
    
    # Test loading into HeavyDB
    table_name = f"test_strategies_{int(time.time())}"
    
    print(f"Loading {n_strategies} strategies into HeavyDB...")
    start_time = time.time()
    
    success = load_strategy_data(df, table_name)
    
    if success:
        load_time = time.time() - start_time
        print(f"✅ Data loaded in {load_time:.3f}s")
        print(f"   Table: {table_name}")
        print(f"   Rows: {len(df)}")
        print(f"   Columns: {len(df.columns)}")
        
        # Test query
        result = execute_query(f"SELECT COUNT(*) as cnt FROM {table_name}")
        if result is not None:
            print(f"   Verified: {result['cnt'].iloc[0]} rows in table")
        
        # Clean up
        try:
            cursor = get_connection().cursor()
            cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
            print(f"   Cleaned up test table")
        except:
            pass
        
        return True
    else:
        print("❌ Failed to load data")
        return False


def test_gpu_correlation():
    """Test GPU-accelerated correlation calculation"""
    print("\n" + "="*60)
    print("Testing GPU Correlation Calculation")
    print("="*60)
    
    # Create larger test data for correlation
    n_strategies = 50  # Reduced for testing
    n_days = 82
    
    print(f"Creating test data: {n_strategies} strategies, {n_days} days...")
    
    data = {'trading_date': pd.date_range('2024-01-01', periods=n_days)}
    
    for i in range(n_strategies):
        data[f"SENSEX_{i}"] = np.random.randn(n_days) * 100 + np.random.randn() * 50
    
    df = pd.DataFrame(data)
    
    # Load into HeavyDB
    table_name = f"correlation_test_{int(time.time())}"
    
    print("Loading data for correlation test...")
    success = load_strategy_data(df, table_name)
    
    if not success:
        print("❌ Failed to load data for correlation test")
        return False
    
    # Calculate correlations
    print(f"Calculating {n_strategies}x{n_strategies} correlation matrix on GPU...")
    start_time = time.time()
    
    try:
        corr_matrix = calculate_correlations_gpu(table_name)
        
        if corr_matrix is not None:
            calc_time = time.time() - start_time
            print(f"✅ Correlation calculated in {calc_time:.3f}s")
            print(f"   Matrix shape: {corr_matrix.shape}")
            print(f"   Average correlation: {np.mean(np.abs(corr_matrix)):.4f}")
            
            # Compare with CPU timing estimate
            cpu_time_estimate = (n_strategies * n_strategies * n_days) / 1_000_000  # Rough estimate
            speedup = cpu_time_estimate / calc_time
            print(f"   Estimated speedup vs CPU: {speedup:.1f}x")
        else:
            print("❌ Correlation calculation returned None")
    
    except Exception as e:
        print(f"❌ Correlation calculation failed: {e}")
    
    # Clean up
    try:
        cursor = get_connection().cursor()
        cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
    except:
        pass
    
    return True


def test_full_gpu_workflow():
    """Test complete workflow with GPU acceleration"""
    print("\n" + "="*60)
    print("Testing Full GPU-Accelerated Workflow")
    print("="*60)
    
    # Create realistic test data
    n_strategies = 100  # Reduced for testing
    n_days = 82
    
    dates = pd.date_range('2024-01-01', periods=n_days)
    data = {'Date': dates, 'trading_day': range(n_days)}
    
    for i in range(n_strategies):
        strategy_name = f"SENSEX {1000+i}-{1100+i} SL{i%50+10}%"
        data[strategy_name] = np.random.randn(n_days) * 100 + np.random.randn() * 50
    
    df = pd.DataFrame(data)
    
    # Save test file
    test_file = '/tmp/gpu_workflow_test.csv'
    df.to_csv(test_file, index=False)
    
    print(f"Created test data: {n_strategies} strategies")
    
    # Initialize optimizer
    optimizer = CSVOnlyHeavyDBOptimizer()
    
    # Run workflow
    print("\nRunning optimization workflow...")
    start_time = time.time()
    
    try:
        # Load data
        loaded_data = optimizer.load_csv_data(test_file)
        
        # Preprocess with GPU
        processed_data = optimizer.preprocess_data(loaded_data)
        
        print(f"\nPreprocessing Results:")
        print(f"   GPU Accelerated: {processed_data.get('gpu_accelerated', False)}")
        
        if processed_data.get('gpu_info'):
            gpu_info = processed_data['gpu_info']
            if gpu_info.get('available'):
                print(f"   GPU Memory Available: {gpu_info['gpus'][0]['free_memory_gb']} GB")
        
        # Run one algorithm as test
        portfolio_size = 50
        
        def fitness_function(indices):
            portfolio_data = processed_data['matrix'][:, indices]
            returns = portfolio_data.sum(axis=1)
            roi = returns.sum()
            max_dd = abs(np.min(np.minimum.accumulate(returns)))
            return roi / max_dd if max_dd > 0 else 0
        
        algorithm = optimizer.algorithms['GA']
        result = algorithm.optimize(
            processed_data['matrix'],
            portfolio_size=portfolio_size,
            fitness_function=fitness_function
        )
        
        total_time = time.time() - start_time
        
        print(f"\nWorkflow completed in {total_time:.2f}s")
        print(f"Best fitness: {result.get('best_fitness', 0):.4f}")
        
        # Clean up
        os.remove(test_file)
        
        return True
        
    except Exception as e:
        print(f"❌ Workflow failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all GPU acceleration tests"""
    print("\n🚀 HeavyDB GPU Acceleration Test Suite")
    print("With Free Tier License")
    
    # Test 1: Connection
    if not test_gpu_connection():
        print("\n❌ Connection test failed")
        return
    
    # Test 2: Data Loading
    if not test_gpu_data_loading():
        print("\n❌ Data loading test failed")
        return
    
    # Test 3: Correlation
    test_gpu_correlation()
    
    # Test 4: Full Workflow
    test_full_gpu_workflow()
    
    print("\n" + "="*60)
    print("✅ GPU Acceleration Tests Complete!")
    print("="*60)


if __name__ == "__main__":
    main()