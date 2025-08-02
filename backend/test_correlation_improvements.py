#!/usr/bin/env python3
"""
Test script for improved correlation calculation and GPU library detection
"""

import os
import sys
import time
import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lib.heavydb_connector import (
    get_connection,
    load_strategy_data,
    calculate_correlations_gpu,
    get_gpu_memory_info,
    get_execution_mode
)

def test_gpu_library_detection():
    """Test GPU library availability detection"""
    print("\n🔍 Testing GPU Library Detection...")
    
    # Get GPU memory info which now includes library availability
    gpu_info = get_gpu_memory_info()
    
    print(f"GPU Available: {gpu_info.get('available', False)}")
    print(f"GPU Libraries (cudf/cupy) Available: {gpu_info.get('gpu_libs_available', False)}")
    
    if gpu_info.get('available'):
        for gpu in gpu_info.get('gpus', []):
            print(f"  - Model: {gpu['model']}")
            print(f"  - Total Memory: {gpu['total_memory_gb']:.2f} GB")
            print(f"  - Free Memory: {gpu['free_memory_gb']:.2f} GB")
            print(f"  - Usage: {gpu['usage_percent']:.1f}%")
    
    print(f"Note: {gpu_info.get('note', '')}")
    
    return gpu_info

def test_correlation_calculation_small():
    """Test correlation calculation with small dataset"""
    print("\n📊 Testing Correlation Calculation (Small Dataset)...")
    
    # Create test data
    n_strategies = 100
    n_days = 30
    
    # Generate random returns data
    np.random.seed(42)
    data = np.random.randn(n_days, n_strategies) * 0.02  # 2% daily volatility
    
    # Create DataFrame
    columns = [f'SENSEX_{i:04d}' for i in range(n_strategies)]
    dates = pd.date_range('2024-01-01', periods=n_days, freq='D')
    df = pd.DataFrame(data, columns=columns)
    df['Date'] = dates
    
    # Test correlation calculation
    conn = get_connection()
    if conn:
        table_name = f"test_corr_{int(time.time())}"
        
        # Load data
        print(f"Loading {n_strategies} strategies x {n_days} days into HeavyDB...")
        success = load_strategy_data(df, table_name, connection=conn)
        
        if success:
            # Calculate correlations with new parameters
            start_time = time.time()
            corr_matrix = calculate_correlations_gpu(
                table_name, 
                connection=conn,
                chunk_size=25,  # Smaller chunks
                max_query_size=300  # Fewer correlations per query
            )
            elapsed = time.time() - start_time
            
            if corr_matrix is not None:
                print(f"✅ Correlation matrix calculated in {elapsed:.2f}s")
                print(f"Matrix shape: {corr_matrix.shape}")
                print(f"Matrix diagonal (should be ~1.0): {np.diag(corr_matrix)[:5]}...")
                
                # Verify it's a valid correlation matrix
                is_symmetric = np.allclose(corr_matrix, corr_matrix.T)
                diagonal_ones = np.allclose(np.diag(corr_matrix), 1.0)
                
                print(f"Is symmetric: {is_symmetric}")
                print(f"Diagonal values ~1.0: {diagonal_ones}")
            else:
                print("❌ Correlation calculation failed")
            
            # Clean up
            try:
                cursor = conn.cursor()
                cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
            except:
                pass
        
        # Close connection
        try:
            conn.close()
        except:
            pass
    else:
        print("❌ No HeavyDB connection available")

def test_correlation_calculation_large():
    """Test correlation calculation with larger dataset"""
    print("\n📊 Testing Correlation Calculation (Large Dataset)...")
    
    # Create test data
    n_strategies = 500  # Test with 500x500 matrix
    n_days = 82
    
    # Generate random returns data
    np.random.seed(42)
    data = np.random.randn(n_days, n_strategies) * 0.02  # 2% daily volatility
    
    # Create DataFrame
    columns = [f'SENSEX_{i:04d}' for i in range(n_strategies)]
    dates = pd.date_range('2024-01-04', periods=n_days, freq='D')
    df = pd.DataFrame(data, columns=columns)
    df['Date'] = dates
    
    # Test correlation calculation
    conn = get_connection()
    if conn:
        table_name = f"test_corr_large_{int(time.time())}"
        
        # Load data
        print(f"Loading {n_strategies} strategies x {n_days} days into HeavyDB...")
        load_start = time.time()
        success = load_strategy_data(df, table_name, connection=conn)
        load_time = time.time() - load_start
        
        if success:
            print(f"✅ Data loaded in {load_time:.2f}s")
            
            # Calculate correlations with optimized parameters
            start_time = time.time()
            corr_matrix = calculate_correlations_gpu(
                table_name, 
                connection=conn,
                chunk_size=20,  # Even smaller chunks for large matrix
                max_query_size=200  # Limit correlations per query
            )
            elapsed = time.time() - start_time
            
            if corr_matrix is not None:
                print(f"✅ Large correlation matrix calculated in {elapsed:.2f}s")
                print(f"Matrix shape: {corr_matrix.shape}")
                
                # Check if we avoided timeout
                if elapsed < 300:  # 5 minute timeout
                    print("✅ Completed within timeout limit!")
                else:
                    print("⚠️ Took longer than expected, but completed")
            else:
                print("❌ Large correlation calculation failed (likely timeout)")
            
            # Clean up
            try:
                cursor = conn.cursor()
                cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
            except:
                pass
        
        # Close connection
        try:
            conn.close()
        except:
            pass
    else:
        print("❌ No HeavyDB connection available for large test")

def test_timeout_configuration():
    """Test timeout configuration"""
    print("\n⏱️ Testing Timeout Configuration...")
    
    # Check if timeout is configured
    from lib.heavydb_connector import get_connection_params
    
    params = get_connection_params()
    print(f"Configured timeout: {params.get('timeout', 'Not set')} seconds")
    
    # Test with HEAVYDB_TIMEOUT environment variable
    os.environ['HEAVYDB_TIMEOUT'] = '600'  # 10 minutes
    params_with_timeout = get_connection_params()
    print(f"Timeout with env var: {params_with_timeout.get('timeout', 'Not set')} seconds")
    
    # Reset
    if 'HEAVYDB_TIMEOUT' in os.environ:
        del os.environ['HEAVYDB_TIMEOUT']

def main():
    """Run all tests"""
    print("🚀 Testing Correlation Improvements and GPU Library Detection")
    print("=" * 60)
    
    # Test GPU library detection
    gpu_info = test_gpu_library_detection()
    
    # Test timeout configuration
    test_timeout_configuration()
    
    # Only run correlation tests if HeavyDB is available
    if get_execution_mode() == 'gpu':
        test_correlation_calculation_small()
        test_correlation_calculation_large()
    else:
        print("\n⚠️ Skipping correlation tests - HeavyDB not available")
    
    print("\n✅ All tests completed!")

if __name__ == "__main__":
    main()