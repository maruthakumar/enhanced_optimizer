#!/usr/bin/env python3
"""
Test GPU improvements with REAL production data
Using: Python_Multi_Consolidated_20250726_161921.csv
"""

import os
import sys
import time
import pandas as pd
import numpy as np
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from csv_only_heavydb_workflow import CSVOnlyHeavyDBOptimizer
from lib.heavydb_connector import (
    get_connection, 
    load_strategy_data, 
    calculate_correlations_gpu,
    get_gpu_memory_info,
    get_execution_mode
)
from lib.correlation_optimizer import estimate_correlation_memory_usage

# Real production data file
REAL_DATA_FILE = "/mnt/optimizer_share/input/Python_Multi_Consolidated_20250726_161921.csv"

def analyze_real_data():
    """Analyze the real production dataset"""
    print("📊 Analyzing Real Production Data")
    print("=" * 60)
    
    print(f"📁 File: {REAL_DATA_FILE}")
    
    # Load data
    print("\n🔄 Loading data...")
    start_time = time.time()
    df = pd.read_csv(REAL_DATA_FILE)
    load_time = time.time() - start_time
    
    print(f"✅ Data loaded in {load_time:.2f}s")
    
    # Get basic stats
    n_rows, n_cols = df.shape
    n_strategies = n_cols - 2  # Exclude Date and Day columns
    
    print(f"\n📈 Dataset Statistics:")
    print(f"  - Trading days: {n_rows - 1}")
    print(f"  - Total strategies: {n_strategies:,}")
    print(f"  - Data points: {(n_rows - 1) * n_strategies:,}")
    print(f"  - Memory usage: {df.memory_usage(deep=True).sum() / (1024**2):.1f} MB")
    
    # Date range
    dates = pd.to_datetime(df['Date'])
    print(f"  - Date range: {dates.min()} to {dates.max()}")
    
    # Sample some strategy statistics
    strategy_cols = [col for col in df.columns if col.startswith('SENSEX')]
    sample_strategies = strategy_cols[:10]
    
    print(f"\n📊 Sample Strategy Performance (first 10):")
    for col in sample_strategies:
        strategy_data = df[col].dropna()
        if len(strategy_data) > 0:
            total_pnl = strategy_data.sum()
            mean_daily = strategy_data.mean()
            print(f"  - {col[:30]:30} | Total P&L: ${total_pnl:>12,.2f} | Daily Avg: ${mean_daily:>8,.2f}")
    
    return df, n_strategies

def test_correlation_with_real_data(df, n_strategies):
    """Test correlation calculation with real 25,544 strategies"""
    print("\n🔬 Testing Correlation Calculation with Real Data")
    print("=" * 60)
    
    # Memory estimate
    mem_est = estimate_correlation_memory_usage(n_strategies)
    print(f"\n💾 Memory Requirements for {n_strategies:,}x{n_strategies:,} correlation matrix:")
    print(f"  - Matrix size: {mem_est['matrix_size_gb']:.2f} GB")
    print(f"  - Working memory: {mem_est['working_memory_gb']:.2f} GB")
    print(f"  - Peak memory: {mem_est['peak_memory_gb']:.2f} GB")
    
    # Check GPU availability
    gpu_info = get_gpu_memory_info()
    print(f"\n🎮 GPU Status:")
    print(f"  - GPU Available: {gpu_info.get('available', False)}")
    print(f"  - GPU Libraries: {gpu_info.get('gpu_libs_available', False)}")
    
    if gpu_info.get('available') and gpu_info.get('gpus'):
        gpu = gpu_info['gpus'][0]
        print(f"  - GPU Model: {gpu['model']}")
        print(f"  - GPU Memory: {gpu['total_memory_gb']:.1f} GB")
        
        if gpu['total_memory_gb'] >= mem_est['peak_memory_gb']:
            print(f"  ✅ Sufficient GPU memory for full correlation matrix")
        else:
            print(f"  ⚠️ GPU memory may be insufficient, will use chunking")
    
    # Test correlation calculation
    conn = get_connection()
    if conn:
        table_name = f"real_data_test_{int(time.time())}"
        
        print(f"\n🔄 Loading {n_strategies:,} strategies into HeavyDB...")
        load_start = time.time()
        success = load_strategy_data(df, table_name, connection=conn)
        load_time = time.time() - load_start
        
        if success:
            print(f"✅ Data loaded in {load_time:.2f}s")
            
            # Calculate correlations
            print(f"\n📊 Calculating {n_strategies:,}x{n_strategies:,} correlation matrix...")
            print("⏳ This may take several minutes for 25,544 strategies...")
            
            corr_start = time.time()
            corr_matrix = calculate_correlations_gpu(table_name, connection=conn)
            corr_time = time.time() - corr_start
            
            if corr_matrix is not None:
                print(f"\n✅ Correlation calculation completed in {corr_time:.2f}s!")
                print(f"  - Matrix shape: {corr_matrix.shape}")
                print(f"  - Calculation rate: {(n_strategies * n_strategies) / corr_time:,.0f} correlations/second")
                
                # Validate
                from lib.correlation_optimizer import validate_correlation_matrix
                validation = validate_correlation_matrix(corr_matrix)
                
                print(f"\n🔍 Correlation Matrix Validation:")
                print(f"  - Valid: {validation['is_valid']}")
                print(f"  - Symmetric: {validation['is_symmetric']}")
                print(f"  - Diagonal ones: {validation['diagonal_ones']}")
                print(f"  - Value range: [{validation['min_correlation']:.3f}, {validation['max_correlation']:.3f}]")
                print(f"  - High correlation pairs (|r| > 0.7): {validation['high_correlation_pairs']:,}")
                
                success = True
            else:
                print(f"\n❌ Correlation calculation failed after {corr_time:.2f}s")
                success = False
            
            # Clean up
            try:
                cursor = conn.cursor()
                cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
            except:
                pass
        else:
            print("❌ Failed to load data into HeavyDB")
            success = False
        
        try:
            conn.close()
        except:
            pass
            
        return success, corr_time if 'corr_time' in locals() else 0
    else:
        print("\n⚠️ No HeavyDB connection - testing CPU fallback")
        
        # Test CPU fallback with subset
        print("📊 Testing CPU correlation with subset (first 1000 strategies)...")
        strategy_cols = [col for col in df.columns if col.startswith('SENSEX')][:1000]
        
        from lib.correlation_optimizer import calculate_correlation_cpu_fallback
        
        cpu_start = time.time()
        corr_matrix = calculate_correlation_cpu_fallback(df, strategy_cols)
        cpu_time = time.time() - cpu_start
        
        if corr_matrix is not None:
            print(f"✅ CPU correlation (1000x1000) completed in {cpu_time:.2f}s")
            
            # Estimate time for full matrix
            full_time_estimate = cpu_time * (n_strategies / 1000) ** 2
            print(f"⏱️ Estimated time for full {n_strategies:,}x{n_strategies:,}: {full_time_estimate/60:.1f} minutes")
        
        return False, 0

def test_full_workflow_with_real_data():
    """Test complete optimization workflow with real data"""
    print("\n🚀 Testing Full Optimization Workflow with Real Data")
    print("=" * 60)
    
    optimizer = CSVOnlyHeavyDBOptimizer()
    
    print(f"📁 Input: {REAL_DATA_FILE}")
    print(f"📊 Portfolio size: 35 strategies")
    
    start_time = time.time()
    
    try:
        # Run optimization
        success = optimizer.run_optimization(
            REAL_DATA_FILE,
            35
        )
        
        total_time = time.time() - start_time
        
        if success:
            print(f"\n✅ Workflow completed in {total_time:.1f}s")
        else:
            print(f"\n❌ Workflow failed after {total_time:.1f}s")
        
        return success, total_time
        
    except Exception as e:
        total_time = time.time() - start_time
        print(f"\n❌ Workflow failed after {total_time:.1f}s")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False, total_time

def main():
    """Run all tests with real production data"""
    print("🔬 Testing GPU Improvements with REAL Production Data")
    print("=" * 80)
    print(f"File: {REAL_DATA_FILE}")
    print("Expected: 25,544 strategies × 82 trading days")
    print("=" * 80)
    
    # Check file exists
    if not os.path.exists(REAL_DATA_FILE):
        print(f"❌ Real data file not found: {REAL_DATA_FILE}")
        return
    
    # Analyze data
    df, n_strategies = analyze_real_data()
    
    # Test correlation with real data
    print("\n" + "="*60)
    corr_success, corr_time = test_correlation_with_real_data(df, n_strategies)
    
    # Test full workflow
    print("\n" + "="*60)
    workflow_success, workflow_time = test_full_workflow_with_real_data()
    
    # Summary
    print("\n" + "="*80)
    print("📊 TEST SUMMARY - REAL PRODUCTION DATA")
    print("="*80)
    print(f"Dataset: {n_strategies:,} strategies × {len(df)-1} trading days")
    print(f"\nCorrelation Test:")
    print(f"  - Status: {'✅ PASSED' if corr_success else '❌ FAILED'}")
    if corr_success:
        print(f"  - Time: {corr_time:.2f}s")
        print(f"  - Matrix: {n_strategies:,}×{n_strategies:,} = {n_strategies**2:,} correlations")
    
    print(f"\nFull Workflow Test:")
    print(f"  - Status: {'✅ PASSED' if workflow_success else '❌ FAILED'}")
    print(f"  - Time: {workflow_time:.1f}s")
    
    print("\n🏁 Real data testing completed!")

if __name__ == "__main__":
    main()