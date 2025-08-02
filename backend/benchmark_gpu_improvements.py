#!/usr/bin/env python3
"""
Benchmark script to measure improvements in GPU/HeavyDB implementation
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def benchmark_correlation_calculation():
    """Benchmark correlation calculation improvements"""
    print("🏁 Benchmarking Correlation Calculation Improvements")
    print("=" * 60)
    
    from lib.heavydb_connector import get_connection, load_strategy_data, calculate_correlations_gpu
    from lib.correlation_optimizer import estimate_correlation_memory_usage, get_correlation_config
    
    # Test different matrix sizes
    test_sizes = [100, 300, 500, 600, 800, 1000]
    results = []
    
    conn = get_connection()
    if not conn:
        print("❌ No HeavyDB connection available")
        # Test CPU fallback instead
        from lib.correlation_optimizer import calculate_correlation_cpu_fallback
        
        for n_strategies in test_sizes:
            print(f"\n📊 Testing {n_strategies}x{n_strategies} matrix (CPU)...")
            
            # Generate test data
            np.random.seed(42)
            data = np.random.randn(30, n_strategies) * 0.02
            columns = [f'SENSEX_{i:04d}' for i in range(n_strategies)]
            df = pd.DataFrame(data, columns=columns)
            
            # Benchmark
            start_time = time.time()
            corr_matrix = calculate_correlation_cpu_fallback(df, columns)
            elapsed = time.time() - start_time
            
            if corr_matrix is not None:
                results.append({
                    'size': n_strategies,
                    'time': elapsed,
                    'mode': 'CPU',
                    'success': True
                })
                print(f"✅ Completed in {elapsed:.2f}s")
            else:
                results.append({
                    'size': n_strategies,
                    'time': elapsed,
                    'mode': 'CPU',
                    'success': False
                })
                print(f"❌ Failed")
    else:
        # Test with GPU
        config = get_correlation_config()
        
        for n_strategies in test_sizes:
            print(f"\n📊 Testing {n_strategies}x{n_strategies} matrix (GPU)...")
            
            # Memory estimate
            mem_est = estimate_correlation_memory_usage(n_strategies)
            print(f"   Memory required: {mem_est['peak_memory_gb']:.2f} GB")
            
            # Generate test data
            np.random.seed(42)
            data = np.random.randn(30, n_strategies) * 0.02
            columns = [f'SENSEX_{i:04d}' for i in range(n_strategies)]
            df = pd.DataFrame(data, columns=columns)
            df['Date'] = pd.date_range('2024-01-01', periods=30, freq='D')
            
            # Load and calculate
            table_name = f"benchmark_{n_strategies}_{int(time.time())}"
            
            load_start = time.time()
            success = load_strategy_data(df, table_name, connection=conn)
            load_time = time.time() - load_start
            
            if success:
                corr_start = time.time()
                corr_matrix = calculate_correlations_gpu(table_name, connection=conn)
                corr_time = time.time() - corr_start
                
                total_time = load_time + corr_time
                
                if corr_matrix is not None:
                    results.append({
                        'size': n_strategies,
                        'time': total_time,
                        'corr_time': corr_time,
                        'load_time': load_time,
                        'mode': 'GPU',
                        'success': True
                    })
                    print(f"✅ Completed in {total_time:.2f}s (load: {load_time:.2f}s, corr: {corr_time:.2f}s)")
                else:
                    results.append({
                        'size': n_strategies,
                        'time': total_time,
                        'mode': 'GPU',
                        'success': False
                    })
                    print(f"❌ Failed after {total_time:.2f}s")
                
                # Clean up
                try:
                    cursor = conn.cursor()
                    cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
                except:
                    pass
        
        try:
            conn.close()
        except:
            pass
    
    # Print summary
    print("\n📊 Benchmark Summary")
    print("=" * 60)
    print(f"{'Size':>8} | {'Mode':>6} | {'Time (s)':>10} | {'Status':>10}")
    print("-" * 60)
    
    for result in results:
        status = "✅ Success" if result['success'] else "❌ Failed"
        print(f"{result['size']:>8} | {result['mode']:>6} | {result['time']:>10.2f} | {status:>10}")
    
    # Save results
    df_results = pd.DataFrame(results)
    output_file = f"/mnt/optimizer_share/output/correlation_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df_results.to_csv(output_file, index=False)
    print(f"\n💾 Results saved to: {output_file}")
    
    return results

def benchmark_large_dataset_workflow():
    """Benchmark full workflow with different dataset sizes"""
    print("\n🏁 Benchmarking Full Workflow Performance")
    print("=" * 60)
    
    # Check for test datasets
    input_dir = "/mnt/optimizer_share/input"
    test_files = []
    
    for file in os.listdir(input_dir):
        if file.startswith("synthetic_data_") and file.endswith(".csv"):
            # Extract size from filename
            parts = file.split('_')
            if 'x' in parts[2]:
                strategies = int(parts[2].split('x')[0])
                size_mb = os.path.getsize(os.path.join(input_dir, file)) / (1024**2)
                test_files.append((file, strategies, size_mb))
    
    if not test_files:
        print("❌ No test datasets found. Generate with:")
        print("   python3 generate_large_test_data.py --small")
        print("   python3 generate_large_test_data.py --medium")
        return
    
    # Sort by number of strategies
    test_files.sort(key=lambda x: x[1])
    
    print(f"\n📁 Found {len(test_files)} test dataset(s):")
    for filename, strategies, size_mb in test_files:
        print(f"   - {strategies:>6} strategies: {filename} ({size_mb:.1f} MB)")
    
    # Run benchmarks
    from csv_only_heavydb_workflow import OptimizationWorkflow
    
    results = []
    
    for filename, strategies, size_mb in test_files[:3]:  # Test up to 3 datasets
        print(f"\n🚀 Testing with {strategies} strategies...")
        
        workflow = OptimizationWorkflow()
        workflow.set_heavydb_enabled(True)
        
        input_path = os.path.join(input_dir, filename)
        
        start_time = time.time()
        try:
            result = workflow.run_optimization(
                input_file=input_path,
                portfolio_size=35
            )
            
            elapsed = time.time() - start_time
            
            results.append({
                'strategies': strategies,
                'file_size_mb': size_mb,
                'total_time': elapsed,
                'best_algorithm': result.get('best_algorithm', 'N/A'),
                'best_fitness': result.get('best_fitness', 0),
                'gpu_accelerated': result.get('heavydb_accelerated', False),
                'success': True
            })
            
            print(f"✅ Completed in {elapsed:.1f}s")
            print(f"   Best: {result.get('best_algorithm')} (fitness: {result.get('best_fitness'):.6f})")
            
        except Exception as e:
            elapsed = time.time() - start_time
            results.append({
                'strategies': strategies,
                'file_size_mb': size_mb,
                'total_time': elapsed,
                'success': False,
                'error': str(e)
            })
            print(f"❌ Failed after {elapsed:.1f}s: {e}")
    
    # Print summary
    print("\n📊 Workflow Benchmark Summary")
    print("=" * 60)
    print(f"{'Strategies':>10} | {'Size (MB)':>10} | {'Time (s)':>10} | {'Status':>10}")
    print("-" * 60)
    
    for result in results:
        status = "✅ Success" if result['success'] else "❌ Failed"
        print(f"{result['strategies']:>10} | {result['file_size_mb']:>10.1f} | {result['total_time']:>10.1f} | {status:>10}")
    
    return results

def main():
    """Run all benchmarks"""
    print("🚀 GPU/HeavyDB Improvements Benchmark Suite")
    print("Testing performance after fixes")
    print("=" * 60)
    
    # Run correlation benchmark
    corr_results = benchmark_correlation_calculation()
    
    # Run workflow benchmark
    workflow_results = benchmark_large_dataset_workflow()
    
    print("\n✅ All benchmarks completed!")
    
    # Check for improvements
    if corr_results:
        successful = [r for r in corr_results if r['success']]
        if successful:
            max_size = max(r['size'] for r in successful)
            print(f"\n🎯 Maximum successful correlation matrix size: {max_size}x{max_size}")
            
            if max_size >= 600:
                print("✅ Successfully handles matrices >500x500 (timeout issue FIXED)")
            else:
                print("⚠️ Still having issues with large matrices")

if __name__ == "__main__":
    main()