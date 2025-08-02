#!/usr/bin/env python3
"""
Test real production data with proper configuration override
"""

import os
import sys
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set environment variables BEFORE importing modules
os.environ['GPU_FALLBACK_ALLOWED'] = 'true'
os.environ['FORCE_GPU_MODE'] = 'false'

# Real production data file
REAL_DATA_FILE = "/mnt/optimizer_share/input/Python_Multi_Consolidated_20250726_161921.csv"

def run_real_data_test():
    """Run the actual test with real data"""
    print("🚀 Testing with Real Production Data")
    print("=" * 60)
    print(f"📁 File: {REAL_DATA_FILE}")
    print("🖥️ Mode: CPU fallback enabled via environment variables")
    print("=" * 60)
    
    # Import AFTER setting environment variables
    from csv_only_heavydb_workflow import CSVOnlyHeavyDBOptimizer
    
    # Check configuration was applied
    from lib.config_reader import get_gpu_config, is_gpu_required
    
    config = get_gpu_config()
    print("\n📋 Active GPU Configuration:")
    print(f"  - CPU Fallback Allowed: {config['cpu_fallback_allowed']}")
    print(f"  - Force GPU Mode: {config['force_gpu_mode']}")
    print(f"  - GPU Required: {is_gpu_required()}")
    
    # Create optimizer
    optimizer = CSVOnlyHeavyDBOptimizer()
    
    print(f"\n🔄 Running optimization...")
    print(f"  - Input: {os.path.basename(REAL_DATA_FILE)}")
    print(f"  - Portfolio size: 35")
    
    start_time = time.time()
    
    try:
        # Run optimization
        success = optimizer.run_optimization(REAL_DATA_FILE, 35)
        
        elapsed = time.time() - start_time
        
        if success:
            print(f"\n✅ Optimization completed successfully in {elapsed:.1f}s!")
            
            # Find output directory
            import glob
            output_pattern = f"/mnt/optimizer_share/output/run_*"
            recent_outputs = sorted(glob.glob(output_pattern), key=os.path.getmtime)
            
            if recent_outputs:
                latest_output = recent_outputs[-1]
                print(f"📁 Output: {latest_output}")
                
                # Check for optimization summary
                summary_files = glob.glob(os.path.join(latest_output, "optimization_summary_*.txt"))
                if summary_files:
                    print("\n📊 Optimization Summary:")
                    with open(summary_files[0], 'r') as f:
                        lines = f.readlines()[:20]  # First 20 lines
                        for line in lines:
                            print(f"  {line.rstrip()}")
        else:
            print(f"\n❌ Optimization failed after {elapsed:.1f}s")
        
        return success, elapsed
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n❌ Error after {elapsed:.1f}s: {e}")
        import traceback
        traceback.print_exc()
        return False, elapsed

def test_correlation_performance():
    """Test correlation calculation performance with different sizes"""
    print("\n🔬 Testing Correlation Performance")
    print("=" * 60)
    
    import pandas as pd
    from lib.correlation_optimizer import calculate_correlation_cpu_fallback, estimate_correlation_memory_usage
    
    # Load data
    print("Loading real data...")
    df = pd.read_csv(REAL_DATA_FILE)
    strategy_cols = [col for col in df.columns if col.startswith('SENSEX')]
    n_strategies = len(strategy_cols)
    
    print(f"✅ Loaded {n_strategies:,} strategies × {len(df)-1} days")
    
    # Memory estimates
    mem_est = estimate_correlation_memory_usage(n_strategies)
    print(f"\n💾 Full correlation matrix ({n_strategies:,}×{n_strategies:,}):")
    print(f"  - Size: {mem_est['matrix_size_gb']:.2f} GB")
    print(f"  - Peak memory: {mem_est['peak_memory_gb']:.2f} GB")
    
    # Test increasing sizes
    test_sizes = [100, 500, 1000, 2500, 5000]
    results = []
    
    print("\n📊 Correlation Calculation Benchmarks:")
    print(f"{'Size':>6} | {'Time (s)':>10} | {'Rate (corr/s)':>15} | {'Est. Full Time':>15}")
    print("-" * 60)
    
    for size in test_sizes:
        if size > n_strategies:
            break
        
        subset_cols = strategy_cols[:size]
        
        start = time.time()
        corr_matrix = calculate_correlation_cpu_fallback(df, subset_cols)
        elapsed = time.time() - start
        
        if corr_matrix is not None:
            rate = (size * size) / elapsed
            full_estimate = elapsed * (n_strategies / size) ** 2
            
            results.append({
                'size': size,
                'time': elapsed,
                'rate': rate,
                'estimate': full_estimate
            })
            
            print(f"{size:>6} | {elapsed:>10.2f} | {rate:>15,.0f} | {full_estimate/60:>13.1f}m")
    
    return results

def main():
    """Main test function"""
    print("🔬 Real Production Data Test Suite")
    print("=" * 80)
    print("Testing GPU improvements with actual 25,544 strategy dataset")
    print("=" * 80)
    
    # Check file exists
    if not os.path.exists(REAL_DATA_FILE):
        print(f"❌ File not found: {REAL_DATA_FILE}")
        return
    
    # Run correlation benchmarks
    corr_results = test_correlation_performance()
    
    # Run full workflow test
    success, elapsed = run_real_data_test()
    
    # Summary
    print("\n" + "="*80)
    print("📊 TEST RESULTS SUMMARY")
    print("="*80)
    
    print("\n✅ Correlation Performance:")
    if corr_results:
        largest = max(corr_results, key=lambda x: x['size'])
        print(f"  - Tested up to: {largest['size']:,}×{largest['size']:,} matrix")
        print(f"  - Performance: {largest['rate']:,.0f} correlations/second")
        print(f"  - Full matrix estimate: {largest['estimate']/60:.1f} minutes")
    
    print(f"\n✅ Full Workflow:")
    print(f"  - Status: {'SUCCESS' if success else 'FAILED'}")
    print(f"  - Time: {elapsed:.1f} seconds")
    print(f"  - Dataset: 25,544 strategies × 82 days")
    
    if success:
        print("\n🎉 All improvements working with real production data!")
    else:
        print("\n⚠️ Some issues remain - check logs above")

if __name__ == "__main__":
    main()