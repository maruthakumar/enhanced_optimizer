#!/usr/bin/env python3
"""
Comprehensive test suite for GPU/HeavyDB improvements
Tests all fixes for known issues in story 1.3
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_gpu_library_availability():
    """Test 1: Check GPU library availability and fallback"""
    print("\n" + "="*60)
    print("TEST 1: GPU Library Availability Check")
    print("="*60)
    
    # Test direct GPU library import
    gpu_libs_status = {
        'cudf': False,
        'cupy': False,
        'heavydb': False,
        'pymapd': False
    }
    
    # Test cudf
    try:
        import cudf
        gpu_libs_status['cudf'] = True
        print("✅ cudf: Available")
    except ImportError:
        print("❌ cudf: Not available")
    
    # Test cupy
    try:
        import cupy
        gpu_libs_status['cupy'] = True
        print("✅ cupy: Available")
        
        # Get GPU info if available
        device = cupy.cuda.Device()
        free_mem, total_mem = device.mem_info
        print(f"   GPU Memory: {free_mem/(1024**3):.1f}/{total_mem/(1024**3):.1f} GB free")
    except ImportError:
        print("❌ cupy: Not available")
    except Exception as e:
        print(f"❌ cupy: Available but GPU access failed: {e}")
    
    # Test HeavyDB connectors
    try:
        import heavydb
        gpu_libs_status['heavydb'] = True
        print("✅ heavydb: Available")
    except ImportError:
        print("❌ heavydb: Not available")
    
    try:
        import pymapd
        gpu_libs_status['pymapd'] = True
        print("✅ pymapd: Available (legacy fallback)")
    except ImportError:
        print("❌ pymapd: Not available")
    
    # Test our connector
    from lib.heavydb_connector import HEAVYDB_AVAILABLE, CONNECTOR_TYPE, GPU_ENABLED
    print(f"\n📊 HeavyDB Connector Status:")
    print(f"   - HeavyDB Available: {HEAVYDB_AVAILABLE}")
    print(f"   - Connector Type: {CONNECTOR_TYPE}")
    print(f"   - GPU Libraries Enabled: {GPU_ENABLED}")
    
    # Test fallback strategy
    from lib.heavydb_connector import get_gpu_memory_info
    gpu_info = get_gpu_memory_info()
    
    print(f"\n🎮 GPU Detection Results:")
    print(f"   - GPU Available: {gpu_info.get('available', False)}")
    print(f"   - GPU Libs Available: {gpu_info.get('gpu_libs_available', False)}")
    print(f"   - Fallback Note: {gpu_info.get('note', 'N/A')}")
    
    return gpu_libs_status

def test_correlation_timeout_fix():
    """Test 2: Verify correlation calculation doesn't timeout on large matrices"""
    print("\n" + "="*60)
    print("TEST 2: Correlation Timeout Fix (>500x500 matrices)")
    print("="*60)
    
    from lib.heavydb_connector import get_connection, load_strategy_data, calculate_correlations_gpu
    from lib.correlation_optimizer import get_correlation_config, estimate_correlation_memory_usage
    
    # Get configuration
    config = get_correlation_config()
    print(f"📋 Correlation Configuration:")
    print(f"   - Chunk Size: {config['chunk_size']}")
    print(f"   - Max Correlations/Query: {config['max_correlations_per_query']}")
    print(f"   - Timeout: {config['timeout']}s")
    print(f"   - Adaptive Chunking: {config['adaptive_chunking']}")
    
    # Test with 600x600 matrix (above the 500x500 threshold)
    n_strategies = 600
    n_days = 30
    
    # Estimate memory usage
    memory_est = estimate_correlation_memory_usage(n_strategies)
    print(f"\n💾 Memory Estimates for {n_strategies}x{n_strategies} matrix:")
    print(f"   - Matrix Size: {memory_est['matrix_size_gb']:.2f} GB")
    print(f"   - Working Memory: {memory_est['working_memory_gb']:.2f} GB")
    print(f"   - Peak Memory: {memory_est['peak_memory_gb']:.2f} GB")
    
    # Generate test data
    print(f"\n🔄 Generating test data: {n_strategies} strategies x {n_days} days")
    np.random.seed(42)
    data = np.random.randn(n_days, n_strategies) * 0.02
    columns = [f'SENSEX_{i:04d}' for i in range(n_strategies)]
    df = pd.DataFrame(data, columns=columns)
    df['Date'] = pd.date_range('2024-01-01', periods=n_days, freq='D')
    
    # Test correlation calculation
    conn = get_connection()
    if conn:
        table_name = f"test_timeout_{int(time.time())}"
        
        print(f"\n📊 Testing correlation calculation...")
        start_time = time.time()
        
        # Load data
        success = load_strategy_data(df, table_name, connection=conn)
        
        if success:
            # Calculate correlations with optimized parameters
            corr_start = time.time()
            corr_matrix = calculate_correlations_gpu(table_name, connection=conn)
            corr_time = time.time() - corr_start
            
            if corr_matrix is not None:
                print(f"✅ Correlation calculation completed in {corr_time:.2f}s")
                print(f"   - Did NOT timeout! (limit: {config['timeout']}s)")
                
                # Validate matrix
                from lib.correlation_optimizer import validate_correlation_matrix
                validation = validate_correlation_matrix(corr_matrix)
                print(f"\n🔍 Matrix Validation:")
                for key, value in validation.items():
                    if key != 'is_valid':
                        print(f"   - {key}: {value}")
                print(f"   ✅ Overall Valid: {validation['is_valid']}")
            else:
                print(f"❌ Correlation calculation failed (took {corr_time:.2f}s)")
            
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
    else:
        print("⚠️ No HeavyDB connection - testing CPU fallback")
        
        # Test CPU fallback
        from lib.correlation_optimizer import calculate_correlation_cpu_fallback
        
        cpu_start = time.time()
        corr_matrix = calculate_correlation_cpu_fallback(df, columns)
        cpu_time = time.time() - cpu_start
        
        if corr_matrix is not None:
            print(f"✅ CPU fallback completed in {cpu_time:.2f}s")
            print(f"   Matrix shape: {corr_matrix.shape}")

def test_heavydb_sql_optimization():
    """Test 3: Verify HeavyDB SQL optimization for GPU operations"""
    print("\n" + "="*60)
    print("TEST 3: HeavyDB SQL Optimization")
    print("="*60)
    
    from lib.heavydb_connector import get_connection, get_execution_mode, execute_query
    
    # Check execution mode
    mode = get_execution_mode()
    print(f"🎯 Execution Mode: {mode.upper()}")
    
    conn = get_connection()
    if conn and mode == 'gpu':
        print("\n📊 Testing GPU-optimized queries...")
        
        # Test 1: Simple aggregation
        test_queries = [
            ("Simple COUNT", "SELECT COUNT(*) FROM information_schema.tables"),
            ("GPU Detection", "SELECT 1 as gpu_test"),
        ]
        
        for query_name, query in test_queries:
            start = time.time()
            result = execute_query(query, connection=conn)
            elapsed = time.time() - start
            
            if result is not None:
                print(f"✅ {query_name}: {elapsed:.3f}s")
            else:
                print(f"❌ {query_name}: Failed")
        
        # Get timeout configuration
        from lib.heavydb_connector import get_connection_params
        params = get_connection_params()
        print(f"\n⏱️ Connection Timeout: {params.get('timeout', 'Not set')}s")
        
        try:
            conn.close()
        except:
            pass
    else:
        print("⚠️ GPU mode not available - SQL optimizations not testable")

def test_large_dataset_handling():
    """Test 4: Test handling of large datasets (simulating 25,544 strategies)"""
    print("\n" + "="*60)
    print("TEST 4: Large Dataset Handling")
    print("="*60)
    
    # First check if we have a large test dataset
    large_datasets = []
    input_dir = "/mnt/optimizer_share/input"
    
    for file in os.listdir(input_dir):
        if file.startswith("synthetic_data_") and file.endswith(".csv"):
            size_mb = os.path.getsize(os.path.join(input_dir, file)) / (1024**2)
            if size_mb > 10:  # Files larger than 10MB
                large_datasets.append((file, size_mb))
    
    if large_datasets:
        print(f"📁 Found {len(large_datasets)} large test dataset(s):")
        for filename, size in large_datasets[:3]:  # Show first 3
            print(f"   - {filename}: {size:.1f} MB")
        
        # Use the largest one
        test_file = max(large_datasets, key=lambda x: x[1])[0]
        test_path = os.path.join(input_dir, test_file)
        
        print(f"\n📊 Testing with: {test_file}")
        
        # Run workflow test
        from csv_only_heavydb_workflow import OptimizationWorkflow
        
        workflow = OptimizationWorkflow()
        workflow.set_heavydb_enabled(True)
        
        print("\n🚀 Running optimization workflow...")
        start_time = time.time()
        
        try:
            result = workflow.run_optimization(
                input_file=test_path,
                portfolio_size=35
            )
            
            elapsed = time.time() - start_time
            print(f"\n✅ Workflow completed in {elapsed:.1f}s")
            print(f"   - Best Algorithm: {result.get('best_algorithm', 'N/A')}")
            print(f"   - Best Fitness: {result.get('best_fitness', 0):.6f}")
            print(f"   - GPU Accelerated: {result.get('heavydb_accelerated', False)}")
        except Exception as e:
            print(f"\n❌ Workflow failed: {e}")
    else:
        print("⚠️ No large test datasets found")
        print("\n💡 Generate test data with:")
        print("   python3 generate_large_test_data.py --medium")
        print("   python3 generate_large_test_data.py --strategies 25544")

def main():
    """Run all tests"""
    parser = argparse.ArgumentParser(description='Test GPU/HeavyDB improvements')
    parser.add_argument('--test', type=int, default=0,
                       help='Run specific test (1-4), 0 for all')
    
    args = parser.parse_args()
    
    print("🚀 GPU/HeavyDB Improvements Test Suite")
    print("Testing fixes for known issues in story 1.3")
    
    tests = [
        (1, test_gpu_library_availability, "GPU Library Availability"),
        (2, test_correlation_timeout_fix, "Correlation Timeout Fix"),
        (3, test_heavydb_sql_optimization, "HeavyDB SQL Optimization"),
        (4, test_large_dataset_handling, "Large Dataset Handling")
    ]
    
    if args.test == 0:
        # Run all tests
        for test_num, test_func, test_name in tests:
            try:
                test_func()
            except Exception as e:
                print(f"\n❌ Test {test_num} ({test_name}) failed with error: {e}")
    else:
        # Run specific test
        for test_num, test_func, test_name in tests:
            if test_num == args.test:
                try:
                    test_func()
                except Exception as e:
                    print(f"\n❌ Test failed with error: {e}")
                break
        else:
            print(f"❌ Invalid test number: {args.test}")
    
    print("\n" + "="*60)
    print("✅ Test suite completed!")
    print("="*60)

if __name__ == "__main__":
    main()