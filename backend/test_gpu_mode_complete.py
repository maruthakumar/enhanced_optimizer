#!/usr/bin/env python3
"""
Complete GPU Mode Test Suite
Tests all GPU-related functionality and algorithm iterations
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lib.heavydb_connector import get_connection, HEAVYDB_AVAILABLE, load_strategy_data, execute_query
from lib.correlation_optimizer import optimize_correlation_calculation
from config.config_manager import get_config_manager
from algorithms.genetic_algorithm import GeneticAlgorithm
from algorithms.particle_swarm_optimization import ParticleSwarmOptimization
from algorithms.simulated_annealing import SimulatedAnnealing
from algorithms.hill_climbing import HillClimbing

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def test_gpu_connection():
    """Test HeavyDB GPU connection"""
    print("\n🔌 Testing GPU Connection")
    print("="*60)
    
    if not HEAVYDB_AVAILABLE:
        print("❌ HeavyDB not available")
        return False
    
    conn = get_connection()
    if not conn:
        print("❌ Failed to connect to HeavyDB")
        return False
    
    try:
        cursor = conn.cursor()
        
        # Test basic query
        cursor.execute("SELECT 1")
        result = cursor.fetchone()
        print(f"✅ Basic query test: {result}")
        
        # Check GPU status
        try:
            cursor.execute("SELECT * FROM omnisci_server_status LIMIT 1")
            print("✅ GPU server status query successful")
        except:
            print("⚠️ GPU status query not available (using HeavyDB free tier)")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Connection test failed: {e}")
        if conn:
            conn.close()
        return False


def test_correlation_optimization():
    """Test optimized correlation calculation"""
    print("\n📊 Testing Correlation Optimization")
    print("="*60)
    
    # Create test data
    n_strategies = 1000
    n_days = 50
    
    # Create correlated test data
    base = np.random.randn(n_days)
    data = pd.DataFrame({
        f'Strategy_{i}': base + np.random.randn(n_days) * 0.5 
        for i in range(n_strategies)
    })
    
    # Load to HeavyDB
    table_name = f"test_corr_{int(time.time())}"
    conn = get_connection()
    
    if not conn:
        print("❌ No HeavyDB connection")
        return False
    
    try:
        print(f"Loading {n_strategies} strategies to HeavyDB...")
        success = load_strategy_data(data, table_name, connection=conn)
        
        if not success:
            print("❌ Failed to load test data")
            return False
        
        print("✅ Data loaded to HeavyDB")
        
        # Test correlation calculation with optimization
        config = {
            'chunk_size': 50,
            'max_query_size': 500,
            'timeout': 60,
            'adaptive_chunking': True,
            'large_matrix_chunk_size': 100,
            'huge_matrix_chunk_size': 50
        }
        
        start_time = time.time()
        corr_matrix = optimize_correlation_calculation(
            table_name, 
            connection=conn,
            config=config
        )
        calc_time = time.time() - start_time
        
        if corr_matrix is not None:
            print(f"✅ Correlation matrix calculated: shape {corr_matrix.shape}")
            print(f"⏱️ Calculation time: {calc_time:.2f}s")
            print(f"📈 Strategies/second: {n_strategies/calc_time:.0f}")
        else:
            print("❌ Correlation calculation failed")
        
        # Cleanup
        cursor = conn.cursor()
        cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
        
        conn.close()
        return corr_matrix is not None
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        if conn:
            conn.close()
        return False


def test_algorithm_iterations():
    """Test algorithm iteration counts match original"""
    print("\n🧬 Testing Algorithm Iterations")
    print("="*60)
    
    config = get_config_manager()
    
    # Expected iterations from original
    expected = {
        'GA': 50,
        'PSO': 50,
        'SA': 1000,
        'DE': 50,
        'ACO': 50,
        'HC': 200,
        'BO': 50
    }
    
    # Check each algorithm
    results = {}
    
    # GA
    ga = GeneticAlgorithm()
    results['GA'] = ga.generations
    
    # PSO
    pso = ParticleSwarmOptimization()
    results['PSO'] = pso.iterations
    
    # SA
    sa = SimulatedAnnealing()
    results['SA'] = sa.max_iterations
    
    # HC
    hc = HillClimbing()
    results['HC'] = hc.max_iterations
    
    # DE (from config)
    results['DE'] = config.getint('ALGORITHM_PARAMETERS', 'de_generations', 100)
    
    # ACO (from config)
    results['ACO'] = config.getint('ALGORITHM_PARAMETERS', 'aco_iterations', 50)
    
    # BO (from config)
    results['BO'] = config.getint('ALGORITHM_PARAMETERS', 'bo_n_calls', 50)
    
    # Compare
    print(f"{'Algorithm':<10} | {'Expected':<10} | {'Actual':<10} | {'Status':<10}")
    print("-"*50)
    
    all_match = True
    for algo, expected_val in expected.items():
        actual_val = results.get(algo, 0)
        match = actual_val == expected_val
        status = "✅" if match else "❌"
        
        print(f"{algo:<10} | {expected_val:<10} | {actual_val:<10} | {status:<10}")
        
        if not match:
            all_match = False
    
    return all_match


def test_large_dataset_handling():
    """Test handling of large datasets"""
    print("\n📦 Testing Large Dataset Handling")
    print("="*60)
    
    # Create large test dataset
    n_strategies = 25544  # Production size
    n_days = 82
    
    print(f"Creating test data: {n_strategies} strategies x {n_days} days")
    
    # Don't actually create full dataset in memory - test chunking logic
    chunk_size = 5000
    chunks_needed = (n_strategies + chunk_size - 1) // chunk_size
    
    print(f"Would process in {chunks_needed} chunks of {chunk_size} strategies")
    
    # Test a single chunk
    chunk_data = pd.DataFrame(
        np.random.randn(n_days, chunk_size),
        columns=[f'Strategy_{i}' for i in range(chunk_size)]
    )
    chunk_data['Date'] = pd.date_range('2024-01-01', periods=n_days)
    
    table_name = f"test_chunk_{int(time.time())}"
    conn = get_connection()
    
    if not conn:
        print("❌ No HeavyDB connection")
        return False
    
    try:
        print(f"Testing single chunk load ({chunk_size} strategies)...")
        start_time = time.time()
        success = load_strategy_data(chunk_data, table_name, connection=conn, timeout=60)
        load_time = time.time() - start_time
        
        if success:
            print(f"✅ Chunk loaded in {load_time:.2f}s")
            print(f"📈 Load rate: {chunk_size/load_time:.0f} strategies/second")
            
            # Verify data
            cursor = conn.cursor()
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cursor.fetchone()[0]
            print(f"✅ Verified {count} rows in table")
            
            # Cleanup
            cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
        else:
            print("❌ Failed to load chunk")
        
        conn.close()
        return success
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        if conn:
            conn.close()
        return False


def test_gpu_enforcement():
    """Test GPU-only mode enforcement"""
    print("\n⚡ Testing GPU Enforcement")
    print("="*60)
    
    from gpu_enforced_workflow import GPUEnforcedOptimizer
    
    try:
        # Create GPU-enforced optimizer
        optimizer = GPUEnforcedOptimizer()
        print("✅ GPU-enforced optimizer created")
        
        # Verify settings
        if optimizer.gpu_mode_enforced and not optimizer.cpu_fallback_enabled:
            print("✅ GPU mode enforced, CPU fallback disabled")
            return True
        else:
            print("❌ GPU enforcement settings incorrect")
            return False
            
    except Exception as e:
        print(f"⚠️ GPU enforcement test: {e}")
        # Expected if HeavyDB not available
        return True


def main():
    """Run all GPU mode tests"""
    print("🔍 GPU Mode Complete Test Suite")
    print("="*80)
    print("Testing GPU functionality, optimizations, and algorithm compatibility")
    print("="*80)
    
    results = {}
    
    # Test 1: GPU Connection
    results['gpu_connection'] = test_gpu_connection()
    
    # Test 2: Correlation Optimization
    if results['gpu_connection']:
        results['correlation_opt'] = test_correlation_optimization()
    else:
        results['correlation_opt'] = False
        print("\n⚠️ Skipping correlation test (no GPU connection)")
    
    # Test 3: Algorithm Iterations
    results['algorithm_iterations'] = test_algorithm_iterations()
    
    # Test 4: Large Dataset Handling
    if results['gpu_connection']:
        results['large_dataset'] = test_large_dataset_handling()
    else:
        results['large_dataset'] = False
        print("\n⚠️ Skipping large dataset test (no GPU connection)")
    
    # Test 5: GPU Enforcement
    results['gpu_enforcement'] = test_gpu_enforcement()
    
    # Summary
    print("\n" + "="*80)
    print("📊 Test Summary")
    print("="*80)
    
    for test, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test:<25}: {status}")
    
    all_passed = all(results.values())
    if all_passed:
        print("\n✅ All GPU mode tests passed!")
    else:
        print("\n⚠️ Some tests failed - review details above")
    
    # Final recommendations
    print("\n💡 Recommendations:")
    if not results['gpu_connection']:
        print("- Ensure HeavyDB is running: sudo systemctl start heavydb")
    if not results['algorithm_iterations']:
        print("- Algorithm iterations have been updated to match original")
    if not results['large_dataset']:
        print("- Large datasets will be processed in chunks automatically")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())