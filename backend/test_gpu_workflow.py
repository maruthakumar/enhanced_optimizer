#!/usr/bin/env python3
"""
Test Complete GPU Workflow with All 8 Algorithms
"""

import os
import sys
import time
import pandas as pd
import numpy as np

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from csv_only_heavydb_workflow import CSVOnlyHeavyDBOptimizer


def test_complete_workflow():
    """Test complete workflow with GPU acceleration"""
    print("\n🚀 Testing Complete GPU-Accelerated Workflow")
    print("="*60)
    
    # Create test data with real strategy names
    n_strategies = 100
    n_days = 82
    
    dates = pd.date_range('2024-01-01', periods=n_days)
    data = {'Date': dates, 'trading_day': range(n_days)}
    
    # Create strategies with SENSEX naming pattern
    for i in range(n_strategies):
        strategy_name = f"SENSEX {1000+i}-{1100+i} SL{i%50+10}%"
        # Generate realistic P&L data
        base_trend = np.linspace(-500, 500, n_days) * (1 if i % 3 == 0 else -1)
        volatility = np.random.randn(n_days) * (50 + i % 30)
        data[strategy_name] = base_trend + volatility
    
    df = pd.DataFrame(data)
    
    # Save test file
    test_file = '/tmp/gpu_workflow_test.csv'
    df.to_csv(test_file, index=False)
    
    print(f"Created test data: {n_strategies} strategies, {n_days} days")
    
    # Initialize optimizer
    optimizer = CSVOnlyHeavyDBOptimizer()
    
    # Test all 8 algorithms
    algorithms = ['GA', 'PSO', 'SA', 'DE', 'ACO', 'HC', 'BO', 'RS']
    portfolio_size = 10
    
    results = {}
    
    print(f"\nTesting all {len(algorithms)} algorithms with portfolio size {portfolio_size}:")
    print("-" * 60)
    
    for algo_name in algorithms:
        print(f"\n{algo_name} (Algorithm {algorithms.index(algo_name)+1}/{len(algorithms)}):")
        
        try:
            start_time = time.time()
            
            # Set the algorithm to use
            optimizer.selected_algorithms = [algo_name]
            
            # Run optimizer
            result = optimizer.run_optimization(
                csv_file_path=test_file,
                portfolio_size=portfolio_size
            )
            
            algo_time = time.time() - start_time
            
            if result and 'algorithm_results' in result and algo_name in result['algorithm_results']:
                algo_result = result['algorithm_results'][algo_name]
                
                print(f"  ✅ Success in {algo_time:.2f}s")
                print(f"     Best Fitness: {algo_result.get('best_fitness', 0):.4f}")
                print(f"     ROI: ${algo_result.get('total_roi', 0):,.2f}")
                print(f"     Max Drawdown: ${algo_result.get('max_drawdown', 0):,.2f}")
                print(f"     Win Rate: {algo_result.get('win_rate', 0):.1f}%")
                print(f"     GPU Accelerated: {result.get('preprocessing', {}).get('gpu_accelerated', False)}")
                
                results[algo_name] = {
                    'success': True,
                    'time': algo_time,
                    'fitness': algo_result.get('best_fitness', 0),
                    'roi': algo_result.get('total_roi', 0),
                    'drawdown': algo_result.get('max_drawdown', 0)
                }
            else:
                print(f"  ❌ Failed - no results returned")
                results[algo_name] = {'success': False, 'time': algo_time}
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
            results[algo_name] = {'success': False, 'error': str(e)}
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY:")
    print("="*60)
    
    successful = sum(1 for r in results.values() if r.get('success', False))
    print(f"✅ Successful algorithms: {successful}/{len(algorithms)}")
    
    if successful > 0:
        # Find best algorithm
        best_algo = max(
            (algo for algo, res in results.items() if res.get('success', False)),
            key=lambda a: results[a].get('fitness', 0)
        )
        
        print(f"\n🏆 Best Algorithm: {best_algo}")
        print(f"   Fitness: {results[best_algo]['fitness']:.4f}")
        print(f"   ROI: ${results[best_algo]['roi']:,.2f}")
        print(f"   Max Drawdown: ${results[best_algo]['drawdown']:,.2f}")
        
        # Average performance
        avg_time = sum(r['time'] for r in results.values() if 'time' in r) / len(results)
        print(f"\n⏱️  Average execution time: {avg_time:.2f}s per algorithm")
    
    # Clean up
    os.remove(test_file)
    
    return results


def test_ulta_logic():
    """Test ULTA strategy inversion logic"""
    print("\n\n🔄 Testing ULTA Strategy Inversion")
    print("="*60)
    
    # Create test data with poor performing strategies
    n_strategies = 50
    n_days = 82
    
    dates = pd.date_range('2024-01-01', periods=n_days)
    data = {'Date': dates, 'trading_day': range(n_days)}
    
    # Create mix of good and poor strategies
    for i in range(n_strategies):
        strategy_name = f"SENSEX {2000+i}-{2100+i} SL{i%30+15}%"
        
        if i < 25:  # First half - poor performers
            # Downward trend with high volatility
            trend = np.linspace(0, -1000, n_days)
            volatility = np.random.randn(n_days) * 100
            data[strategy_name] = trend + volatility
        else:  # Second half - good performers
            # Upward trend with low volatility
            trend = np.linspace(0, 500, n_days)
            volatility = np.random.randn(n_days) * 30
            data[strategy_name] = trend + volatility
    
    df = pd.DataFrame(data)
    
    # Save test file
    test_file = '/tmp/ulta_test.csv'
    df.to_csv(test_file, index=False)
    
    print(f"Created test data: {n_strategies} strategies (50% poor performers)")
    
    # Run optimizer
    optimizer = CSVOnlyHeavyDBOptimizer()
    
    try:
        optimizer.selected_algorithms = ['GA']  # Use GA for ULTA test
        result = optimizer.run_optimization(
            csv_file_path=test_file,
            portfolio_size=10
        )
        
        if result and 'preprocessing' in result:
            ulta_count = result['preprocessing'].get('ulta_inverted_count', 0)
            print(f"\n✅ ULTA inversions applied: {ulta_count} strategies")
            
            if ulta_count > 0:
                print("   ULTA successfully identified and inverted poor performers!")
            
            # Check if results improved
            if 'algorithm_results' in result and 'GA' in result['algorithm_results']:
                ga_result = result['algorithm_results']['GA']
                print(f"\nOptimized Portfolio Results:")
                print(f"   ROI: ${ga_result.get('total_roi', 0):,.2f}")
                print(f"   Max Drawdown: ${ga_result.get('max_drawdown', 0):,.2f}")
                print(f"   Win Rate: {ga_result.get('win_rate', 0):.1f}%")
                
    except Exception as e:
        print(f"❌ ULTA test failed: {e}")
    
    # Clean up
    os.remove(test_file)


if __name__ == "__main__":
    # Run tests
    results = test_complete_workflow()
    test_ulta_logic()
    
    print("\n✅ GPU Workflow Tests Complete!")