#!/usr/bin/env python3
"""
Test Workflow Integration with Retrofitted Algorithms
"""

import numpy as np
import pandas as pd
import sys
import os
import json
import tempfile
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parquet_cudf_workflow import ParquetCuDFWorkflow

def create_test_csv():
    """Create a test CSV file"""
    np.random.seed(42)
    
    # Create test data
    dates = pd.date_range(start='2024-01-01', periods=82, freq='D')
    num_strategies = 50
    
    data = {'Date': dates}
    
    # Generate strategy returns
    for i in range(num_strategies):
        # Random daily returns between -5% and 5%
        returns = np.random.randn(len(dates)) * 0.02
        
        # Make some strategies better
        if i < 10:
            returns += 0.001  # Add bias to first 10 strategies
        
        data[f'Strategy_{i+1:03d}'] = returns
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to temporary CSV
    temp_csv = tempfile.mktemp(suffix='.csv')
    df.to_csv(temp_csv, index=False)
    
    return temp_csv

def test_workflow():
    """Test the integrated workflow"""
    print("="*60)
    print("Testing Workflow Integration with Retrofitted Algorithms")
    print("="*60)
    
    # Create test data
    print("\n1. Creating test CSV data...")
    csv_path = create_test_csv()
    print(f"   Created test CSV: {csv_path}")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"/tmp/test_workflow_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"   Output directory: {output_dir}")
    
    # Create workflow configuration
    config = {
        'portfolio_size': 10,
        'use_gpu': False,  # Force CPU mode for testing
        'chunk_size': 1000000,
        'compression': 'snappy',
        'correlation_threshold': 0.7,
        'fitness_weights': {
            'roi_dd_ratio_weight': 0.35,
            'total_roi_weight': 0.25,
            'max_drawdown_weight': 0.15,
            'win_rate_weight': 0.15,
            'profit_factor_weight': 0.10
        },
        'algorithm_timeouts': {
            'genetic_algorithm': 60,
            'particle_swarm': 60,
            'simulated_annealing': 60,
            'differential_evolution': 60,
            'ant_colony': 60,
            'hill_climbing': 30,
            'bayesian_optimization': 60,
            'random_search': 30
        }
    }
    
    # Save config
    config_path = os.path.join(output_dir, 'test_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print("\n2. Initializing workflow...")
    workflow = ParquetCuDFWorkflow(config_path)
    
    print("\n3. Running optimization workflow...")
    print("   This will test all 8 retrofitted algorithms...")
    
    try:
        results = workflow.run_optimization(csv_path, output_dir)
        
        if results['status'] == 'success':
            print(f"\n✅ Workflow completed successfully!")
            print(f"   Total time: {results['total_time']:.2f} seconds")
            print(f"   Strategies: {results['num_strategies']}")
            print(f"   Days: {results['num_days']}")
            
            print("\n4. Algorithm Results:")
            for algo_name, algo_result in results['algorithms'].items():
                if algo_result['status'] == 'success':
                    print(f"\n   {algo_name}:")
                    print(f"      Status: ✅ Success")
                    print(f"      Fitness: {algo_result.get('fitness_score', 'N/A')}")
                    print(f"      Time: {algo_result['execution_time']:.3f}s")
                    print(f"      Data type: {algo_result.get('data_type', 'N/A')}")
                    print(f"      GPU: {algo_result.get('gpu_accelerated', False)}")
                else:
                    print(f"\n   {algo_name}:")
                    print(f"      Status: ❌ Failed - {algo_result.get('error', 'Unknown error')}")
            
            # Find best algorithm
            successful = [(name, res) for name, res in results['algorithms'].items() 
                         if res['status'] == 'success']
            if successful:
                best = max(successful, key=lambda x: x[1].get('fitness_score', 0))
                print(f"\n5. Best Algorithm: {best[0]} (fitness: {best[1]['fitness_score']:.4f})")
            
            print(f"\n6. Output files generated in: {output_dir}")
            
        else:
            print(f"\n❌ Workflow failed: {results.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"\n❌ Test failed with exception: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        if os.path.exists(csv_path):
            os.remove(csv_path)
            print(f"\n7. Cleaned up test CSV file")
    
    print("\n" + "="*60)
    print("Workflow Integration Test Complete")
    print("="*60)

if __name__ == "__main__":
    test_workflow()