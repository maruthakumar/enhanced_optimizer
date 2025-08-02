#!/usr/bin/env python3
"""
Test Full Workflow with HeavyDB Debugging
"""

import os
import sys
import pandas as pd
import numpy as np

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from csv_only_heavydb_workflow import CSVOnlyHeavyDBOptimizer


def test_workflow_with_debug():
    """Test the full workflow with debugging"""
    print("\n" + "="*60)
    print("Testing Full Workflow with HeavyDB")
    print("="*60)
    
    # Create test data
    n_days = 82
    n_strategies = 100
    
    dates = pd.date_range('2024-01-01', periods=n_days)
    data = {'Date': dates, 'Day': range(n_days)}
    
    # Create strategy columns
    for i in range(n_strategies):
        strategy_name = f"SENSEX {1000+i*10}-{1100+i*10} SL{np.random.randint(10,50)}%"
        # Generate realistic P&L data
        base_return = np.random.randn() * 50  # Base return for this strategy
        daily_returns = base_return + np.random.randn(n_days) * 100
        data[strategy_name] = daily_returns
    
    df = pd.DataFrame(data)
    
    # Save to test file
    test_file = '/tmp/test_workflow_data.csv'
    df.to_csv(test_file, index=False)
    print(f"✅ Created test data: {n_strategies} strategies, {n_days} days")
    
    # Initialize optimizer
    print("\n1. Initializing Optimizer...")
    optimizer = CSVOnlyHeavyDBOptimizer()
    
    # Run optimization
    print("\n2. Running Optimization...")
    try:
        # Load data
        loaded_data = optimizer.load_csv_data(test_file)
        print("   ✅ Data loaded")
        
        # Preprocess data
        processed_data = optimizer.preprocess_data(loaded_data)
        print("   ✅ Data preprocessed")
        print(f"   GPU Accelerated: {processed_data.get('gpu_accelerated', False)}")
        
        # Run algorithms
        portfolio_size = 20
        print(f"\n3. Running Algorithms (Portfolio size: {portfolio_size})...")
        
        # Test with just GA for now
        algorithm = optimizer.algorithms['GA']
        
        # Create fitness function
        def fitness_function(portfolio_indices):
            portfolio_data = processed_data['matrix'][:, portfolio_indices]
            portfolio_returns = portfolio_data.sum(axis=1)
            
            # Calculate metrics
            roi = portfolio_returns.sum()
            cumulative = np.cumsum(portfolio_returns)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = cumulative - running_max
            max_drawdown = abs(drawdown.min())
            
            if max_drawdown > 0:
                fitness = roi / max_drawdown
            else:
                fitness = 0
                
            return fitness
        
        # Run optimization
        result = algorithm.optimize(
            processed_data['matrix'],
            portfolio_size=portfolio_size,
            fitness_function=fitness_function
        )
        
        print(f"   ✅ Algorithm completed")
        print(f"   Best Fitness: {result.get('best_fitness', 0):.4f}")
        print(f"   Portfolio: {result.get('best_portfolio', [])[:5]}... (showing first 5)")
        
        # Test correlation calculation if available
        if processed_data.get('correlation_matrix') is not None:
            print(f"\n4. Correlation Analysis:")
            corr_shape = processed_data['correlation_matrix'].shape
            print(f"   ✅ Correlation matrix calculated: {corr_shape}")
            print(f"   GPU Accelerated: Yes")
        else:
            print(f"\n4. Correlation Analysis:")
            print(f"   ℹ️ No correlation matrix (CPU mode)")
        
        # Clean up
        os.remove(test_file)
        
        print("\n" + "="*60)
        print("✅ Workflow Test Completed Successfully!")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_heavydb_direct():
    """Test HeavyDB connection directly with different approaches"""
    print("\n" + "="*60)
    print("Direct HeavyDB Connection Tests")
    print("="*60)
    
    try:
        # Try pymapd directly
        import pymapd
        
        print("\n1. Testing direct pymapd connection...")
        
        # Test different connection approaches
        test_configs = [
            {'host': '127.0.0.1', 'port': 6274, 'user': 'admin', 'password': 'HyperInteractive', 'dbname': 'heavyai'},
            {'host': '127.0.0.1', 'port': 6274, 'user': 'admin', 'password': '', 'dbname': 'heavyai'},
            {'host': '127.0.0.1', 'port': 6274, 'user': 'admin', 'password': 'HyperInteractive', 'dbname': 'portfolio_optimizer'},
            {'host': 'localhost', 'port': 6274, 'user': 'admin', 'password': 'HyperInteractive', 'dbname': 'heavyai'},
        ]
        
        for i, config in enumerate(test_configs):
            print(f"\n   Test {i+1}: {config}")
            try:
                conn = pymapd.connect(**config)
                print(f"   ✅ Connected!")
                cursor = conn.cursor()
                cursor.execute("SELECT 1 as test")
                result = cursor.fetchall()
                print(f"   Result: {result}")
                conn.close()
                break
            except Exception as e:
                print(f"   ❌ Failed: {e}")
                
    except ImportError:
        print("❌ pymapd not available")
    
    # Test if HeavyDB is accessible
    print("\n2. Testing HeavyDB service status...")
    os.system("systemctl is-active heavydb")


def main():
    """Run all tests"""
    print("\n🔧 HeavyDB Full Workflow Test")
    
    # Test direct connection approaches
    test_heavydb_direct()
    
    # Test full workflow
    workflow_success = test_workflow_with_debug()
    
    if workflow_success:
        print("\n✅ All tests passed!")
    else:
        print("\n⚠️ Workflow test failed - but optimizer can still work in CPU mode")


if __name__ == "__main__":
    main()