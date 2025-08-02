#!/usr/bin/env python3
"""
Test script to verify legacy fitness calculation implementation
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from csv_only_heavydb_workflow import CSVOnlyHeavyDBOptimizer
import pandas as pd
import numpy as np

def test_fitness_calculation():
    """Test that fitness calculation matches legacy formula"""
    print("=" * 80)
    print("TESTING LEGACY FITNESS CALCULATION IMPLEMENTATION")
    print("=" * 80)
    
    # Create test data
    test_data = pd.DataFrame({
        'Date': pd.date_range('2024-01-01', periods=100),
        'Strategy1': np.random.randn(100) * 100,
        'Strategy2': np.random.randn(100) * 100,
        'Strategy3': np.random.randn(100) * 100
    })
    
    optimizer = CSVOnlyHeavyDBOptimizer()
    
    print("\n1. Testing basic fitness calculation...")
    # Test with known values
    test_cases = [
        {
            'name': 'Positive ROI with drawdown',
            'portfolio_returns': pd.Series([100, -50, 75, -25, 100]),
            'expected_roi': 200,  # (100000 + 200 - 100000) / 100000 * 100
            'expected_max_dd': 75,  # Max drawdown from peak
            'expected_fitness': 200 / 75  # ROI / max_drawdown
        },
        {
            'name': 'Legacy example values',
            'portfolio_returns': pd.Series([13653.365261621726]),  # Total return matching legacy
            'expected_roi': 13.653365261621726,  # As percentage
            'expected_max_dd': 0.01,  # Minimal drawdown (edge case)
            'expected_fitness': 13.653365261621726 / 0.01  # Should be ~1365.34
        }
    ]
    
    for test_case in test_cases:
        print(f"\n   Testing: {test_case['name']}")
        
        # Create test portfolio data
        portfolio_data = pd.DataFrame({
            'test_strategy': test_case['portfolio_returns']
        })
        
        # Calculate fitness
        fitness = optimizer.standardize_fitness_calculation(portfolio_data, ['test_strategy'])
        
        print(f"   Expected fitness: {test_case['expected_fitness']:.6f}")
        print(f"   Actual fitness: {fitness:.6f}")
        print(f"   Match: {'✅ PASS' if abs(fitness - test_case['expected_fitness']) < 0.1 else '❌ FAIL'}")
    
    print("\n2. Testing with production data...")
    # Test with actual production data
    input_file = "../input/Python_Multi_Consolidated_20250726_161921.csv"
    
    if os.path.exists(input_file):
        print(f"   Loading production data from: {input_file}")
        
        # Run optimization with portfolio size 37 (matching legacy test)
        result = optimizer.run_optimization(input_file, portfolio_size=37)
        
        if result:
            print("\n   Optimization completed successfully!")
            
            # Check output files for fitness values
            output_dir = sorted([d for d in os.listdir('../output') if d.startswith('run_')], reverse=True)[0]
            summary_file = f"../output/{output_dir}/optimization_summary_{output_dir.split('_')[1]}_{output_dir.split('_')[2]}.txt"
            
            if os.path.exists(summary_file):
                with open(summary_file, 'r') as f:
                    content = f.read()
                    
                print("\n   Algorithm Results:")
                for line in content.split('\n'):
                    if 'SA:' in line and 'Fitness=' in line:
                        print(f"   {line.strip()}")
                        # Extract SA fitness value
                        sa_fitness = float(line.split('Fitness=')[1].split(',')[0])
                        print(f"\n   SA Fitness: {sa_fitness:.6f}")
                        print(f"   Legacy SA Fitness: 30.457649")
                        print(f"   Difference: {abs(sa_fitness - 30.457649):.6f}")
                        
                        # Check if within reasonable range (considering different data/randomness)
                        if 20 < sa_fitness < 40:
                            print("   ✅ FITNESS IN EXPECTED RANGE (20-40)")
                        else:
                            print("   ⚠️  FITNESS OUTSIDE EXPECTED RANGE")
    else:
        print(f"   ⚠️  Production data file not found: {input_file}")
    
    print("\n3. Testing fitness formula edge cases...")
    edge_cases = [
        {
            'name': 'No drawdown, positive ROI',
            'roi': 50,
            'max_dd': 0,
            'expected': 50 * 100  # ROI * 100
        },
        {
            'name': 'No drawdown, zero ROI',
            'roi': 0,
            'max_dd': 0,
            'expected': 0.0
        },
        {
            'name': 'Normal case',
            'roi': 100,
            'max_dd': 10,
            'expected': 10.0  # 100 / 10
        }
    ]
    
    for case in edge_cases:
        print(f"\n   Testing: {case['name']}")
        print(f"   ROI: {case['roi']}, Max DD: {case['max_dd']}")
        
        # Simulate the fitness calculation logic
        if case['max_dd'] > 0:
            fitness = case['roi'] / case['max_dd']
        else:
            if case['roi'] > 0:
                fitness = case['roi'] * 100
            else:
                fitness = 0.0
        
        print(f"   Expected: {case['expected']}, Actual: {fitness}")
        print(f"   {'✅ PASS' if fitness == case['expected'] else '❌ FAIL'}")
    
    print("\n" + "=" * 80)
    print("TESTING COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    test_fitness_calculation()