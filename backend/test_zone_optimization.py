#!/usr/bin/env python3
"""
Test script for Zone-based Optimization with HeavyDB Integration
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def create_sample_zone_data():
    """Create sample data with Zone column for testing"""
    
    # Generate dates
    dates = pd.date_range('2024-01-01', periods=20, freq='D')
    
    # Define zones
    zones = ['Zone 1', 'Zone 2', 'Zone 3', 'Zone 4']
    
    # Create data rows
    data_rows = []
    
    for date in dates:
        for zone in zones:
            row = {
                'Date': date.strftime('%Y-%m-%d'),
                'Zone': zone,
                'Day': date.strftime('%A')
            }
            
            # Add strategy columns with zone-specific patterns
            for i in range(50):  # 50 strategies
                strategy_name = f'Strategy_{i:03d}'
                
                # Zone-specific returns
                if zone == 'Zone 1':
                    # Morning volatility
                    base_return = np.random.normal(0.001, 0.02)
                elif zone == 'Zone 2':
                    # Mid-day stability
                    base_return = np.random.normal(0.0005, 0.01)
                elif zone == 'Zone 3':
                    # Afternoon trends
                    base_return = np.random.normal(0.0008, 0.015)
                else:  # Zone 4
                    # Closing volatility
                    base_return = np.random.normal(0.0002, 0.025)
                
                # Add strategy-specific bias
                strategy_bias = (i - 25) * 0.00001
                row[strategy_name] = base_return + strategy_bias
            
            data_rows.append(row)
    
    df = pd.DataFrame(data_rows)
    
    # Save to CSV
    output_path = '/mnt/optimizer_share/input/test_zone_data.csv'
    df.to_csv(output_path, index=False)
    
    print(f"✅ Created test zone data: {output_path}")
    print(f"   Shape: {df.shape}")
    print(f"   Zones: {df['Zone'].unique()}")
    print(f"   Date range: {df['Date'].min()} to {df['Date'].max()}")
    
    return output_path

def test_zone_optimizer():
    """Test the IntradayZoneOptimizer"""
    
    print("\n=== Testing IntradayZoneOptimizer ===")
    
    from zone_optimizer_heavydb import IntradayZoneOptimizer
    
    # Create test data
    test_file = create_sample_zone_data()
    
    # Initialize optimizer
    optimizer = IntradayZoneOptimizer()
    
    try:
        # Process the file
        result = optimizer.process_consolidated_file(
            test_file,
            portfolio_size=15,
            algorithm="genetic"
        )
        
        print("\n✅ Zone optimization successful!")
        print(f"   Fitness: {result.fitness_score:.6f}")
        print(f"   Portfolio size: {len(result.portfolio_indices)}")
        print(f"   Optimization time: {result.optimization_time:.2f}s")
        
        print("\n📊 Zone Performance:")
        for zone, perf in result.zone_performance.items():
            print(f"   {zone}: {perf:.6f}")
        
        print(f"\n🏆 Weighted Performance: {result.weighted_performance:.6f}")
        
        print("\n📈 Selected Strategies:")
        for i, strategy in enumerate(result.portfolio_columns[:5]):
            print(f"   {i+1}. {strategy}")
        if len(result.portfolio_columns) > 5:
            print(f"   ... and {len(result.portfolio_columns) - 5} more")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Zone optimization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_zone_detection_in_workflow():
    """Test zone detection in main workflow"""
    
    print("\n=== Testing Zone Detection in Workflow ===")
    
    from csv_only_heavydb_workflow import CSVOnlyHeavyDBOptimizer
    
    # Create optimizer
    optimizer = CSVOnlyHeavyDBOptimizer()
    
    # Create test data
    test_file = '/mnt/optimizer_share/input/test_zone_data.csv'
    
    if not os.path.exists(test_file):
        test_file = create_sample_zone_data()
    
    try:
        # Run optimization
        success = optimizer.run_optimization(test_file, portfolio_size=20)
        
        if success:
            print("\n✅ Workflow successfully detected and processed zone data!")
        else:
            print("\n❌ Workflow failed to process zone data")
        
        return success
        
    except Exception as e:
        print(f"\n❌ Workflow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all zone optimization tests"""
    
    print("=" * 80)
    print("ZONE-BASED OPTIMIZATION TEST SUITE")
    print("=" * 80)
    
    tests = []
    
    # Test 1: IntradayZoneOptimizer
    test1_result = test_zone_optimizer()
    tests.append(("IntradayZoneOptimizer", test1_result))
    
    # Test 2: Zone detection in workflow
    test2_result = test_zone_detection_in_workflow()
    tests.append(("Zone Detection in Workflow", test2_result))
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for _, result in tests if result)
    
    for test_name, result in tests:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("\n🎉 All zone optimization tests passed!")
        return 0
    else:
        print("\n⚠️ Some tests failed. Check the output above.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)