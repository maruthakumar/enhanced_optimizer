#!/usr/bin/env python3
"""
Test script to compare legacy and new system results
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from csv_only_heavydb_workflow import CSVOnlyHeavyDBOptimizer
from legacy_system_integration import LegacyOutputParser, FitnessCalculationValidator
import json

def main():
    # Test data
    input_file = "../input/Python_Multi_Consolidated_20250726_161921.csv"
    portfolio_size = 37
    legacy_output_file = "../input/archive/pre_samba_migration/Zone Optimization new/Output/run_20250726_163251/best_portfolio_size37_20250726_163251.txt"
    
    print("=" * 80)
    print("LEGACY VS NEW SYSTEM COMPARISON TEST")
    print("=" * 80)
    
    # 1. Parse legacy output
    print("\n1. Parsing legacy system output...")
    parser = LegacyOutputParser()
    
    # Parse the legacy output file manually
    legacy_result = {
        'algorithm': 'SA',
        'fitness': 30.45764862187442,
        'portfolio_size': 37,
        'total_roi': 13653.365261621726,
        'max_drawdown': 443.7869616216202,
        'win_percentage': 0.6219512195121951,
        'profit_factor': 3.276700272201562
    }
    
    print(f"   Legacy Algorithm: {legacy_result['algorithm']}")
    print(f"   Legacy Fitness: {legacy_result['fitness']:.6f}")
    print(f"   Legacy Portfolio Size: {legacy_result['portfolio_size']}")
    
    # 2. Run new system
    print("\n2. Running new system optimization...")
    optimizer = CSVOnlyHeavyDBOptimizer()
    
    try:
        # Run the optimization
        result = optimizer.run_optimization(input_file, portfolio_size)
        
        if result and 'best_result' in result:
            new_fitness = result['best_result']['fitness']
            new_algorithm = result['best_result']['algorithm']
            
            print(f"   New Best Algorithm: {new_algorithm}")
            print(f"   New Best Fitness: {new_fitness:.6f}")
            
            # Find SA algorithm result specifically
            sa_result = None
            for algo_result in result.get('algorithm_results', []):
                if algo_result['algorithm'] == 'SA':
                    sa_result = algo_result
                    break
            
            if sa_result:
                print(f"\n   SA Algorithm Fitness: {sa_result['fitness']:.6f}")
                
                # 3. Validate fitness calculations
                print("\n3. Validating fitness calculation parity...")
                validator = FitnessCalculationValidator()
                
                is_valid, message, details = validator.validate_fitness_parity(
                    legacy_result['fitness'],
                    sa_result['fitness'],
                    tolerance=0.01  # 1% tolerance
                )
                
                print(f"   Validation Result: {'PASS' if is_valid else 'FAIL'}")
                print(f"   Message: {message}")
                if details:
                    print(f"   Details: {json.dumps(details, indent=2)}")
                
                # Summary
                print("\n" + "=" * 80)
                print("SUMMARY:")
                print(f"  Legacy SA Fitness: {legacy_result['fitness']:.6f}")
                print(f"  New SA Fitness: {sa_result['fitness']:.6f}")
                print(f"  Difference: {abs(legacy_result['fitness'] - sa_result['fitness']):.6f}")
                print(f"  Relative Difference: {details.get('relative_difference', 0)*100:.2f}%")
                print(f"  Status: {'✅ VALIDATED' if is_valid else '❌ VALIDATION FAILED'}")
                print("=" * 80)
                
            else:
                print("   ERROR: SA algorithm result not found in new system")
                
        else:
            print("   ERROR: Optimization failed or returned no results")
            
    except Exception as e:
        print(f"   ERROR: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()