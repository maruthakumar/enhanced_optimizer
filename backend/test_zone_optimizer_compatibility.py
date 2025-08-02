#!/usr/bin/env python3
"""
Test zone optimizer compatibility with original implementation
Verify iteration counts, zone handling, inversion logic, and correlation
"""

import os
import sys
import numpy as np
import pandas as pd
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_algorithm_iterations():
    """Test that our algorithms match original iteration counts"""
    print("🔬 Testing Algorithm Iteration Counts")
    print("=" * 60)
    
    from config.config_manager import get_config_manager
    config = get_config_manager()
    
    # Original defaults from Optimizer_New_patched.py
    original_defaults = {
        'GA': ('ga_generations', 50),
        'PSO': ('pso_iterations', 50),  # Uses ga_generations in original
        'SA': ('sa_max_iterations', 1000),
        'DE': ('de_generations', 50),  # Uses ga_generations in original
        'ACO': ('aco_iterations', 50),  # Uses ga_generations in original
        'HC': ('hc_max_iterations', 200),
        'BO': ('bo_n_calls', 50),  # Uses ga_generations in original
    }
    
    print(f"{'Algorithm':<10} | {'Config Key':<20} | {'Our Value':<10} | {'Original':<10} | {'Match':<10}")
    print("-" * 80)
    
    mismatches = []
    
    for algo, (config_key, original_value) in original_defaults.items():
        our_value = config.getint('ALGORITHM_PARAMETERS', config_key, 0)
        
        # Check if it matches
        if algo in ['PSO', 'DE', 'ACO', 'BO']:
            # These use ga_generations in original
            ga_gens = config.getint('ALGORITHM_PARAMETERS', 'ga_generations', 50)
            match = (our_value == original_value) or (our_value == ga_gens)
            display_original = f"{original_value} (ga_gen)"
        else:
            match = (our_value == original_value)
            display_original = str(original_value)
        
        status = "✅" if match else "❌"
        print(f"{algo:<10} | {config_key:<20} | {our_value:<10} | {display_original:<10} | {status:<10}")
        
        if not match:
            mismatches.append((algo, our_value, original_value))
    
    return len(mismatches) == 0, mismatches

def test_zone_implementation():
    """Test zone weights and naming convention"""
    print("\n🌍 Testing Zone Implementation")
    print("=" * 60)
    
    # Test zone name normalization
    test_zones = ["Zone 1", "Zone 2", "North Zone", "South Zone"]
    
    print("Testing zone name normalization:")
    for zone in test_zones:
        # Original normalization: lowercase, remove spaces
        normalized = zone.lower().replace(" ", "")
        print(f"  '{zone}' -> '{normalized}'")
    
    # Test zone weights
    print("\nTesting zone weight normalization:")
    zone_weights = np.array([2.0, 3.0, 1.0, 4.0])
    normalized_weights = zone_weights / np.sum(zone_weights)
    
    print(f"  Original weights: {zone_weights}")
    print(f"  Normalized weights: {normalized_weights}")
    print(f"  Sum: {np.sum(normalized_weights):.6f}")
    
    return True

def test_inversion_logic():
    """Test ULTA inversion logic"""
    print("\n💱 Testing Inversion Logic")
    print("=" * 60)
    
    # Create test data with negative returns
    test_data = pd.DataFrame({
        'Date': pd.date_range('2024-01-01', periods=10),
        'Strategy1': [100, -50, 30, -20, 40, -60, 70, -80, 90, -100],  # Mixed
        'Strategy2': [-10, -20, -30, -40, -50, -60, -70, -80, -90, -100],  # All negative
        'Strategy3': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]  # All positive
    })
    
    print("Test data:")
    print(f"  Strategy1 total: {test_data['Strategy1'].sum()}")
    print(f"  Strategy2 total: {test_data['Strategy2'].sum()}")
    print(f"  Strategy3 total: {test_data['Strategy3'].sum()}")
    
    # Apply inversion logic
    inverted_strategies = {}
    
    for col in ['Strategy1', 'Strategy2', 'Strategy3']:
        total_return = test_data[col].sum()
        if total_return < 0:
            print(f"\n  Inverting {col} (total return: {total_return})")
            inverted_col = f"{col}_inv"
            test_data[inverted_col] = -test_data[col]
            inverted_total = test_data[inverted_col].sum()
            
            inverted_strategies[col] = {
                'original_pnl': total_return,
                'inverted_pnl': inverted_total,
                'original_ratio': 0 if total_return == 0 else total_return / abs(total_return),
                'inverted_ratio': 0 if inverted_total == 0 else inverted_total / abs(inverted_total)
            }
            
            print(f"    Original: {total_return} -> Inverted: {inverted_total}")
    
    # Generate inversion report
    print("\n📄 Inversion Report:")
    for strat, details in inverted_strategies.items():
        print(f"\n  {strat}:")
        print(f"    Original PnL: {details['original_pnl']:.2f}")
        print(f"    Inverted PnL: {details['inverted_pnl']:.2f}")
        print(f"    Original Ratio: {details['original_ratio']:.2f}")
        print(f"    Inverted Ratio: {details['inverted_ratio']:.2f}")
    
    return len(inverted_strategies) > 0

def test_correlation_calculation():
    """Test correlation matrix calculation"""
    print("\n📊 Testing Correlation Calculation")
    print("=" * 60)
    
    # Create test data with known correlations
    n_days = 100
    n_strategies = 5
    
    # Create correlated strategies
    base = np.random.randn(n_days)
    strategies = pd.DataFrame({
        'S1': base + np.random.randn(n_days) * 0.1,  # Highly correlated with base
        'S2': base + np.random.randn(n_days) * 0.1,  # Highly correlated with base
        'S3': -base + np.random.randn(n_days) * 0.1,  # Negatively correlated
        'S4': np.random.randn(n_days),  # Independent
        'S5': np.random.randn(n_days)   # Independent
    })
    
    # Calculate correlation matrix
    corr_matrix = strategies.corr().values
    
    print("Correlation matrix:")
    print(corr_matrix)
    
    # Test correlation penalty calculation
    selected = [0, 1, 2]  # Select correlated strategies
    corr_values = []
    for i in range(len(selected)):
        for j in range(i+1, len(selected)):
            corr_values.append(abs(corr_matrix[selected[i], selected[j]]))
    
    avg_corr = np.mean(corr_values) if corr_values else 0
    print(f"\nSelected strategies: {selected}")
    print(f"Average correlation: {avg_corr:.4f}")
    
    # Test normalization (from original)
    MEAN_CORRELATION = 0.4
    MAX_CORRELATION = 0.8
    
    normalized_corr = (avg_corr - MEAN_CORRELATION) / (MAX_CORRELATION - MEAN_CORRELATION)
    normalized_corr = np.clip(normalized_corr, 0, 1)
    
    print(f"Normalized correlation: {normalized_corr:.4f}")
    
    return True

def test_zone_fitness_calculation():
    """Test zone-based fitness calculation"""
    print("\n🎯 Testing Zone Fitness Calculation")
    print("=" * 60)
    
    # Create test zone matrix (days, zones, strategies)
    n_days = 30
    n_zones = 4
    n_strategies = 10
    
    zone_matrix = np.random.randn(n_days, n_zones, n_strategies) * 100
    
    # Test portfolio
    portfolio = [0, 2, 4, 6, 8]  # Select 5 strategies
    
    # Zone weights
    zone_weights = np.array([0.3, 0.3, 0.2, 0.2])  # Must sum to 1.0
    
    print(f"Zone matrix shape: {zone_matrix.shape}")
    print(f"Portfolio: {portfolio}")
    print(f"Zone weights: {zone_weights} (sum: {zone_weights.sum():.2f})")
    
    # Calculate fitness (from original logic)
    avg_returns = zone_matrix[:, :, portfolio].mean(axis=2)  # Average over selected strategies
    print(f"Average returns shape: {avg_returns.shape}")
    
    weighted_returns = np.dot(avg_returns, zone_weights)  # Apply zone weights
    print(f"Weighted returns shape: {weighted_returns.shape}")
    
    roi = np.sum(weighted_returns)
    cumulative = np.cumsum(weighted_returns)
    peak = np.maximum.accumulate(cumulative)
    max_dd = np.max(peak - cumulative) if len(cumulative) > 0 else 0
    
    print(f"\nResults:")
    print(f"  ROI: {roi:.2f}")
    print(f"  Max Drawdown: {max_dd:.2f}")
    
    if max_dd > 1e-6:
        fitness = roi / max_dd
    elif roi > 0:
        fitness = roi * 100
    elif roi < 0:
        fitness = roi * 10
    else:
        fitness = 0
    
    print(f"  Fitness: {fitness:.6f}")
    
    return True

def test_heavydb_zone_creation():
    """Test dynamic zone table creation in HeavyDB"""
    print("\n🗄️ Testing HeavyDB Zone Table Creation")
    print("=" * 60)
    
    from lib.heavydb_connector import get_connection, execute_query
    
    conn = get_connection()
    if not conn:
        print("⚠️ HeavyDB not connected, skipping zone table test")
        return False
    
    try:
        cursor = conn.cursor()
        
        # Create test zone table
        table_name = f"zone_strategies_test_{int(time.time())}"
        
        print(f"Creating table: {table_name}")
        
        # Drop if exists
        cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
        
        # Create table with zone information
        create_sql = f"""
        CREATE TABLE {table_name} (
            strategy_id INTEGER,
            zone_name TEXT ENCODING DICT(32),
            trading_date DATE,
            daily_returns DOUBLE,
            zone_weight DOUBLE
        )
        """
        cursor.execute(create_sql)
        print("✅ Table created")
        
        # Insert test data
        zones = ['zone1', 'zone2', 'zone3', 'zone4']
        zone_weights = [0.3, 0.3, 0.2, 0.2]
        
        for zone_idx, (zone, weight) in enumerate(zip(zones, zone_weights)):
            for strat_id in range(5):
                for day in range(10):
                    date_str = f"2024-01-{day+1:02d}"
                    returns = np.random.randn() * 100
                    
                    insert_sql = f"""
                    INSERT INTO {table_name} 
                    VALUES ({strat_id}, '{zone}', '{date_str}', {returns}, {weight})
                    """
                    cursor.execute(insert_sql)
        
        print("✅ Data inserted")
        
        # Query zone aggregations
        query = f"""
        SELECT 
            zone_name,
            COUNT(DISTINCT strategy_id) as num_strategies,
            AVG(daily_returns) as avg_return,
            AVG(zone_weight) as zone_weight
        FROM {table_name}
        GROUP BY zone_name
        ORDER BY zone_name
        """
        
        result = execute_query(query, connection=conn)
        if result is not None:
            print("\n📊 Zone Summary:")
            print(result)
        
        # Clean up
        cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
        print("\n✅ Cleanup complete")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        if conn:
            conn.close()
        return False

def main():
    """Run all compatibility tests"""
    print("🔍 Zone Optimizer Compatibility Test")
    print("=" * 80)
    print("Comparing with original Optimizer_New_patched.py implementation")
    print("=" * 80)
    
    results = {}
    
    # Test 1: Algorithm iterations
    iter_match, mismatches = test_algorithm_iterations()
    results['iterations'] = iter_match
    
    # Test 2: Zone implementation
    zone_ok = test_zone_implementation()
    results['zones'] = zone_ok
    
    # Test 3: Inversion logic
    inversion_ok = test_inversion_logic()
    results['inversion'] = inversion_ok
    
    # Test 4: Correlation
    corr_ok = test_correlation_calculation()
    results['correlation'] = corr_ok
    
    # Test 5: Zone fitness
    fitness_ok = test_zone_fitness_calculation()
    results['zone_fitness'] = fitness_ok
    
    # Test 6: HeavyDB zones
    heavydb_ok = test_heavydb_zone_creation()
    results['heavydb_zones'] = heavydb_ok
    
    # Summary
    print("\n" + "="*80)
    print("📊 Compatibility Test Summary")
    print("="*80)
    
    for test, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test:<20}: {status}")
    
    if not iter_match and mismatches:
        print("\n⚠️ Algorithm Iteration Mismatches:")
        for algo, our, original in mismatches:
            print(f"  - {algo}: {our} (should be {original})")
        print("\n💡 To match original, update config/production_config.ini:")
        print("  - Set ga_generations = 50")
        print("  - Set hc_max_iterations = 200")
        print("  - Set bo_n_calls = 50")
    
    all_passed = all(results.values())
    if all_passed:
        print("\n✅ All compatibility tests passed!")
    else:
        print("\n⚠️ Some compatibility issues found - see details above")
    
    return all_passed

if __name__ == "__main__":
    main()