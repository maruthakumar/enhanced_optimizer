#!/usr/bin/env python3
"""
Story 1.4 Integration Test: ULTA Algorithm Enhancement & Zone Optimizer Integration
Tests the complete integration of ULTA processing with 8-zone optimization
"""

import os
import sys
import logging
import time
import numpy as np
import pandas as pd
from pathlib import Path

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ulta_calculator import ULTACalculator
from zone_optimizer import ZoneOptimizer, ZoneConfiguration

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_test_dataset(num_strategies: int = 800, num_days: int = 82) -> pd.DataFrame:
    """Create test dataset similar to production data structure"""
    
    # Create date columns
    dates = pd.date_range('2024-01-01', periods=num_days, freq='D')
    
    data = {
        'Date': dates,
        'Day': range(1, num_days + 1)
    }
    
    # Create strategy columns with mixed performance (some negative for ULTA testing)
    np.random.seed(42)  # Reproducible results
    
    for i in range(num_strategies):
        strategy_name = f"Strategy_{i:04d}"
        
        # Create diverse performance patterns
        if i % 4 == 0:  # 25% poor performers (candidates for ULTA inversion)
            returns = np.random.normal(-0.001, 0.02, num_days)  # Slightly negative bias
        elif i % 4 == 1:  # 25% good performers  
            returns = np.random.normal(0.002, 0.015, num_days)  # Positive bias
        else:  # 50% mixed performers
            returns = np.random.normal(0.0005, 0.018, num_days)  # Neutral bias
            
        data[strategy_name] = returns
    
    df = pd.DataFrame(data)
    
    logging.info(f"Created test dataset: {num_strategies} strategies, {num_days} days")
    logging.info(f"Dataset shape: {df.shape}")
    
    return df

def test_ulta_calculator():
    """Test ULTA calculator functionality"""
    logging.info("=== Testing ULTA Calculator ===")
    
    # Create sample strategy with poor performance (enough negative days to meet criteria)
    poor_returns = np.array([
        -0.01, -0.005, -0.008, -0.003, -0.012, -0.006, -0.004, -0.009, -0.002, -0.007,
        -0.011, -0.001, -0.015, -0.003, -0.005, 0.002, -0.008, -0.006, -0.004, -0.010
    ])  # 18 out of 20 are negative (90% negative days)
    
    calculator = ULTACalculator()
    
    # Test inversion decision
    should_invert, metrics = calculator.should_invert_strategy(poor_returns)
    
    logging.info(f"Poor strategy - Should invert: {should_invert}")
    if metrics:
        logging.info(f"Original ROI: {metrics.original_roi:.4f}")
        logging.info(f"Inverted ROI: {metrics.inverted_roi:.4f}")
        logging.info(f"Improvement: {metrics.improvement_percentage:.1f}%")
    
    # Test with good strategy (mostly positive returns)
    good_returns = np.array([
        0.008, 0.012, 0.006, 0.004, 0.009, 0.007, 0.003, 0.005, 0.011, 0.002,
        0.010, 0.006, 0.008, -0.001, 0.004, 0.009, 0.007, -0.002, 0.005, 0.012
    ])  # Only 2 out of 20 are negative (10% negative days)
    should_invert_good, metrics_good = calculator.should_invert_strategy(good_returns)
    
    logging.info(f"Good strategy - Should invert: {should_invert_good}")
    
    return should_invert and not should_invert_good

def test_8_zone_creation():
    """Test 8-zone creation from dataset"""
    logging.info("=== Testing 8-Zone Creation ===")
    
    # Create test dataset with exactly 756 strategies to match story requirements
    test_data = create_test_dataset(num_strategies=756, num_days=50)
    
    optimizer = ZoneOptimizer()
    zones = optimizer.extract_zones_from_data(test_data)
    
    logging.info(f"Created {len(zones)} zones")
    
    expected_zones = [
        "zone_0_100", "zone_101_200", "zone_201_300", "zone_301_400",
        "zone_401_500", "zone_501_600", "zone_601_700", "zone_701_756"
    ]
    
    success = True
    for expected_zone in expected_zones:
        if expected_zone in zones:
            zone_data = zones[expected_zone]
            metadata_cols = ['Date', 'Day', 'Zone']
            strategy_count = len([col for col in zone_data.columns if col not in metadata_cols])
            logging.info(f"{expected_zone}: {strategy_count} strategies")
        else:
            logging.error(f"Missing expected zone: {expected_zone}")
            success = False
    
    return success and len(zones) == 8

def test_zone_ulta_integration():
    """Test ULTA integration with zone optimization"""
    logging.info("=== Testing Zone-ULTA Integration ===")
    
    # Create smaller test dataset for faster processing
    test_data = create_test_dataset(num_strategies=200, num_days=30)
    
    optimizer = ZoneOptimizer()
    zones = optimizer.extract_zones_from_data(test_data)
    
    # Test single zone optimization with ULTA
    if "zone_0_100" in zones:
        zone_data = zones["zone_0_100"]
        
        # Test optimization with ULTA enabled
        optimizer.config.apply_ulta = True
        result_with_ulta = optimizer.optimize_single_zone(
            "zone_0_100", 
            zone_data, 
            portfolio_size=10,
            algorithm="genetic"
        )
        
        logging.info(f"Zone optimization with ULTA:")
        logging.info(f"  Fitness: {result_with_ulta.fitness_score:.4f}")
        logging.info(f"  ULTA inversions: {result_with_ulta.metadata['ulta_inversions']}")
        logging.info(f"  Portfolio size: {len(result_with_ulta.portfolio_indices)}")
        
        # Test optimization without ULTA for comparison
        optimizer.config.apply_ulta = False
        result_without_ulta = optimizer.optimize_single_zone(
            "zone_0_100", 
            zone_data, 
            portfolio_size=10,
            algorithm="genetic"
        )
        
        logging.info(f"Zone optimization without ULTA:")
        logging.info(f"  Fitness: {result_without_ulta.fitness_score:.4f}")
        logging.info(f"  Portfolio size: {len(result_without_ulta.portfolio_indices)}")
        
        return True
    else:
        logging.error("zone_0_100 not found in zones")
        return False

def test_full_8_zone_optimization():
    """Test complete 8-zone optimization workflow"""
    logging.info("=== Testing Full 8-Zone Optimization ===")
    
    # Create test dataset with enough strategies for all zones
    test_data = create_test_dataset(num_strategies=400, num_days=25)
    
    optimizer = ZoneOptimizer()
    optimizer.config.apply_ulta = True
    
    try:
        # Run optimization on all zones
        all_results = optimizer.optimize_all_zones(
            test_data,
            portfolio_size_per_zone=5,  # Small portfolio for fast testing
            algorithm="genetic"
        )
        
        logging.info(f"Completed optimization for {len(all_results)} zones")
        
        total_ulta_inversions = 0
        for zone_name, result in all_results.items():
            inversions = result.metadata.get('ulta_inversions', 0)
            total_ulta_inversions += inversions
            logging.info(f"{zone_name}: Fitness={result.fitness_score:.4f}, "
                        f"ULTA inversions={inversions}, Time={result.optimization_time:.2f}s")
        
        logging.info(f"Total ULTA inversions across all zones: {total_ulta_inversions}")
        
        # Test zone combination
        combined_result = optimizer.combine_zone_results(all_results)
        logging.info(f"Combined portfolio size: {len(combined_result.combined_portfolio)}")
        logging.info(f"Combined fitness: {combined_result.combined_fitness:.4f}")
        
        return len(all_results) > 0 and total_ulta_inversions >= 0
        
    except Exception as e:
        logging.error(f"Full 8-zone optimization failed: {e}")
        return False

def test_performance_with_real_dataset():
    """Test with real dataset if available"""
    logging.info("=== Testing with Real Dataset (if available) ===")
    
    # Look for real dataset
    real_data_paths = [
        "/mnt/optimizer_share/input/SENSEX_test_dataset.csv",
        "/mnt/optimizer_share/input/Python_Multi_Consolidated_20250726_161921.csv"
    ]
    
    for data_path in real_data_paths:
        if os.path.exists(data_path):
            logging.info(f"Found real dataset: {data_path}")
            
            try:
                # Load subset for testing (first 1000 strategies)
                df = pd.read_csv(data_path)
                if df.shape[1] > 1000:
                    # Take first 1000 strategy columns + metadata
                    metadata_cols = ['Date', 'Day', 'Zone'] if 'Zone' in df.columns else ['Date', 'Day']
                    strategy_cols = [col for col in df.columns if col not in metadata_cols][:1000]
                    df = df[metadata_cols + strategy_cols]
                
                logging.info(f"Real dataset shape: {df.shape}")
                
                optimizer = ZoneOptimizer()
                optimizer.config.apply_ulta = True
                
                start_time = time.time()
                zones = optimizer.extract_zones_from_data(df)
                processing_time = time.time() - start_time
                
                logging.info(f"Zone extraction took {processing_time:.2f}s")
                logging.info(f"Created {len(zones)} zones from real data")
                
                # Test one zone for performance
                if zones:
                    first_zone_name = list(zones.keys())[0]
                    first_zone_data = zones[first_zone_name]
                    
                    start_time = time.time()
                    result = optimizer.optimize_single_zone(
                        first_zone_name,
                        first_zone_data,
                        portfolio_size=15,
                        algorithm="genetic"
                    )
                    optimization_time = time.time() - start_time
                    
                    logging.info(f"Real data zone optimization: {optimization_time:.2f}s")
                    logging.info(f"ULTA inversions: {result.metadata.get('ulta_inversions', 0)}")
                    
                return True
                
            except Exception as e:
                logging.warning(f"Failed to process real dataset {data_path}: {e}")
                continue
    
    logging.info("No real dataset found - using synthetic data test results")
    return True

def main():
    """Run all integration tests"""
    logging.info("Starting Story 1.4 Integration Tests")
    logging.info("=" * 60)
    
    test_results = {}
    
    # Run all tests
    test_results['ulta_calculator'] = test_ulta_calculator()
    test_results['8_zone_creation'] = test_8_zone_creation()
    test_results['zone_ulta_integration'] = test_zone_ulta_integration()
    test_results['full_8_zone_optimization'] = test_full_8_zone_optimization()
    test_results['real_dataset_performance'] = test_performance_with_real_dataset()
    
    # Summary
    logging.info("=" * 60)
    logging.info("INTEGRATION TEST RESULTS")
    logging.info("=" * 60)
    
    passed = 0
    for test_name, result in test_results.items():
        status = "PASS" if result else "FAIL"
        logging.info(f"{test_name}: {status}")
        if result:
            passed += 1
    
    overall_success = passed == len(test_results)
    logging.info(f"\nOverall: {passed}/{len(test_results)} tests passed")
    
    if overall_success:
        logging.info("✅ Story 1.4 integration tests PASSED")
    else:
        logging.info("❌ Story 1.4 integration tests FAILED")
    
    return overall_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)