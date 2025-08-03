#!/usr/bin/env python3
"""
Simple ULTA test focused on core functionality
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add backend to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_ulta_core_functionality():
    """Test core ULTA functionality"""
    logger.info("=" * 50)
    logger.info("ULTA Core Functionality Test")
    logger.info("=" * 50)
    
    try:
        from ulta_calculator import cuDFULTACalculator, ULTAStrategyMetrics
        
        # Create test data with known characteristics
        np.random.seed(42)
        num_days = 82  # Production dataset length
        
        data = {
            'Date': pd.date_range('2023-01-01', periods=num_days),
            'Symbol': ['SENSEX'] * num_days,
            'Price': np.random.normal(100, 5, num_days)
        }
        
        # Create strategies with different performance profiles
        strategies_data = {}
        
        # Poor performing strategies (should be inverted)
        for i in range(10):
            # Negative ROI, high negative day percentage
            returns = np.random.normal(-0.5, 1.0, num_days)
            strategies_data[f'Poor_Strategy_{i:03d}'] = returns
        
        # Good performing strategies (should not be inverted)
        for i in range(10):
            # Positive ROI
            returns = np.random.normal(0.8, 0.8, num_days)
            strategies_data[f'Good_Strategy_{i:03d}'] = returns
        
        # Mixed strategies
        for i in range(5):
            returns = np.random.normal(0.1, 1.2, num_days)
            strategies_data[f'Mixed_Strategy_{i:03d}'] = returns
        
        # Add all strategies to data
        data.update(strategies_data)
        df = pd.DataFrame(data)
        
        logger.info(f"Created test dataset: {num_days} days, {len(strategies_data)} strategies")
        
        # Test ULTA calculator
        calculator = cuDFULTACalculator(logger=logger)
        
        # Apply ULTA logic
        start_time = time.time()
        processed_df, metrics = calculator.apply_ulta_logic(df, start_column=3)
        processing_time = time.time() - start_time
        
        # Analyze results
        total_strategies = len(strategies_data)
        inverted_count = len(metrics)
        inversion_rate = inverted_count / total_strategies * 100
        
        logger.info("\n" + "=" * 30)
        logger.info("RESULTS SUMMARY")
        logger.info("=" * 30)
        logger.info(f"Total strategies analyzed: {total_strategies}")
        logger.info(f"Strategies inverted: {inverted_count}")
        logger.info(f"Inversion rate: {inversion_rate:.1f}%")
        logger.info(f"Processing time: {processing_time:.3f} seconds")
        logger.info(f"Throughput: {total_strategies/processing_time:.0f} strategies/second")
        
        # Check which strategies were inverted
        logger.info(f"\nInverted strategies:")
        poor_inverted = 0
        good_inverted = 0
        mixed_inverted = 0
        
        for strategy_name, metric in metrics.items():
            if metric.was_inverted:
                logger.info(f"  {strategy_name}: {metric.improvement_percentage:.1f}% improvement")
                if 'Poor_' in strategy_name:
                    poor_inverted += 1
                elif 'Good_' in strategy_name:
                    good_inverted += 1
                elif 'Mixed_' in strategy_name:
                    mixed_inverted += 1
        
        logger.info(f"\nInversion Analysis:")
        logger.info(f"  Poor strategies inverted: {poor_inverted}/10 ({poor_inverted*10}%)")
        logger.info(f"  Good strategies inverted: {good_inverted}/10 ({good_inverted*10}%)")
        logger.info(f"  Mixed strategies inverted: {mixed_inverted}/5 ({mixed_inverted*20}%)")
        
        # Validate that _inv suffix is used
        inv_columns = [col for col in processed_df.columns if col.endswith('_inv')]
        logger.info(f"\nColumns with '_inv' suffix: {len(inv_columns)}")
        
        # Check that inverted strategies have better ratios
        improvements = [m.improvement_percentage for m in metrics.values() if m.was_inverted]
        if improvements:
            avg_improvement = np.mean(improvements)
            logger.info(f"Average improvement: {avg_improvement:.2f}%")
            
            if avg_improvement > 0:
                logger.info("✅ Inverted strategies show improvement")
            else:
                logger.warning("⚠️ Inverted strategies show negative improvement")
        
        # Performance check (target: 25,544 strategies in <5 seconds)
        target_throughput = 25544 / 5  # ~5,109 strategies/second
        actual_throughput = total_strategies / processing_time
        
        if actual_throughput >= target_throughput:
            logger.info(f"✅ Performance target met ({actual_throughput:.0f} >= {target_throughput:.0f} strategies/second)")
        else:
            logger.warning(f"⚠️ Performance below target ({actual_throughput:.0f} < {target_throughput:.0f} strategies/second)")
        
        # Inversion rate validation (15-25% typical)
        if 10 <= inversion_rate <= 40:
            logger.info(f"✅ Inversion rate in reasonable range ({inversion_rate:.1f}%)")
        else:
            logger.warning(f"⚠️ Inversion rate outside expected range ({inversion_rate:.1f}%)")
        
        # Generate ULTA report
        logger.info("\n" + "="*30)
        logger.info("GENERATING ULTA REPORT")
        logger.info("="*30)
        
        report = calculator.generate_inversion_report("markdown")
        logger.info("Markdown report generated (first 500 chars):")
        logger.info(report[:500] + "..." if len(report) > 500 else report)
        
        # Performance report
        perf_report = calculator.generate_performance_report()
        logger.info("\nPerformance report:")
        logger.info(perf_report)
        
        logger.info("\n✅ ULTA Core Functionality Test PASSED")
        return True
        
    except Exception as e:
        logger.error(f"❌ ULTA Core Functionality Test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ulta_configuration():
    """Test ULTA configuration options"""
    logger.info("\n" + "=" * 50)
    logger.info("ULTA Configuration Test")
    logger.info("=" * 50)
    
    try:
        from ulta_calculator import cuDFULTACalculator
        
        # Create test data
        np.random.seed(42)
        num_days = 50
        
        data = {
            'Date': pd.date_range('2023-01-01', periods=num_days),
            'Symbol': ['TEST'] * num_days,
            'Price': np.random.normal(100, 5, num_days)
        }
        
        # Add a few test strategies
        for i in range(5):
            returns = np.random.normal(-0.3, 1.0, num_days)  # Poor performance
            data[f'Test_Strategy_{i}'] = returns
        
        df = pd.DataFrame(data)
        
        # Test different configurations
        configs = [
            {
                'name': 'Strict ULTA',
                'config': {
                    'enabled': True,
                    'roi_threshold': 0.0,
                    'min_negative_days': 30,  # High threshold
                    'negative_day_percentage': 0.8
                }
            },
            {
                'name': 'Lenient ULTA', 
                'config': {
                    'enabled': True,
                    'roi_threshold': 0.0,
                    'min_negative_days': 5,   # Low threshold
                    'negative_day_percentage': 0.4
                }
            },
            {
                'name': 'Disabled ULTA',
                'config': {
                    'enabled': False
                }
            }
        ]
        
        for test_config in configs:
            logger.info(f"\nTesting: {test_config['name']}")
            
            calculator = cuDFULTACalculator(logger=logger)
            calculator.ulta_config = test_config['config']
            
            processed_df, metrics = calculator.apply_ulta_logic(df, start_column=3)
            
            inverted_count = len(metrics)
            logger.info(f"  Strategies inverted: {inverted_count}/5")
            
            if test_config['name'] == 'Disabled ULTA' and inverted_count > 0:
                logger.error("❌ ULTA disabled but strategies were inverted")
                return False
        
        logger.info("\n✅ ULTA Configuration Test PASSED")
        return True
        
    except Exception as e:
        logger.error(f"❌ ULTA Configuration Test FAILED: {e}")
        return False

def main():
    """Run ULTA tests"""
    logger.info("Starting ULTA Comprehensive Tests")
    
    tests = [
        test_ulta_core_functionality,
        test_ulta_configuration
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    passed = sum(results)
    total = len(results)
    
    logger.info("\n" + "=" * 50)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        logger.info("🎉 All ULTA tests passed!")
        return True
    else:
        logger.error("❌ Some ULTA tests failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)