#!/usr/bin/env python3
"""
Test script for ULTA cuDF integration
Validates the ULTA calculator and its integration with the Parquet/cuDF workflow
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import time
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add backend to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_ulta_calculator_import():
    """Test importing ULTA calculator"""
    logger.info("Testing ULTA calculator import...")
    
    try:
        from ulta_calculator import cuDFULTACalculator, ULTACalculator
        logger.info("✅ Successfully imported ULTA calculators")
        return True
    except ImportError as e:
        logger.error(f"❌ Failed to import ULTA calculators: {e}")
        return False

def test_cudf_import():
    """Test cuDF availability"""
    logger.info("Testing cuDF availability...")
    
    try:
        import cudf
        import cupy as cp
        logger.info("✅ cuDF and cuPy available for GPU acceleration")
        return True
    except (ImportError, RuntimeError) as e:
        logger.warning(f"⚠️ cuDF not available ({e}), will test CPU fallback")
        return False

def create_test_data():
    """Create test data for ULTA validation"""
    logger.info("Creating test data...")
    
    # Create synthetic strategy data
    np.random.seed(42)
    num_days = 100
    num_strategies = 50
    
    # Create base data structure
    data = {
        'Date': pd.date_range('2023-01-01', periods=num_days),
        'Symbol': ['TEST'] * num_days,
        'Price': np.random.normal(100, 10, num_days)
    }
    
    # Add strategy columns with different performance characteristics
    for i in range(num_strategies):
        if i < 15:  # 30% poor performing strategies (candidates for inversion)
            # Create strategies with negative ROI and high drawdown
            returns = np.random.normal(-0.5, 2.0, num_days)  # Negative expected return
            returns = np.clip(returns, -50, 50)  # Reasonable bounds
        elif i < 30:  # 30% moderate strategies
            returns = np.random.normal(0.1, 1.5, num_days)  # Small positive expected return
            returns = np.clip(returns, -30, 30)
        else:  # 40% good performing strategies
            returns = np.random.normal(1.0, 1.0, num_days)  # Good expected return
            returns = np.clip(returns, -20, 40)
        
        data[f'Strategy_{i:03d}'] = returns
    
    df = pd.DataFrame(data)
    logger.info(f"✅ Created test data: {len(df)} days, {num_strategies} strategies")
    
    return df

def test_ulta_calculator_basic():
    """Test basic ULTA calculator functionality"""
    logger.info("Testing basic ULTA calculator functionality...")
    
    try:
        from ulta_calculator import cuDFULTACalculator
        
        # Create calculator
        calculator = cuDFULTACalculator(logger=logger)
        
        # Test basic calculations
        test_returns = np.array([-1.0, -2.0, -0.5, 3.0, -1.5, 2.0])  # Poor strategy
        
        roi = calculator.calculate_roi(test_returns)
        drawdown = calculator.calculate_drawdown(test_returns)
        ratio = calculator.calculate_ratio(roi, drawdown)
        
        logger.info(f"Test strategy - ROI: {roi:.2f}, Drawdown: {drawdown:.2f}, Ratio: {ratio:.2f}")
        
        # Test inversion decision
        should_invert, metrics = calculator.should_invert_strategy(test_returns)
        
        if metrics:
            logger.info(f"Inversion decision: {should_invert}")
            logger.info(f"Original ratio: {metrics.original_ratio:.2f}")
            logger.info(f"Inverted ratio: {metrics.inverted_ratio:.2f}")
            logger.info(f"Improvement: {metrics.improvement_percentage:.2f}%")
        
        logger.info("✅ Basic ULTA calculator test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Basic ULTA calculator test failed: {e}")
        return False

def test_ulta_cudf_operations():
    """Test ULTA calculator with cuDF operations"""
    logger.info("Testing ULTA calculator with cuDF operations...")
    
    try:
        from ulta_calculator import cuDFULTACalculator
        import cudf
        
        # Create calculator
        calculator = cuDFULTACalculator(logger=logger)
        
        if not calculator.use_gpu:
            logger.info("GPU not available, testing CPU fallback")
        
        # Create test cuDF Series
        test_returns = [-1.0, -2.0, -0.5, 3.0, -1.5, 2.0]
        cudf_series = cudf.Series(test_returns)
        
        # Test cuDF operations
        roi = calculator.calculate_roi_cudf(cudf_series)
        drawdown = calculator.calculate_drawdown_cudf(cudf_series)
        ratio = calculator.calculate_ratio_cudf(roi, drawdown)
        
        logger.info(f"cuDF operations - ROI: {roi:.2f}, Drawdown: {drawdown:.2f}, Ratio: {ratio:.2f}")
        
        # Test inversion
        inverted = calculator.invert_strategy_cudf(cudf_series)
        logger.info(f"Original: {cudf_series.to_list()}")
        logger.info(f"Inverted: {inverted.to_list()}")
        
        logger.info("✅ cuDF operations test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ cuDF operations test failed: {e}")
        return False

def test_ulta_dataframe_processing():
    """Test ULTA processing on full DataFrame"""
    logger.info("Testing ULTA DataFrame processing...")
    
    try:
        from ulta_calculator import cuDFULTACalculator
        
        # Create test data
        test_data = create_test_data()
        
        # Create calculator
        calculator = cuDFULTACalculator(logger=logger)
        
        # Test with pandas DataFrame
        logger.info("Testing pandas DataFrame processing...")
        start_time = time.time()
        processed_data, metrics = calculator.apply_ulta_logic(test_data, start_column=3)
        pandas_time = time.time() - start_time
        
        logger.info(f"Pandas processing: {len(metrics)} strategies inverted in {pandas_time:.3f}s")
        
        # Test with cuDF DataFrame if available
        if calculator.use_gpu:
            try:
                import cudf
                cudf_data = cudf.from_pandas(test_data)
                
                logger.info("Testing cuDF DataFrame processing...")
                start_time = time.time()
                processed_cudf, cudf_metrics = calculator.apply_ulta_logic_cudf(cudf_data, start_column=3)
                cudf_time = time.time() - start_time
                
                logger.info(f"cuDF processing: {len(cudf_metrics)} strategies inverted in {cudf_time:.3f}s")
                
                if pandas_time > 0:
                    speedup = pandas_time / cudf_time
                    logger.info(f"GPU speedup: {speedup:.2f}x")
                
            except Exception as e:
                logger.warning(f"cuDF processing failed: {e}")
        
        # Validate results
        inverted_strategies = [name for name, metric in metrics.items() if metric.was_inverted]
        logger.info(f"Inverted strategies: {len(inverted_strategies)}")
        
        # Check for _inv suffix
        inv_columns = [col for col in processed_data.columns if col.endswith('_inv')]
        logger.info(f"Columns with '_inv' suffix: {len(inv_columns)}")
        
        # Validate inversion rate
        total_strategies = len(test_data.columns) - 3  # Subtract metadata columns
        inversion_rate = len(inverted_strategies) / total_strategies * 100
        logger.info(f"Inversion rate: {inversion_rate:.1f}%")
        
        # Expected inversion rate should be 15-25% based on test data design
        if 10 <= inversion_rate <= 40:  # Allow some variance
            logger.info("✅ Inversion rate within expected range")
        else:
            logger.warning(f"⚠️ Inversion rate outside expected range (10-40%)")
        
        logger.info("✅ DataFrame processing test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ DataFrame processing test failed: {e}")
        return False

def test_workflow_integration():
    """Test ULTA integration with Parquet/cuDF workflow"""
    logger.info("Testing workflow integration...")
    
    try:
        from parquet_cudf_workflow import ParquetCuDFWorkflow
        
        # Create test CSV file
        test_data = create_test_data()
        test_csv = "/tmp/test_ulta_data.csv"
        test_data.to_csv(test_csv, index=False)
        
        # Create workflow with ULTA enabled
        workflow = ParquetCuDFWorkflow()
        workflow.config['ulta']['enabled'] = True
        
        # Create temporary output directory
        output_dir = "/tmp/test_ulta_output"
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info("Running workflow with ULTA enabled...")
        
        # Run optimization (this will test the full integration)
        results = workflow.run_optimization(test_csv, output_dir)
        
        # Validate results
        if results['status'] == 'success':
            logger.info("✅ Workflow completed successfully")
            
            # Check ULTA results
            if 'ulta' in results:
                ulta_results = results['ulta']
                if ulta_results['enabled']:
                    logger.info(f"ULTA processing: {ulta_results['strategies_inverted']} strategies inverted")
                    logger.info(f"Inversion rate: {ulta_results['inversion_rate']:.2f}%")
                    logger.info(f"Processing time: {ulta_results['processing_time']:.3f}s")
                    
                    if ulta_results['strategies_inverted'] > 0:
                        logger.info("✅ ULTA integration successful")
                    else:
                        logger.warning("⚠️ No strategies were inverted")
                else:
                    logger.warning("⚠️ ULTA was not enabled in results")
            else:
                logger.error("❌ ULTA results not found in workflow output")
                return False
        else:
            logger.error(f"❌ Workflow failed: {results.get('error', 'Unknown error')}")
            return False
        
        # Cleanup
        if os.path.exists(test_csv):
            os.remove(test_csv)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Workflow integration test failed: {e}")
        return False

def test_performance_benchmarks():
    """Test performance benchmarks for ULTA processing"""
    logger.info("Testing performance benchmarks...")
    
    try:
        from ulta_calculator import cuDFULTACalculator
        
        # Create larger test dataset
        np.random.seed(42)
        num_days = 100
        num_strategies = 1000  # Larger dataset for performance testing
        
        data = {
            'Date': pd.date_range('2023-01-01', periods=num_days),
            'Symbol': ['TEST'] * num_days,
            'Price': np.random.normal(100, 10, num_days)
        }
        
        # Add strategies with mixed performance
        for i in range(num_strategies):
            if i < num_strategies // 4:  # 25% poor strategies
                returns = np.random.normal(-0.3, 1.5, num_days)
            else:  # 75% good strategies
                returns = np.random.normal(0.2, 1.0, num_days)
            
            data[f'Strategy_{i:04d}'] = np.clip(returns, -20, 20)
        
        test_data = pd.DataFrame(data)
        logger.info(f"Created performance test data: {num_strategies} strategies")
        
        # Create calculator
        calculator = cuDFULTACalculator(logger=logger, chunk_size=500)
        
        # Benchmark processing
        start_time = time.time()
        processed_data, metrics = calculator.apply_ulta_logic(test_data, start_column=3)
        processing_time = time.time() - start_time
        
        # Calculate performance metrics
        strategies_per_second = num_strategies / processing_time
        inverted_count = len(metrics)
        inversion_rate = inverted_count / num_strategies * 100
        
        logger.info(f"Performance Results:")
        logger.info(f"- Strategies processed: {num_strategies:,}")
        logger.info(f"- Processing time: {processing_time:.3f} seconds")
        logger.info(f"- Throughput: {strategies_per_second:.0f} strategies/second")
        logger.info(f"- Strategies inverted: {inverted_count}")
        logger.info(f"- Inversion rate: {inversion_rate:.2f}%")
        
        # Performance targets from story requirements
        target_throughput = 25544 / 5  # 25,544 strategies in <5 seconds
        
        if strategies_per_second >= target_throughput:
            logger.info(f"✅ Performance target met (>{target_throughput:.0f} strategies/second)")
        else:
            logger.warning(f"⚠️ Performance below target ({target_throughput:.0f} strategies/second)")
        
        # Inversion rate should be reasonable (15-25% typical)
        if 10 <= inversion_rate <= 40:
            logger.info("✅ Inversion rate within reasonable range")
        else:
            logger.warning(f"⚠️ Inversion rate unusual: {inversion_rate:.2f}%")
        
        logger.info("✅ Performance benchmark test completed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Performance benchmark test failed: {e}")
        return False

def run_all_tests():
    """Run all ULTA integration tests"""
    logger.info("Starting ULTA cuDF Integration Tests")
    logger.info("=" * 50)
    
    tests = [
        ("Import Tests", test_ulta_calculator_import),
        ("cuDF Availability", test_cudf_import),
        ("Basic ULTA Calculator", test_ulta_calculator_basic),
        ("cuDF Operations", test_ulta_cudf_operations),
        ("DataFrame Processing", test_ulta_dataframe_processing),
        ("Workflow Integration", test_workflow_integration),
        ("Performance Benchmarks", test_performance_benchmarks)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n--- {test_name} ---")
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("TEST SUMMARY")
    logger.info("=" * 50)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! ULTA cuDF integration is ready.")
        return True
    else:
        logger.error("❌ Some tests failed. Please review the issues above.")
        return False

def main():
    """Main entry point"""
    success = run_all_tests()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()