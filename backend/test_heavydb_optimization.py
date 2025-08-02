#!/usr/bin/env python3
"""
HeavyDB Optimization Test Script

Tests the optimized HeavyDB schema with production dataset to validate
performance targets specified in story_heavydb_optimization.md
"""

import os
import sys
import time
import logging
import argparse
from typing import Dict, Any

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dal.heavydb_dal import HeavyDBDAL
from lib.heavydb_connector.schema_optimizer import HeavyDBSchemaOptimizer, ProductionBenchmarkValidator


def setup_logging(log_level: str = 'INFO') -> logging.Logger:
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def test_production_dataset_optimization(input_file: str, table_name: str) -> Dict[str, Any]:
    """
    Test HeavyDB optimization with production dataset
    
    Args:
        input_file: Path to production CSV file
        table_name: Name for the test table
        
    Returns:
        Dict with test results and performance metrics
    """
    logger = logging.getLogger(__name__)
    logger.info("🚀 Starting HeavyDB Optimization Test")
    logger.info(f"   Input file: {input_file}")
    logger.info(f"   Target table: {table_name}")
    
    results = {
        'test_status': 'FAILED',
        'error': None,
        'performance_metrics': {},
        'optimization_features': [],
        'benchmark_results': {}
    }
    
    try:
        # Initialize HeavyDB DAL
        logger.info("📡 Initializing HeavyDB connection...")
        dal = HeavyDBDAL()
        
        if not dal.connect():
            logger.error("❌ Failed to connect to HeavyDB")
            results['error'] = 'Database connection failed'
            return results
        
        logger.info("✅ Connected to HeavyDB")
        logger.info(f"   GPU acceleration: {'ACTIVE' if dal.supports_gpu else 'INACTIVE'}")
        
        # Test 1: Load data with production optimization
        logger.info("📊 Test 1: Loading data with production optimization...")
        start_time = time.time()
        
        success = dal.load_csv_to_heavydb_optimized(
            filepath=input_file,
            table_name=table_name,
            use_production_optimization=True
        )
        
        load_time = time.time() - start_time
        
        if not success:
            logger.error("❌ Optimized data loading failed")
            results['error'] = 'Data loading failed'
            return results
        
        logger.info(f"✅ Data loaded in {load_time:.2f}s")
        results['performance_metrics']['load_time_seconds'] = load_time
        
        # Test 2: Validate table structure and optimization features
        logger.info("🔍 Test 2: Validating optimization features...")
        
        optimization_status = dal.get_optimization_status(table_name)
        if 'error' in optimization_status:
            logger.error(f"❌ Optimization validation failed: {optimization_status['error']}")
            results['error'] = optimization_status['error']
            return results
        
        results['optimization_features'] = optimization_status.get('optimization_features', [])
        results['performance_metrics']['row_count'] = optimization_status.get('row_count', 0)
        results['performance_metrics']['column_count'] = optimization_status.get('column_count', 0)
        
        logger.info("✅ Optimization features validated:")
        for feature in results['optimization_features']:
            logger.info(f"   • {feature}")
        
        # Test 3: Performance benchmarking
        logger.info("⚡ Test 3: Running performance benchmarks...")
        
        benchmark_validator = ProductionBenchmarkValidator(dal.schema_optimizer)
        benchmark_results = benchmark_validator.run_production_benchmark(
            table_name, dal.connection_manager
        )
        
        results['benchmark_results'] = benchmark_results
        results['performance_metrics'].update(benchmark_results)
        
        # Test 4: Validate against production targets
        logger.info("🎯 Test 4: Validating against production targets...")
        
        performance_targets = {
            'load_time_seconds': 30,
            'memory_usage_gb': 2,
            'correlation_query_seconds': 5,
            'optimization_seconds': 300
        }
        
        meets_targets, issues = dal.schema_optimizer.validate_optimization_performance(
            None, results['performance_metrics']
        )
        
        if meets_targets:
            logger.info("✅ All production performance targets met!")
            results['test_status'] = 'PASSED'
        else:
            logger.warning("⚠️  Some performance targets not met:")
            for issue in issues:
                logger.warning(f"   • {issue}")
            results['test_status'] = 'PARTIAL'
            results['performance_issues'] = issues
        
        # Test 5: Specific optimization validations
        logger.info("🔬 Test 5: Specific optimization validations...")
        
        # Check columnar storage optimization
        if results['performance_metrics']['column_count'] > 25000:
            logger.info("✅ Wide table optimization applied (25,000+ columns)")
        
        # Check date partitioning
        if 'date_partitioning' in results['optimization_features']:
            logger.info("✅ Date-based partitioning implemented")
        
        # Check GPU acceleration
        if dal.supports_gpu:
            logger.info("✅ GPU acceleration available and active")
        else:
            logger.warning("⚠️  GPU acceleration not available, using CPU fallback")
        
        # Clean up test table
        logger.info("🧹 Cleaning up test table...")
        dal.drop_table(table_name)
        
        # Disconnect
        dal.disconnect()
        logger.info("📡 Disconnected from HeavyDB")
        
        logger.info("🎉 HeavyDB Optimization Test Completed!")
        
    except Exception as e:
        logger.error(f"❌ Test failed with exception: {str(e)}")
        results['error'] = str(e)
        results['test_status'] = 'FAILED'
    
    return results


def print_test_summary(results: Dict[str, Any]) -> None:
    """Print comprehensive test summary"""
    logger = logging.getLogger(__name__)
    
    print("\n" + "="*60)
    print("HeavyDB OPTIMIZATION TEST SUMMARY")
    print("="*60)
    
    print(f"Test Status: {results['test_status']}")
    
    if results.get('error'):
        print(f"Error: {results['error']}")
        return
    
    print("\n📊 PERFORMANCE METRICS:")
    for metric, value in results.get('performance_metrics', {}).items():
        if isinstance(value, float):
            print(f"   {metric}: {value:.3f}")
        else:
            print(f"   {metric}: {value}")
    
    print("\n🎯 OPTIMIZATION FEATURES:")
    for feature in results.get('optimization_features', []):
        print(f"   ✅ {feature}")
    
    if 'performance_issues' in results:
        print("\n⚠️  PERFORMANCE ISSUES:")
        for issue in results['performance_issues']:
            print(f"   • {issue}")
    
    print("\n🔥 BENCHMARK RESULTS:")
    for benchmark, value in results.get('benchmark_results', {}).items():
        if isinstance(value, float):
            print(f"   {benchmark}: {value:.3f}")
        else:
            print(f"   {benchmark}: {value}")
    
    print("\n" + "="*60)


def main():
    """Main test execution"""
    parser = argparse.ArgumentParser(description='Test HeavyDB optimization with production data')
    parser.add_argument('--input', '-i', 
                       default='/mnt/optimizer_share/input/Python_Multi_Consolidated_20250726_161921.csv',
                       help='Path to input CSV file')
    parser.add_argument('--table', '-t', 
                       default='test_optimization_strategies',
                       help='Name for test table')
    parser.add_argument('--log-level', '-l', 
                       default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level)
    
    # Validate input file
    if not os.path.exists(args.input):
        logger.error(f"❌ Input file not found: {args.input}")
        sys.exit(1)
    
    # Get file info
    file_size_mb = os.path.getsize(args.input) / (1024 * 1024)
    logger.info(f"📁 Input file size: {file_size_mb:.1f} MB")
    
    # Run optimization test
    results = test_production_dataset_optimization(args.input, args.table)
    
    # Print summary
    print_test_summary(results)
    
    # Exit with appropriate code
    if results['test_status'] == 'PASSED':
        logger.info("🎉 All tests passed!")
        sys.exit(0)
    elif results['test_status'] == 'PARTIAL':
        logger.warning("⚠️  Some tests failed, but optimization is functional")
        sys.exit(1)
    else:
        logger.error("❌ Tests failed!")
        sys.exit(2)


if __name__ == '__main__':
    main()