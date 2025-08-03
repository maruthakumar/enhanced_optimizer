#!/usr/bin/env python3
"""
Test script for Parquet/Arrow/cuDF workflow
Verifies functionality and performance
"""

import os
import sys
import json
import time
import logging
from datetime import datetime
import pandas as pd
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add backend to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_csv_to_parquet_conversion():
    """Test CSV to Parquet conversion"""
    logger.info("Testing CSV to Parquet conversion...")
    
    from lib.parquet_pipeline import csv_to_parquet, detect_csv_schema, validate_parquet_file
    
    # Use test dataset
    test_csv = "/mnt/optimizer_share/input/SENSEX_test_dataset.csv"
    if not os.path.exists(test_csv):
        logger.warning(f"Test CSV not found: {test_csv}")
        return False
    
    # Detect schema
    schema, strategy_cols = detect_csv_schema(test_csv)
    logger.info(f"Detected schema with {len(strategy_cols)} strategy columns")
    
    # Convert to Parquet
    output_parquet = "/tmp/test_sensex.parquet"
    success = csv_to_parquet(test_csv, output_parquet)
    
    if not success:
        logger.error("CSV to Parquet conversion failed")
        return False
    
    # Validate Parquet
    valid = validate_parquet_file(output_parquet)
    logger.info(f"Parquet validation: {'PASSED' if valid else 'FAILED'}")
    
    # Clean up
    if os.path.exists(output_parquet):
        os.remove(output_parquet)
    
    return valid

def test_arrow_memory_management():
    """Test Arrow memory management"""
    logger.info("Testing Arrow memory management...")
    
    from lib.arrow_connector import create_memory_pool, monitor_memory_usage
    
    # Create memory pool
    pool = create_memory_pool(size_gb=2.0)
    
    # Check memory usage
    usage = pool.get_usage()
    logger.info(f"Memory pool initialized: {usage['allocated_gb']:.2f}GB allocated")
    
    # Monitor system memory
    stats = monitor_memory_usage()
    logger.info(f"System memory: {stats['system']['available_gb']:.2f}GB available")
    
    # Cleanup
    pool.cleanup()
    
    return True

def test_cudf_calculations():
    """Test cuDF calculation functions"""
    logger.info("Testing cuDF calculations...")
    
    try:
        import cudf
        from lib.cudf_engine import (
            calculate_correlations_cudf,
            calculate_fitness_cudf,
            calculate_sharpe_ratio_cudf
        )
    except ImportError:
        logger.warning("cuDF not available, skipping GPU tests")
        return True
    
    # Create test data
    np.random.seed(42)
    n_days = 100
    n_strategies = 50
    
    data = {
        'Date': pd.date_range('2023-01-01', periods=n_days)
    }
    
    # Add strategy columns with random returns
    strategy_cols = []
    for i in range(n_strategies):
        col_name = f'Strategy_{i+1}'
        data[col_name] = np.random.randn(n_days) * 100
        strategy_cols.append(col_name)
    
    # Create cuDF DataFrame
    df = cudf.DataFrame(data)
    
    # Test correlation calculation
    logger.info("Testing correlation calculation...")
    corr_matrix = calculate_correlations_cudf(df, strategy_cols[:10])
    logger.info(f"Correlation matrix shape: {corr_matrix.shape}")
    
    # Test fitness calculation
    logger.info("Testing fitness calculation...")
    test_portfolio = strategy_cols[:5]
    metrics_config = {
        'roi_dd_ratio_weight': 0.35,
        'total_roi_weight': 0.25,
        'max_drawdown_weight': 0.15,
        'win_rate_weight': 0.15,
        'profit_factor_weight': 0.10
    }
    
    fitness_metrics = calculate_fitness_cudf(df, test_portfolio, metrics_config)
    logger.info(f"Fitness score: {fitness_metrics['fitness_score']:.4f}")
    logger.info(f"Total ROI: {fitness_metrics['total_roi']:.2f}")
    logger.info(f"Max Drawdown: {fitness_metrics['max_drawdown']:.2f}")
    
    # Test Sharpe ratio
    portfolio_returns = df[test_portfolio].sum(axis=1)
    sharpe = calculate_sharpe_ratio_cudf(portfolio_returns)
    logger.info(f"Sharpe ratio: {sharpe:.4f}")
    
    return True

def test_full_workflow():
    """Test complete Parquet/cuDF workflow"""
    logger.info("Testing full Parquet/cuDF workflow...")
    
    # Use test dataset
    test_csv = "/mnt/optimizer_share/input/SENSEX_test_dataset.csv"
    if not os.path.exists(test_csv):
        # Create a small test dataset
        logger.info("Creating test dataset...")
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=50)
        data = {'Date': dates}
        
        # Add 20 strategy columns
        for i in range(20):
            data[f'Strategy_{i+1}'] = np.random.randn(50) * 100 + np.random.randn() * 10
        
        df = pd.DataFrame(data)
        test_csv = "/tmp/test_data.csv"
        df.to_csv(test_csv, index=False)
    
    # Run workflow
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"/tmp/test_parquet_workflow_{timestamp}"
    
    cmd = f"python3 parquet_cudf_workflow.py --input {test_csv} --output {output_dir} --portfolio-size 10"
    
    logger.info(f"Running command: {cmd}")
    status = os.system(cmd)
    
    if status != 0:
        logger.error("Workflow execution failed")
        return False
    
    # Check output files
    expected_files = [
        'workflow_config.json',
        'workflow_results.json',
        'execution_summary_*.json',
        '*_result.json'
    ]
    
    output_files = os.listdir(output_dir) if os.path.exists(output_dir) else []
    logger.info(f"Generated {len(output_files)} output files")
    
    # Load and check results
    results_path = os.path.join(output_dir, 'workflow_results.json')
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            results = json.load(f)
        
        logger.info(f"Workflow status: {results['status']}")
        logger.info(f"Total execution time: {results['total_time']:.2f}s")
        
        # Check algorithm results
        if 'algorithms' in results:
            for algo, algo_results in results['algorithms'].items():
                if algo_results.get('status') == 'success':
                    logger.info(f"{algo}: fitness={algo_results['metrics']['fitness_score']:.4f}")
    
    # Clean up
    if os.path.exists(output_dir):
        import shutil
        shutil.rmtree(output_dir)
    
    if test_csv == "/tmp/test_data.csv" and os.path.exists(test_csv):
        os.remove(test_csv)
    
    return True

def test_enhanced_metrics():
    """Test enhanced financial metrics"""
    logger.info("Testing enhanced financial metrics...")
    
    try:
        from lib.cudf_engine.enhanced_metrics import (
            kelly_position_sizing,
            calculate_var_cvar_cudf,
            calculate_omega_ratio_cudf
        )
        import cudf
    except ImportError:
        logger.warning("Enhanced metrics or cuDF not available, skipping")
        return True
    
    # Test Kelly Criterion
    kelly_frac = kelly_position_sizing(
        win_rate=0.6,
        avg_win=150,
        avg_loss=100,
        max_allocation=0.25
    )
    logger.info(f"Kelly fraction: {kelly_frac:.4f}")
    
    # Test VaR/CVaR with cuDF
    returns = cudf.Series(np.random.randn(1000) * 100)
    var_cvar = calculate_var_cvar_cudf(returns)
    logger.info(f"95% VaR: {var_cvar['95%']['var']:.2f}")
    logger.info(f"95% CVaR: {var_cvar['95%']['cvar']:.2f}")
    
    # Test Omega ratio
    omega = calculate_omega_ratio_cudf(returns, threshold=0)
    logger.info(f"Omega ratio: {omega:.4f}")
    
    return True

def run_all_tests():
    """Run all tests"""
    tests = [
        ("CSV to Parquet Conversion", test_csv_to_parquet_conversion),
        ("Arrow Memory Management", test_arrow_memory_management),
        ("cuDF Calculations", test_cudf_calculations),
        ("Enhanced Metrics", test_enhanced_metrics),
        ("Full Workflow", test_full_workflow)
    ]
    
    results = []
    
    logger.info("=" * 60)
    logger.info("Running Parquet/Arrow/cuDF Tests")
    logger.info("=" * 60)
    
    for test_name, test_func in tests:
        logger.info(f"\n🧪 {test_name}")
        logger.info("-" * 40)
        
        try:
            success = test_func()
            results.append((test_name, success))
            logger.info(f"✅ {test_name}: PASSED" if success else f"❌ {test_name}: FAILED")
        except Exception as e:
            logger.error(f"❌ {test_name}: ERROR - {str(e)}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Test Summary")
    logger.info("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "PASSED" if success else "FAILED"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nTotal: {passed}/{total} tests passed")
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)