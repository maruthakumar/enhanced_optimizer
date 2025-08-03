#!/usr/bin/env python3
"""
Test script for algorithm retrofit validation (Story 1.1R)
Tests both legacy numpy and new cuDF interfaces
"""

import sys
import os
import time
import numpy as np
import pandas as pd
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add backend to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Try to import cuDF
try:
    import cudf
    CUDF_AVAILABLE = True
    logger.info("✅ cuDF available for testing")
except (ImportError, RuntimeError) as e:
    CUDF_AVAILABLE = False
    logger.warning(f"⚠️ cuDF not available ({str(e)}), testing CPU mode only")

from algorithms.genetic_algorithm import GeneticAlgorithm
from algorithms.fitness_functions import FitnessCalculator

def create_test_data():
    """Create synthetic test data for validation"""
    logger.info("Creating synthetic test data...")
    
    # Create synthetic strategy data (100 days, 50 strategies)
    np.random.seed(42)  # For reproducible results
    num_days = 100
    num_strategies = 50
    
    # Generate realistic-looking financial returns
    daily_returns = np.random.normal(0.01, 0.02, (num_days, num_strategies))  # 1% mean, 2% std
    
    # Create column names
    strategy_names = [f"Strategy_{i:03d}" for i in range(num_strategies)]
    dates = pd.date_range('2023-01-01', periods=num_days, freq='D')
    
    # Create pandas DataFrame
    df_pandas = pd.DataFrame(daily_returns, columns=strategy_names, index=dates)
    df_pandas.reset_index(inplace=True)
    df_pandas.rename(columns={'index': 'Date'}, inplace=True)
    
    logger.info(f"Created test data: {num_days} days, {num_strategies} strategies")
    return daily_returns, df_pandas, strategy_names

def test_legacy_numpy_interface():
    """Test legacy numpy interface"""
    logger.info("\n🔄 Testing legacy numpy interface...")
    
    daily_matrix, _, _ = create_test_data()
    
    # Initialize algorithm with GPU disabled for legacy test
    ga = GeneticAlgorithm(use_gpu=False)
    
    # Create legacy fitness function
    fitness_calc = FitnessCalculator(use_gpu=False)
    fitness_function = fitness_calc.create_fitness_function(daily_matrix, 'numpy')
    
    # Run optimization
    start_time = time.time()
    result = ga.optimize(
        data=daily_matrix,
        portfolio_size=10,
        fitness_function=fitness_function
    )
    execution_time = time.time() - start_time
    
    logger.info(f"✅ Legacy numpy test completed in {execution_time:.3f}s")
    logger.info(f"   Best fitness: {result['best_fitness']:.6f}")
    logger.info(f"   Portfolio size: {len(result['best_portfolio'])}")
    logger.info(f"   Data type: {result['data_type']}")
    logger.info(f"   GPU accelerated: {result['gpu_accelerated']}")
    
    return result

def test_cudf_gpu_interface():
    """Test new cuDF GPU interface"""
    if not CUDF_AVAILABLE:
        logger.warning("⚠️ Skipping cuDF test - not available")
        return None
        
    logger.info("\n🔄 Testing cuDF GPU interface...")
    
    _, df_pandas, strategy_names = create_test_data()
    
    # Convert to cuDF
    df_cudf = cudf.from_pandas(df_pandas)
    
    # Initialize algorithm with GPU enabled
    ga = GeneticAlgorithm(use_gpu=True)
    
    # Run optimization (fitness function auto-created)
    start_time = time.time()
    result = ga.optimize(
        data=df_cudf,
        portfolio_size=10
    )
    execution_time = time.time() - start_time
    
    logger.info(f"✅ cuDF GPU test completed in {execution_time:.3f}s")
    logger.info(f"   Best fitness: {result['best_fitness']:.6f}")
    logger.info(f"   Portfolio size: {len(result['best_portfolio'])}")
    logger.info(f"   Data type: {result['data_type']}")
    logger.info(f"   GPU accelerated: {result['gpu_accelerated']}")
    
    # Print sample portfolio strategies
    portfolio_strategies = result['best_portfolio'][:5]  # First 5
    logger.info(f"   Sample portfolio strategies: {portfolio_strategies}")
    
    return result

def compare_results(numpy_result, cudf_result):
    """Compare results between numpy and cuDF implementations"""
    if numpy_result is None or cudf_result is None:
        logger.warning("⚠️ Cannot compare - one or both results missing")
        return
        
    logger.info("\n📊 Comparing results...")
    
    numpy_fitness = numpy_result['best_fitness']
    cudf_fitness = cudf_result['best_fitness']
    
    # Calculate accuracy tolerance
    accuracy_tolerance = 0.001  # ±0.001% as per story requirements
    fitness_diff = abs(numpy_fitness - cudf_fitness)
    relative_diff = fitness_diff / abs(numpy_fitness) if numpy_fitness != 0 else 0
    
    logger.info(f"   Numpy fitness: {numpy_fitness:.6f}")
    logger.info(f"   cuDF fitness:  {cudf_fitness:.6f}")
    logger.info(f"   Absolute diff: {fitness_diff:.6f}")
    logger.info(f"   Relative diff: {relative_diff:.6f} ({relative_diff*100:.4f}%)")
    
    if relative_diff <= accuracy_tolerance:
        logger.info("✅ ACCURACY TEST PASSED - Within ±0.001% tolerance")
    else:
        logger.warning(f"⚠️ ACCURACY TEST FAILED - Exceeds {accuracy_tolerance*100:.3f}% tolerance")
    
    # Compare execution times
    numpy_time = numpy_result['execution_time']
    cudf_time = cudf_result['execution_time']
    speedup = numpy_time / cudf_time if cudf_time > 0 else 1
    
    logger.info(f"   Numpy time: {numpy_time:.3f}s")
    logger.info(f"   cuDF time:  {cudf_time:.3f}s")
    logger.info(f"   Speedup:    {speedup:.2f}x")

def main():
    """Main test function"""
    logger.info("🚀 Starting Algorithm Retrofit Validation (Story 1.1R)")
    logger.info("=" * 60)
    
    try:
        # Test legacy interface
        numpy_result = test_legacy_numpy_interface()
        
        # Test new cuDF interface
        cudf_result = test_cudf_gpu_interface()
        
        # Compare results
        compare_results(numpy_result, cudf_result)
        
        logger.info("\n" + "=" * 60)
        logger.info("🎉 Algorithm retrofit validation completed successfully!")
        
        if cudf_result:
            logger.info(f"✅ GPU acceleration: {cudf_result['gpu_accelerated']}")
            logger.info(f"✅ cuDF compatibility: {cudf_result['data_type'] == 'cudf'}")
            logger.info(f"✅ Legacy compatibility: {numpy_result['data_type'] == 'numpy'}")
        
    except Exception as e:
        logger.error(f"❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)