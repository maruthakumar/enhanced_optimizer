#!/usr/bin/env python3
"""
Test script for the Correlation Matrix Calculator module.
Tests basic functionality, GPU acceleration, and caching.
"""

import sys
import os
import numpy as np
import time
import logging

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.correlation import CorrelationMatrixCalculator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_test_data(n_days=82, n_strategies=100):
    """Generate synthetic trading data for testing"""
    logger.info(f"Generating test data: {n_days} days x {n_strategies} strategies")
    
    # Generate random daily returns with some correlation structure
    # Create base returns
    base_returns = np.random.randn(n_days, 5) * 0.02  # 5 base factors
    
    # Generate strategy returns as combinations of base factors
    weights = np.random.randn(5, n_strategies)
    strategy_returns = base_returns @ weights
    
    # Add noise
    noise = np.random.randn(n_days, n_strategies) * 0.01
    daily_matrix = strategy_returns + noise
    
    return daily_matrix


def test_basic_functionality():
    """Test basic correlation calculation"""
    logger.info("\n" + "="*60)
    logger.info("Testing basic functionality")
    logger.info("="*60)
    
    # Initialize calculator
    config_path = "/mnt/optimizer_share/config/production_config.ini"
    calculator = CorrelationMatrixCalculator(config_path)
    
    # Generate small test data
    daily_matrix = generate_test_data(n_days=50, n_strategies=10)
    
    # Test pairwise correlation
    portfolio = np.array([0, 2, 4, 6, 8])
    avg_corr = calculator.compute_avg_pairwise_correlation(daily_matrix, portfolio)
    logger.info(f"✅ Average pairwise correlation: {avg_corr:.4f}")
    
    # Test fitness evaluation
    base_fitness = 1.5
    adjusted_fitness, penalty = calculator.evaluate_fitness_with_correlation(
        base_fitness, daily_matrix, portfolio
    )
    logger.info(f"✅ Base fitness: {base_fitness:.4f}")
    logger.info(f"✅ Correlation penalty: {penalty:.4f}")
    logger.info(f"✅ Adjusted fitness: {adjusted_fitness:.4f}")
    
    # Verify legacy formula
    expected_fitness = base_fitness * (1 - avg_corr * calculator.config.penalty_weight)
    assert abs(adjusted_fitness - expected_fitness) < 1e-6, "Legacy formula not preserved!"
    logger.info("✅ Legacy formula verified")


def test_full_matrix_calculation():
    """Test full correlation matrix calculation"""
    logger.info("\n" + "="*60)
    logger.info("Testing full matrix calculation")
    logger.info("="*60)
    
    calculator = CorrelationMatrixCalculator()
    
    # Test different sizes
    sizes = [50, 100, 500]
    for n_strategies in sizes:
        daily_matrix = generate_test_data(n_strategies=n_strategies)
        
        start_time = time.time()
        corr_matrix = calculator.calculate_full_correlation_matrix(daily_matrix)
        elapsed = time.time() - start_time
        
        logger.info(f"✅ Matrix size {n_strategies}x{n_strategies} calculated in {elapsed:.3f}s")
        
        # Validate matrix properties
        assert corr_matrix.shape == (n_strategies, n_strategies)
        assert np.allclose(np.diag(corr_matrix), 1.0), "Diagonal should be 1"
        assert np.allclose(corr_matrix, corr_matrix.T), "Matrix should be symmetric"
        assert np.all((corr_matrix >= -1) & (corr_matrix <= 1)), "Values should be in [-1, 1]"


def test_gpu_acceleration():
    """Test GPU acceleration if available"""
    logger.info("\n" + "="*60)
    logger.info("Testing GPU acceleration")
    logger.info("="*60)
    
    calculator = CorrelationMatrixCalculator()
    
    # Check GPU availability
    gpu_info = calculator.gpu_accelerator.get_gpu_info()
    logger.info(f"GPU Backend: {gpu_info.get('backend', 'None')}")
    logger.info(f"GPU Available: {gpu_info.get('is_available', False)}")
    
    if gpu_info.get('is_available'):
        logger.info(f"Device: {gpu_info.get('device_name', 'Unknown')}")
        logger.info(f"Memory: {gpu_info.get('memory_free_gb', 0):.1f}/{gpu_info.get('memory_total_gb', 0):.1f} GB")
        
        # Compare CPU vs GPU performance
        daily_matrix = generate_test_data(n_strategies=2000)
        
        # Force CPU calculation
        calculator.config.gpu_acceleration = False
        start_time = time.time()
        cpu_result = calculator.calculate_full_correlation_matrix(daily_matrix)
        cpu_time = time.time() - start_time
        
        # Clear cache
        calculator.clear_cache()
        
        # Enable GPU calculation
        calculator.config.gpu_acceleration = True
        start_time = time.time()
        gpu_result = calculator.calculate_full_correlation_matrix(daily_matrix)
        gpu_time = time.time() - start_time
        
        logger.info(f"✅ CPU time: {cpu_time:.3f}s")
        logger.info(f"✅ GPU time: {gpu_time:.3f}s")
        logger.info(f"✅ Speedup: {cpu_time/gpu_time:.2f}x")
        
        # Verify results match
        assert np.allclose(cpu_result, gpu_result, rtol=1e-5), "CPU and GPU results should match"
    else:
        logger.info("⚠️ No GPU acceleration available")


def test_caching():
    """Test caching functionality"""
    logger.info("\n" + "="*60)
    logger.info("Testing caching")
    logger.info("="*60)
    
    calculator = CorrelationMatrixCalculator()
    calculator.config.cache_enabled = True
    
    # Generate test data
    daily_matrix = generate_test_data(n_strategies=100)
    
    # First calculation
    start_time = time.time()
    result1 = calculator.calculate_full_correlation_matrix(daily_matrix)
    time1 = time.time() - start_time
    
    # Second calculation (should use cache)
    start_time = time.time()
    result2 = calculator.calculate_full_correlation_matrix(daily_matrix)
    time2 = time.time() - start_time
    
    logger.info(f"✅ First calculation: {time1:.3f}s")
    logger.info(f"✅ Cached calculation: {time2:.3f}s")
    logger.info(f"✅ Cache speedup: {time1/time2:.1f}x")
    
    # Verify results are identical
    assert np.array_equal(result1, result2), "Cached result should be identical"
    
    # Check cache info
    cache_info = calculator.get_cache_info()
    logger.info(f"✅ Cached matrices: {cache_info['cached_matrices']}")


def test_correlation_analysis():
    """Test correlation matrix analysis features"""
    logger.info("\n" + "="*60)
    logger.info("Testing correlation analysis")
    logger.info("="*60)
    
    calculator = CorrelationMatrixCalculator()
    
    # Generate data with known correlations
    n_strategies = 50
    daily_matrix = generate_test_data(n_strategies=n_strategies)
    
    # Calculate correlation matrix
    corr_matrix = calculator.calculate_full_correlation_matrix(daily_matrix)
    
    # Analyze the matrix
    analysis = calculator.analyze_correlation_matrix(corr_matrix)
    
    logger.info(f"✅ Average correlation: {analysis['avg_correlation']:.4f}")
    logger.info(f"✅ Max correlation: {analysis['max_correlation']:.4f}")
    logger.info(f"✅ Min correlation: {analysis['min_correlation']:.4f}")
    logger.info(f"✅ Std deviation: {analysis['std_correlation']:.4f}")
    
    # Show correlation distribution
    logger.info("✅ Correlation distribution:")
    for range_str, count in analysis['correlation_distribution'].items():
        logger.info(f"   {range_str}: {count} pairs")
    
    # Show top correlated pairs
    high_corr_pairs = analysis['high_correlation_pairs'][:5]
    if high_corr_pairs:
        logger.info("✅ Top 5 correlated pairs:")
        for i, j, corr in high_corr_pairs:
            logger.info(f"   Strategy {i} & {j}: {corr:.4f}")


def test_chunked_processing():
    """Test chunked processing for large matrices"""
    logger.info("\n" + "="*60)
    logger.info("Testing chunked processing")
    logger.info("="*60)
    
    calculator = CorrelationMatrixCalculator()
    calculator.config.matrix_chunk_size = 100  # Force chunking
    
    # Generate large matrix
    daily_matrix = generate_test_data(n_strategies=250)
    
    # Calculate with chunking
    start_time = time.time()
    corr_matrix = calculator.calculate_chunked_correlation_matrix(daily_matrix)
    elapsed = time.time() - start_time
    
    logger.info(f"✅ Chunked calculation completed in {elapsed:.3f}s")
    
    # Validate result
    assert corr_matrix.shape == (250, 250)
    assert np.allclose(np.diag(corr_matrix), 1.0)
    assert np.allclose(corr_matrix, corr_matrix.T)


def test_memory_estimation():
    """Test memory usage estimation"""
    logger.info("\n" + "="*60)
    logger.info("Testing memory estimation")
    logger.info("="*60)
    
    calculator = CorrelationMatrixCalculator()
    
    # Test different matrix sizes
    test_sizes = [1000, 5000, 10000, 25000, 28044]  # Include production size
    
    for n_strategies in test_sizes:
        memory_gb, recommendation = calculator.gpu_accelerator.estimate_memory_usage(n_strategies)
        logger.info(f"✅ {n_strategies} strategies: {memory_gb:.2f} GB - {recommendation}")


def main():
    """Run all tests"""
    logger.info("🚀 Starting Correlation Module Tests")
    logger.info("="*80)
    
    try:
        test_basic_functionality()
        test_full_matrix_calculation()
        test_gpu_acceleration()
        test_caching()
        test_correlation_analysis()
        test_chunked_processing()
        test_memory_estimation()
        
        logger.info("\n" + "="*80)
        logger.info("✅ All tests passed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()