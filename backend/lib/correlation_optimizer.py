"""
Correlation calculation optimizer for large matrices
Handles configuration and fallback strategies
"""

import os
import configparser
import logging
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

def get_correlation_config() -> Dict[str, Any]:
    """
    Get correlation optimization configuration from ini file
    
    Returns:
        Dictionary with correlation optimization parameters
    """
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'config',
        'heavydb_optimization.ini'
    )
    
    # Default values
    config = {
        'chunk_size': 50,
        'max_correlations_per_query': 500,
        'timeout': 300,
        'adaptive_chunking': True,
        'large_matrix_chunk_size': 25,
        'huge_matrix_chunk_size': 10,
        'cpu_fallback': True
    }
    
    # Read from config file if exists
    if os.path.exists(config_path):
        parser = configparser.ConfigParser()
        parser.read(config_path)
        
        if 'correlation_optimization' in parser:
            section = parser['correlation_optimization']
            config['chunk_size'] = section.getint('correlation_chunk_size', 50)
            config['max_correlations_per_query'] = section.getint('max_correlations_per_query', 500)
            config['timeout'] = section.getint('correlation_query_timeout', 300)
            config['adaptive_chunking'] = section.getboolean('adaptive_chunking', True)
            config['large_matrix_chunk_size'] = section.getint('large_matrix_chunk_size', 25)
            config['huge_matrix_chunk_size'] = section.getint('huge_matrix_chunk_size', 10)
        
        if 'gpu_optimization' in parser:
            section = parser['gpu_optimization']
            config['cpu_fallback'] = section.getboolean('cpu_correlation_fallback', True)
    
    return config

def get_optimal_chunk_size(n_strategies: int, config: Dict[str, Any]) -> int:
    """
    Determine optimal chunk size based on matrix size
    
    Args:
        n_strategies: Number of strategies
        config: Correlation configuration
        
    Returns:
        Optimal chunk size
    """
    if not config.get('adaptive_chunking', True):
        return config['chunk_size']
    
    if n_strategies > 5000:
        return config['huge_matrix_chunk_size']
    elif n_strategies > 1000:
        return config['large_matrix_chunk_size']
    else:
        return config['chunk_size']

def calculate_correlation_cpu_fallback(df: pd.DataFrame, 
                                     strategy_columns: list) -> Optional[np.ndarray]:
    """
    CPU-based correlation calculation fallback
    
    Args:
        df: DataFrame with strategy data
        strategy_columns: List of strategy column names
        
    Returns:
        Correlation matrix or None on error
    """
    try:
        logger.info("📊 Using CPU fallback for correlation calculation")
        
        # Extract strategy data
        strategy_data = df[strategy_columns]
        
        # Calculate correlation using pandas (efficient C implementation)
        start_time = pd.Timestamp.now()
        corr_matrix = strategy_data.corr().values
        elapsed = (pd.Timestamp.now() - start_time).total_seconds()
        
        logger.info(f"✅ CPU correlation calculated in {elapsed:.2f}s")
        
        # Ensure diagonal is exactly 1.0
        np.fill_diagonal(corr_matrix, 1.0)
        
        return corr_matrix
        
    except Exception as e:
        logger.error(f"❌ CPU correlation calculation failed: {e}")
        return None

def estimate_correlation_memory_usage(n_strategies: int) -> Dict[str, float]:
    """
    Estimate memory usage for correlation matrix calculation
    
    Args:
        n_strategies: Number of strategies
        
    Returns:
        Dictionary with memory estimates in GB
    """
    # Correlation matrix size (float64)
    matrix_size_gb = (n_strategies * n_strategies * 8) / (1024**3)
    
    # Working memory (approximately 2x matrix size for calculations)
    working_memory_gb = matrix_size_gb * 2
    
    # Peak memory (including temporary arrays)
    peak_memory_gb = matrix_size_gb * 3
    
    return {
        'matrix_size_gb': matrix_size_gb,
        'working_memory_gb': working_memory_gb,
        'peak_memory_gb': peak_memory_gb,
        'strategies': n_strategies
    }

def validate_correlation_matrix(corr_matrix: np.ndarray) -> Dict[str, Any]:
    """
    Validate correlation matrix properties
    
    Args:
        corr_matrix: Correlation matrix to validate
        
    Returns:
        Dictionary with validation results
    """
    n = corr_matrix.shape[0]
    
    # Check if square
    is_square = corr_matrix.shape[0] == corr_matrix.shape[1]
    
    # Check if symmetric
    is_symmetric = np.allclose(corr_matrix, corr_matrix.T, rtol=1e-5, atol=1e-8)
    
    # Check diagonal values
    diagonal_ones = np.allclose(np.diag(corr_matrix), 1.0, rtol=1e-5, atol=1e-8)
    
    # Check value range
    min_val = np.min(corr_matrix)
    max_val = np.max(corr_matrix)
    valid_range = (min_val >= -1.0) and (max_val <= 1.0)
    
    # Check for NaN values
    has_nan = np.any(np.isnan(corr_matrix))
    
    # Count high correlations
    upper_triangle = np.triu(corr_matrix, k=1)
    high_corr_count = np.sum(np.abs(upper_triangle) > 0.7)
    
    return {
        'is_valid': is_square and is_symmetric and diagonal_ones and valid_range and not has_nan,
        'is_square': is_square,
        'is_symmetric': is_symmetric,
        'diagonal_ones': diagonal_ones,
        'valid_range': valid_range,
        'has_nan': has_nan,
        'min_correlation': min_val,
        'max_correlation': max_val,
        'high_correlation_pairs': high_corr_count,
        'matrix_size': n
    }

def optimize_correlation_calculation(table_name: str,
                                   connection: Optional[Any] = None,
                                   config: Optional[Dict[str, Any]] = None) -> Optional[np.ndarray]:
    """
    Optimize correlation calculation based on matrix size and configuration
    
    Args:
        table_name: HeavyDB table name
        connection: Optional HeavyDB connection
        config: Optional configuration override
    
    Returns:
        Correlation matrix or None if failed
    """
    from .heavydb_connector import calculate_correlations_gpu
    
    # Use provided config or get default
    if config is None:
        config = get_correlation_config()
    
    # Determine optimal approach based on configuration
    chunk_size = get_optimal_chunk_size(0, config)  # Will be determined by GPU function
    
    logger.info(f"🚀 Starting optimized correlation calculation with config: {config}")
    
    # Try GPU calculation
    try:
        return calculate_correlations_gpu(
            table_name,
            connection=connection,
            chunk_size=chunk_size,
            max_query_size=config.get('max_correlations_per_query', 500)
        )
    except Exception as e:
        logger.error(f"❌ Optimized correlation calculation failed: {e}")
        return None

# Export configuration function for use in other modules
__all__ = [
    'get_correlation_config',
    'get_optimal_chunk_size',
    'calculate_correlation_cpu_fallback',
    'estimate_correlation_memory_usage',
    'validate_correlation_matrix',
    'optimize_correlation_calculation'
]