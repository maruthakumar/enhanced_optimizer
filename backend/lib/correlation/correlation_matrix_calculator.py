import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
import configparser
import os
from dataclasses import dataclass
import hashlib
import json
from .gpu_accelerator import GPUAccelerator

logger = logging.getLogger(__name__)


@dataclass
class CorrelationConfig:
    """Configuration for correlation penalty calculations"""
    penalty_weight: float = 0.1
    cache_enabled: bool = True
    gpu_acceleration: bool = True
    matrix_chunk_size: int = 5000  # For large matrices, process in chunks
    correlation_threshold: float = 0.7  # For analysis purposes


class CorrelationMatrixCalculator:
    """
    Standalone module for calculating correlation matrices with HeavyDB GPU acceleration.
    Preserves legacy correlation penalty logic while providing enhanced performance.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self._cache: Dict[str, np.ndarray] = {}
        self.gpu_accelerator = GPUAccelerator()
        self._gpu_available = self.gpu_accelerator.is_available
        
    def _load_config(self, config_path: Optional[str]) -> CorrelationConfig:
        """Load configuration from file or use defaults"""
        config = CorrelationConfig()
        
        if config_path and os.path.exists(config_path):
            parser = configparser.ConfigParser()
            parser.read(config_path)
            
            if 'correlation' in parser:
                config.penalty_weight = parser.getfloat('correlation', 'penalty_weight', fallback=0.1)
                config.cache_enabled = parser.getboolean('correlation', 'cache_enabled', fallback=True)
                config.gpu_acceleration = parser.getboolean('correlation', 'gpu_acceleration', fallback=True)
                config.matrix_chunk_size = parser.getint('correlation', 'matrix_chunk_size', fallback=5000)
                config.correlation_threshold = parser.getfloat('correlation', 'correlation_threshold', fallback=0.7)
        
        logger.info(f"Correlation config loaded: penalty_weight={config.penalty_weight}, "
                   f"gpu_acceleration={config.gpu_acceleration}")
        return config
    
    
    def _generate_cache_key(self, data: np.ndarray) -> str:
        """Generate a unique cache key for the data"""
        # Use hash of data shape and sample values for efficiency
        data_summary = f"{data.shape}_{data[0,0]:.6f}_{data[-1,-1]:.6f}_{np.sum(data):.6f}"
        return hashlib.md5(data_summary.encode()).hexdigest()
    
    def calculate_full_correlation_matrix(self, daily_matrix: np.ndarray) -> np.ndarray:
        """
        Calculate the full correlation matrix for all strategies.
        Uses GPU acceleration if available and configured.
        """
        # Check cache first
        if self.config.cache_enabled:
            cache_key = self._generate_cache_key(daily_matrix)
            if cache_key in self._cache:
                logger.info("📊 Using cached correlation matrix")
                return self._cache[cache_key]
        
        n_strategies = daily_matrix.shape[1]
        logger.info(f"📊 Calculating correlation matrix for {n_strategies} strategies")
        
        # Estimate memory usage
        memory_gb, recommendation = self.gpu_accelerator.estimate_memory_usage(n_strategies)
        logger.info(f"💾 Estimated memory usage: {memory_gb:.2f} GB - {recommendation}")
        
        if self.config.gpu_acceleration and self._gpu_available and n_strategies > 1000:
            correlation_matrix = self.gpu_accelerator.calculate_correlation_matrix(daily_matrix)
        else:
            correlation_matrix = self._calculate_with_numpy(daily_matrix)
        
        # Cache the result
        if self.config.cache_enabled:
            self._cache[cache_key] = correlation_matrix
            
        return correlation_matrix
    
    def _calculate_with_numpy(self, daily_matrix: np.ndarray) -> np.ndarray:
        """Standard NumPy correlation calculation"""
        logger.info("📊 Using NumPy for correlation calculation")
        return np.corrcoef(daily_matrix.T)
    
    def calculate_correlation_matrix(self, daily_matrix: np.ndarray) -> np.ndarray:
        """
        Main public method to calculate correlation matrix.
        Alias for calculate_full_correlation_matrix for backward compatibility.
        """
        return self.calculate_full_correlation_matrix(daily_matrix)
    
    
    def calculate_chunked_correlation_matrix(self, daily_matrix: np.ndarray) -> np.ndarray:
        """Calculate correlation matrix in chunks for very large datasets"""
        n_strategies = daily_matrix.shape[1]
        chunk_size = self.config.matrix_chunk_size
        correlation_matrix = np.zeros((n_strategies, n_strategies))
        
        logger.info(f"📊 Processing {n_strategies} strategies in chunks of {chunk_size}")
        
        for i in range(0, n_strategies, chunk_size):
            for j in range(i, n_strategies, chunk_size):
                end_i = min(i + chunk_size, n_strategies)
                end_j = min(j + chunk_size, n_strategies)
                
                # Calculate correlation for this chunk
                if i == j:
                    # Same chunk - calculate correlation within the chunk
                    chunk_data = daily_matrix[:, i:end_i]
                    if self._gpu_available and self.config.gpu_acceleration:
                        chunk_corr = self.gpu_accelerator.calculate_correlation_matrix(chunk_data)
                    else:
                        chunk_corr = np.corrcoef(chunk_data.T)
                    correlation_matrix[i:end_i, j:end_j] = chunk_corr
                else:
                    # Different chunks - calculate cross-correlation
                    chunk_i_data = daily_matrix[:, i:end_i]
                    chunk_j_data = daily_matrix[:, j:end_j]
                    
                    if self._gpu_available and self.config.gpu_acceleration:
                        # Combine chunks for GPU calculation
                        combined_data = np.hstack([chunk_i_data, chunk_j_data])
                        full_corr = self.gpu_accelerator.calculate_correlation_matrix(combined_data)
                        # Extract the cross-correlation part
                        n_i = end_i - i
                        correlation_matrix[i:end_i, j:end_j] = full_corr[:n_i, n_i:]
                        correlation_matrix[j:end_j, i:end_i] = full_corr[:n_i, n_i:].T
                    else:
                        # NumPy cross-correlation
                        chunk_corr = np.corrcoef(chunk_i_data.T, chunk_j_data.T)
                        n_i = end_i - i
                        correlation_matrix[i:end_i, j:end_j] = chunk_corr[:n_i, n_i:]
                        correlation_matrix[j:end_j, i:end_i] = chunk_corr[:n_i, n_i:].T
        
        return correlation_matrix
    
    def compute_avg_pairwise_correlation(self, daily_matrix: np.ndarray, portfolio: np.ndarray) -> float:
        """
        Compute average pairwise correlation for a portfolio.
        Preserves legacy logic from evaluate_fitness_with_correlation.
        """
        if len(portfolio) < 2:
            return 0.0
        
        portfolio_data = daily_matrix[:, portfolio]
        correlation_matrix = np.corrcoef(portfolio_data.T)
        
        # Calculate average correlation from upper triangle
        upper_triangle = np.triu(correlation_matrix, k=1)
        correlations = upper_triangle[upper_triangle != 0]
        avg_correlation = np.mean(np.abs(correlations)) if len(correlations) > 0 else 0.0
        
        return avg_correlation
    
    def evaluate_fitness_with_correlation(self, base_fitness: float, daily_matrix: np.ndarray, 
                                        portfolio: np.ndarray) -> Tuple[float, float]:
        """
        Apply correlation penalty to base fitness.
        Preserves exact legacy formula: fitness = base_fitness * (1 - avg_pairwise_correlation * correlation_penalty_weight)
        
        Returns: (adjusted_fitness, correlation_penalty)
        """
        avg_correlation = self.compute_avg_pairwise_correlation(daily_matrix, portfolio)
        correlation_penalty = avg_correlation * self.config.penalty_weight
        
        # Apply legacy formula
        adjusted_fitness = base_fitness * (1 - avg_correlation * self.config.penalty_weight)
        
        logger.info(f"📊 Correlation penalty applied: avg_corr={avg_correlation:.3f}, "
                   f"penalty={correlation_penalty:.6f}, adjusted_fitness={adjusted_fitness:.6f}")
        
        return adjusted_fitness, correlation_penalty
    
    def analyze_correlation_matrix(self, correlation_matrix: np.ndarray) -> Dict[str, any]:
        """Analyze correlation matrix for insights"""
        # Get upper triangle (excluding diagonal)
        upper_triangle = np.triu(correlation_matrix, k=1)
        correlations = upper_triangle[upper_triangle != 0]
        
        analysis = {
            'avg_correlation': np.mean(np.abs(correlations)),
            'max_correlation': np.max(np.abs(correlations)),
            'min_correlation': np.min(np.abs(correlations)),
            'std_correlation': np.std(correlations),
            'high_correlation_pairs': self._find_high_correlation_pairs(correlation_matrix),
            'correlation_distribution': self._get_correlation_distribution(correlations)
        }
        
        return analysis
    
    def _find_high_correlation_pairs(self, correlation_matrix: np.ndarray) -> List[Tuple[int, int, float]]:
        """Find strategy pairs with correlation above threshold"""
        high_corr_pairs = []
        n = correlation_matrix.shape[0]
        
        for i in range(n):
            for j in range(i + 1, n):
                corr = abs(correlation_matrix[i, j])
                if corr > self.config.correlation_threshold:
                    high_corr_pairs.append((i, j, corr))
        
        # Sort by correlation value
        high_corr_pairs.sort(key=lambda x: x[2], reverse=True)
        return high_corr_pairs[:100]  # Return top 100
    
    def _get_correlation_distribution(self, correlations: np.ndarray) -> Dict[str, int]:
        """Get distribution of correlation values"""
        abs_corr = np.abs(correlations)
        return {
            '0.0-0.2': np.sum((abs_corr >= 0.0) & (abs_corr < 0.2)),
            '0.2-0.4': np.sum((abs_corr >= 0.2) & (abs_corr < 0.4)),
            '0.4-0.6': np.sum((abs_corr >= 0.4) & (abs_corr < 0.6)),
            '0.6-0.8': np.sum((abs_corr >= 0.6) & (abs_corr < 0.8)),
            '0.8-1.0': np.sum((abs_corr >= 0.8) & (abs_corr <= 1.0))
        }
    
    def clear_cache(self):
        """Clear the correlation matrix cache"""
        self._cache.clear()
        logger.info("📊 Correlation matrix cache cleared")
    
    def get_cache_info(self) -> Dict[str, any]:
        """Get information about the cache"""
        return {
            'cache_enabled': self.config.cache_enabled,
            'cached_matrices': len(self._cache),
            'cache_keys': list(self._cache.keys())
        }
    
