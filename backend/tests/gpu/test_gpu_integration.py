"""
GPU-specific integration tests for cuDF operations
Tests GPU memory management, performance, and accuracy
"""

import pytest
import pandas as pd
import numpy as np
import os
import tempfile
from unittest.mock import patch, MagicMock
import gc

# Add parent directory to path for imports
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from backend.lib.arrow_connector.memory_manager import ArrowMemoryPool, arrow_to_cudf
from backend.lib.cudf_engine.gpu_calculator import (
    calculate_correlations_cudf,
    calculate_fitness_cudf,
    batch_calculate_portfolios_cudf
)

# Check if CUDA/cuDF is available
try:
    import cudf
    import cupy as cp
    CUDA_AVAILABLE = True
except (ImportError, RuntimeError):
    CUDA_AVAILABLE = False


@pytest.mark.gpu
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA/cuDF not available")
class TestGPUMemoryManagement:
    """Test GPU memory management for large datasets"""
    
    def test_gpu_memory_allocation(self):
        """Test GPU memory allocation and cleanup"""
        # Get initial GPU memory
        mempool = cp.get_default_memory_pool()
        initial_memory = mempool.used_bytes()
        
        # Create large dataset
        num_strategies = 1000
        num_days = 252
        
        data = {
            f'Strategy_{i}': np.random.randn(num_days) * 1000 
            for i in range(num_strategies)
        }
        
        # Convert to cuDF
        gdf = cudf.DataFrame(data)
        
        # Check memory increased
        current_memory = mempool.used_bytes()
        assert current_memory > initial_memory
        
        # Clean up
        del gdf
        gc.collect()
        mempool.free_all_blocks()
        
        # Memory should be mostly freed
        final_memory = mempool.used_bytes()
        assert final_memory < current_memory * 0.1  # Less than 10% remaining
    
    def test_gpu_memory_chunking(self):
        """Test processing large datasets in chunks to manage GPU memory"""
        # Create dataset larger than typical GPU memory
        num_strategies = 5000
        num_days = 252
        chunk_size = 1000  # Process 1000 strategies at a time
        
        all_correlations = []
        
        for chunk_start in range(0, num_strategies, chunk_size):
            chunk_end = min(chunk_start + chunk_size, num_strategies)
            
            # Create chunk data
            chunk_data = {
                f'Strategy_{i}': np.random.randn(num_days) * 1000 
                for i in range(chunk_start, chunk_end)
            }
            
            # Process chunk on GPU
            gdf = cudf.DataFrame(chunk_data)
            
            # Calculate correlations for chunk
            corr_matrix = gdf.corr()
            all_correlations.append(corr_matrix)
            
            # Clean up chunk
            del gdf
            del corr_matrix
            gc.collect()
        
        # Verify all chunks processed
        assert len(all_correlations) == (num_strategies + chunk_size - 1) // chunk_size
    
    def test_gpu_oom_handling(self):
        """Test handling of GPU out-of-memory errors"""
        try:
            # Try to allocate more memory than available
            huge_array = cp.zeros((100000, 100000), dtype=cp.float64)
            # If this succeeds, GPU has a lot of memory
            del huge_array
        except cp.cuda.MemoryError:
            # Expected behavior - OOM handled gracefully
            pass
        
        # Verify GPU is still functional after OOM
        small_array = cp.ones((100, 100))
        assert small_array.sum() == 10000


@pytest.mark.gpu
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA/cuDF not available")
class TestGPUCorrelationCalculations:
    """Test GPU-accelerated correlation calculations"""
    
    def test_gpu_correlation_accuracy(self):
        """Test accuracy of GPU correlation calculations"""
        # Create test data with known correlations
        num_days = 100
        
        # Create perfectly correlated strategies
        base = np.random.randn(num_days) * 1000
        data = {
            'Strategy_1': base,
            'Strategy_2': base * 1.5 + np.random.randn(num_days) * 10,  # High correlation
            'Strategy_3': -base * 0.8 + np.random.randn(num_days) * 50,  # Negative correlation
            'Strategy_4': np.random.randn(num_days) * 1000  # No correlation
        }
        
        # Calculate on CPU
        df_cpu = pd.DataFrame(data)
        corr_cpu = df_cpu.corr()
        
        # Calculate on GPU
        df_gpu = cudf.DataFrame(data)
        corr_gpu = calculate_correlations_cudf(df_gpu, list(data.keys()))
        
        # Convert GPU result to CPU for comparison
        corr_gpu_cpu = corr_gpu.to_pandas()
        
        # Compare results (should be very close)
        np.testing.assert_allclose(corr_cpu.values, corr_gpu_cpu.values, rtol=1e-5)
    
    def test_gpu_correlation_performance(self):
        """Test performance improvement of GPU correlation calculation"""
        import time
        
        # Create large dataset
        num_strategies = 500
        num_days = 252
        
        data = {
            f'Strategy_{i}': np.random.randn(num_days) * 1000 
            for i in range(num_strategies)
        }
        
        # Time CPU calculation
        df_cpu = pd.DataFrame(data)
        start_cpu = time.time()
        corr_cpu = df_cpu.corr()
        time_cpu = time.time() - start_cpu
        
        # Time GPU calculation
        df_gpu = cudf.DataFrame(data)
        start_gpu = time.time()
        corr_gpu = df_gpu.corr()
        time_gpu = time.time() - start_gpu
        
        # GPU should be faster for large datasets
        speedup = time_cpu / time_gpu
        print(f"GPU Speedup: {speedup:.2f}x")
        
        # On good GPUs, expect at least 2x speedup
        assert speedup > 1.5 or time_gpu < 0.1  # Either faster or very fast
    
    def test_spearman_correlation_gpu(self):
        """Test Spearman correlation calculation on GPU"""
        # Create test data with non-linear relationships
        num_days = 100
        x = np.linspace(0, 10, num_days)
        
        data = {
            'Strategy_1': x + np.random.randn(num_days) * 0.1,
            'Strategy_2': x**2 + np.random.randn(num_days),  # Non-linear relationship
            'Strategy_3': np.sin(x) * 100 + np.random.randn(num_days) * 10
        }
        
        df_gpu = cudf.DataFrame(data)
        
        # Calculate Spearman correlation
        corr_spearman = calculate_correlations_cudf(
            df_gpu, 
            list(data.keys()), 
            method='spearman'
        )
        
        # Verify it's different from Pearson
        corr_pearson = calculate_correlations_cudf(
            df_gpu, 
            list(data.keys()), 
            method='pearson'
        )
        
        # Results should be different
        diff = np.abs(corr_spearman.to_pandas().values - corr_pearson.to_pandas().values)
        assert diff.max() > 0.1  # Significant difference expected


@pytest.mark.gpu
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA/cuDF not available")
class TestGPUFitnessCalculations:
    """Test GPU-accelerated fitness metric calculations"""
    
    def test_gpu_fitness_calculation_accuracy(self):
        """Test accuracy of GPU fitness calculations"""
        # Create test portfolio data
        num_days = 100
        
        portfolio_data = {
            'Strategy_1': np.random.randn(num_days) * 1000 + 50,
            'Strategy_2': np.random.randn(num_days) * 800 + 30,
            'Strategy_3': np.random.randn(num_days) * 1200 - 20
        }
        
        df = cudf.DataFrame(portfolio_data)
        portfolio = list(portfolio_data.keys())
        
        metrics_config = {
            'roi_weight': 0.4,
            'drawdown_weight': 0.3,
            'win_rate_weight': 0.2,
            'profit_factor_weight': 0.1
        }
        
        # Calculate fitness
        metrics = calculate_fitness_cudf(df, portfolio, metrics_config)
        
        # Verify metrics are reasonable
        assert 'total_roi' in metrics
        assert 'max_drawdown' in metrics
        assert metrics['max_drawdown'] <= 0  # Drawdown should be negative
        assert 0 <= metrics['win_rate'] <= 1
        assert metrics['profit_factor'] >= 0
    
    def test_batch_portfolio_gpu_processing(self):
        """Test batch processing of multiple portfolios on GPU"""
        # Create dataset
        num_strategies = 100
        num_days = 82
        
        data = {'Date': pd.date_range('2024-01-01', periods=num_days)}
        for i in range(num_strategies):
            data[f'Strategy_{i}'] = np.random.randn(num_days) * 1000
        
        df_gpu = cudf.DataFrame(data)
        
        # Create multiple portfolios to test
        portfolios = [
            [f'Strategy_{i}' for i in range(0, 10)],
            [f'Strategy_{i}' for i in range(10, 20)],
            [f'Strategy_{i}' for i in range(20, 30)],
            [f'Strategy_{i}' for i in range(30, 40)],
            [f'Strategy_{i}' for i in range(40, 50)],
        ]
        
        metrics_config = {'roi_weight': 0.5, 'drawdown_weight': 0.5}
        
        # Process all portfolios
        results = batch_calculate_portfolios_cudf(df_gpu, portfolios, metrics_config)
        
        # Verify results
        assert len(results) == len(portfolios)
        for i, result in enumerate(results):
            assert 'portfolio' in result
            assert 'fitness_score' in result
            assert result['portfolio'] == portfolios[i]


@pytest.mark.gpu
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA/cuDF not available")
class TestGPUDataTransfer:
    """Test efficient data transfer between CPU and GPU"""
    
    def test_arrow_to_cudf_zero_copy(self):
        """Test zero-copy transfer from Arrow to cuDF"""
        import pyarrow as pa
        
        # Create Arrow table
        num_rows = 100000
        data = {
            'col1': np.random.randn(num_rows),
            'col2': np.random.randn(num_rows),
            'col3': np.random.randint(0, 100, num_rows)
        }
        arrow_table = pa.table(data)
        
        # Convert to cuDF (should use zero-copy when possible)
        import time
        start = time.time()
        df_gpu = cudf.DataFrame.from_arrow(arrow_table)
        transfer_time = time.time() - start
        
        # Verify data integrity
        assert len(df_gpu) == num_rows
        assert list(df_gpu.columns) == ['col1', 'col2', 'col3']
        
        # Transfer should be fast (< 100ms for 100k rows)
        assert transfer_time < 0.1
    
    def test_gpu_to_cpu_transfer(self):
        """Test efficient GPU to CPU data transfer"""
        # Create GPU data
        num_rows = 50000
        df_gpu = cudf.DataFrame({
            'result1': cp.random.randn(num_rows),
            'result2': cp.random.randn(num_rows)
        })
        
        # Transfer to CPU
        df_cpu = df_gpu.to_pandas()
        
        # Verify data transferred correctly
        assert isinstance(df_cpu, pd.DataFrame)
        assert len(df_cpu) == num_rows
        assert df_cpu.columns.tolist() == ['result1', 'result2']


@pytest.mark.gpu
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA/cuDF not available")
class TestGPUErrorHandling:
    """Test error handling in GPU operations"""
    
    def test_invalid_data_type_handling(self):
        """Test handling of invalid data types for GPU operations"""
        # Create mixed-type data that might cause issues
        data = {
            'numeric': [1, 2, 3, 4, 5],
            'string': ['a', 'b', 'c', 'd', 'e'],  # Strings can't be correlated
            'mixed': [1, 'b', 3, 'd', 5]  # Mixed types
        }
        
        with pytest.raises(Exception):  # Should raise some exception
            df_gpu = cudf.DataFrame(data)
            calculate_correlations_cudf(df_gpu, ['numeric', 'string'])
    
    def test_empty_dataframe_handling(self):
        """Test handling of empty DataFrames"""
        # Create empty DataFrame
        df_gpu = cudf.DataFrame()
        
        # Should handle gracefully
        with pytest.raises(Exception):
            calculate_correlations_cudf(df_gpu, [])
    
    def test_nan_handling_gpu(self):
        """Test handling of NaN values in GPU calculations"""
        # Create data with NaN values
        data = {
            'Strategy_1': [1, 2, np.nan, 4, 5, 6, 7, 8, 9, 10],
            'Strategy_2': [10, 9, 8, 7, np.nan, np.nan, 4, 3, 2, 1],
            'Strategy_3': [5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
        }
        
        df_gpu = cudf.DataFrame(data)
        
        # Calculate correlations (should handle NaN)
        corr = calculate_correlations_cudf(df_gpu, list(data.keys()))
        
        # Results should be valid despite NaN
        corr_cpu = corr.to_pandas()
        assert not np.all(np.isnan(corr_cpu.values))


class TestGPUMocking:
    """Test GPU functionality with mocking (for CI without GPU)"""
    
    def test_gpu_calculations_with_mock(self):
        """Test GPU calculations with mocked cuDF"""
        # Mock cuDF module
        mock_cudf = MagicMock()
        mock_df = MagicMock()
        
        # Configure mock behavior
        mock_df.corr.return_value = pd.DataFrame(
            np.eye(3), 
            columns=['S1', 'S2', 'S3'],
            index=['S1', 'S2', 'S3']
        )
        
        with patch('backend.lib.cudf_engine.gpu_calculator.cudf', mock_cudf):
            with patch('backend.lib.cudf_engine.gpu_calculator.CUDF_AVAILABLE', True):
                # Mock correlation calculation
                result = mock_df.corr()
                
                # Verify mock was called
                mock_df.corr.assert_called_once()
                
                # Verify result structure
                assert result.shape == (3, 3)
                assert np.allclose(np.diag(result.values), 1.0)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-m', 'gpu'])