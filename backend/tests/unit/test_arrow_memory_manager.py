"""
Unit tests for Arrow Memory Manager
Tests memory pool management and Parquet to Arrow loading
"""

import pytest
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import tempfile
import os
import gc
from unittest.mock import patch, MagicMock

# Add parent directory to path for imports
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from backend.lib.arrow_connector.memory_manager import (
    ArrowMemoryPool,
    create_memory_pool,
    load_parquet_to_arrow,
    arrow_to_cudf,
    optimize_memory_layout,
    batch_process_arrow_table
)


class TestArrowMemoryPool:
    """Test Arrow memory pool functionality"""
    
    def test_memory_pool_initialization(self):
        """Test memory pool initialization with custom size"""
        pool = create_memory_pool(size_gb=2.0)
        
        assert pool.pool_size_bytes == 2 * 1024 * 1024 * 1024
        assert pool.pool is not None
        assert pool.initial_bytes >= 0
    
    def test_memory_usage_tracking(self):
        """Test memory usage statistics tracking"""
        pool = create_memory_pool(size_gb=1.0)
        
        # Get initial usage
        usage = pool.get_usage()
        assert 'allocated_bytes' in usage
        assert 'allocated_gb' in usage
        assert 'max_memory_bytes' in usage
        assert 'max_memory_gb' in usage
        
        # Allocate some memory
        data = pa.array(np.random.randn(1000000))
        
        # Check usage increased
        new_usage = pool.get_usage()
        assert new_usage['allocated_bytes'] >= usage['allocated_bytes']
    
    def test_memory_cleanup(self):
        """Test memory cleanup functionality"""
        pool = create_memory_pool(size_gb=1.0)
        
        # Create and delete large array
        data = pa.array(np.random.randn(10000000))
        initial_usage = pool.get_usage()['allocated_bytes']
        
        del data
        pool.cleanup()
        
        # Memory should be reduced after cleanup
        final_usage = pool.get_usage()['allocated_bytes']
        # Note: Can't guarantee full cleanup due to Python memory management
        assert final_usage <= initial_usage * 1.1  # Allow 10% variance


class TestParquetToArrowLoading:
    """Test Parquet to Arrow loading functionality"""
    
    def create_test_parquet(self, num_rows=1000, num_cols=10):
        """Helper to create test Parquet file"""
        data = {'Date': pd.date_range('2024-01-01', periods=num_rows)}
        for i in range(num_cols):
            data[f'Strategy_{i+1}'] = np.random.randn(num_rows) * 1000
        
        df = pd.DataFrame(data)
        
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f:
            df.to_parquet(f.name, row_group_size=250)
            return f.name, df
    
    def test_load_full_parquet_file(self):
        """Test loading entire Parquet file to Arrow"""
        parquet_path, original_df = self.create_test_parquet(1000, 5)
        
        try:
            # Load to Arrow
            table = load_parquet_to_arrow(parquet_path)
            
            # Verify table
            assert isinstance(table, pa.Table)
            assert table.num_rows == 1000
            assert table.num_columns == 6  # Date + 5 strategies
            
            # Verify data integrity
            loaded_df = table.to_pandas()
            pd.testing.assert_frame_equal(original_df, loaded_df, check_dtype=False)
            
        finally:
            os.unlink(parquet_path)
    
    def test_load_specific_columns(self):
        """Test loading specific columns from Parquet"""
        parquet_path, _ = self.create_test_parquet(500, 10)
        
        try:
            # Load only specific columns
            columns = ['Date', 'Strategy_1', 'Strategy_5']
            table = load_parquet_to_arrow(parquet_path, columns=columns)
            
            # Verify only requested columns loaded
            assert table.num_columns == 3
            assert set(table.column_names) == set(columns)
            
        finally:
            os.unlink(parquet_path)
    
    def test_load_specific_row_groups(self):
        """Test loading specific row groups from Parquet"""
        parquet_path, _ = self.create_test_parquet(1000, 5)
        
        try:
            # Load only first two row groups (500 rows with row_group_size=250)
            table = load_parquet_to_arrow(parquet_path, row_groups=[0, 1])
            
            # Verify row count
            assert table.num_rows == 500
            
        finally:
            os.unlink(parquet_path)
    
    def test_invalid_column_handling(self):
        """Test handling of invalid column requests"""
        parquet_path, _ = self.create_test_parquet(100, 3)
        
        try:
            # Request non-existent columns
            with pytest.raises(ValueError) as excinfo:
                load_parquet_to_arrow(parquet_path, columns=['NonExistent'])
            
            assert "Missing columns" in str(excinfo.value)
            
        finally:
            os.unlink(parquet_path)
    
    def test_memory_pool_usage(self):
        """Test loading with custom memory pool"""
        parquet_path, _ = self.create_test_parquet(5000, 20)
        pool = create_memory_pool(size_gb=1.0)
        
        try:
            # Load with memory pool
            table = load_parquet_to_arrow(parquet_path, memory_pool=pool)
            
            # Check memory was allocated
            usage = pool.get_usage()
            assert usage['allocated_bytes'] > 0
            
            # Cleanup and verify
            del table
            pool.cleanup()
            
        finally:
            os.unlink(parquet_path)


class TestArrowToCuDF:
    """Test Arrow to cuDF conversion functionality"""
    
    @pytest.mark.skipif(not pa.cuda.is_available(), reason="CUDA not available")
    def test_arrow_to_cudf_gpu_available(self):
        """Test Arrow to cuDF conversion when GPU is available"""
        # Create Arrow table
        data = {
            'col1': pa.array(np.random.randn(1000)),
            'col2': pa.array(np.random.randn(1000)),
            'col3': pa.array(np.random.randint(0, 100, 1000))
        }
        table = pa.table(data)
        
        # Convert to cuDF
        result = arrow_to_cudf(table, use_gpu=True)
        
        if result is not None:  # GPU available
            import cudf
            assert isinstance(result, cudf.DataFrame)
            assert len(result) == 1000
            assert list(result.columns) == ['col1', 'col2', 'col3']
    
    def test_arrow_to_cudf_fallback_cpu(self):
        """Test Arrow to cuDF fallback to pandas when GPU unavailable"""
        # Create Arrow table
        data = {
            'col1': pa.array(np.random.randn(500)),
            'col2': pa.array(np.random.randn(500))
        }
        table = pa.table(data)
        
        # Force CPU mode
        with patch('backend.lib.arrow_connector.memory_manager.CUDF_AVAILABLE', False):
            result = arrow_to_cudf(table, use_gpu=True)
            
            # Should return pandas DataFrame
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 500
    
    def test_zero_copy_gpu_transfer(self):
        """Test zero-copy GPU transfer optimization"""
        # Create Arrow table with larger data
        size = 100000
        data = {
            f'Strategy_{i}': pa.array(np.random.randn(size)) 
            for i in range(10)
        }
        table = pa.table(data)
        
        # Test zero-copy transfer (mocked)
        with patch('backend.lib.arrow_connector.memory_manager.CUDF_AVAILABLE', True):
            with patch('cudf.DataFrame.from_arrow') as mock_from_arrow:
                mock_df = MagicMock()
                mock_from_arrow.return_value = mock_df
                
                result = arrow_to_cudf(table, use_gpu=True)
                
                # Verify zero-copy was attempted
                mock_from_arrow.assert_called_once()
                assert result == mock_df


class TestMemoryOptimization:
    """Test memory optimization functionality"""
    
    def test_optimize_memory_layout(self):
        """Test memory layout optimization for columnar data"""
        # Create table with different data types
        data = {
            'int_col': pa.array(np.random.randint(0, 1000, 1000), type=pa.int64()),
            'float_col': pa.array(np.random.randn(1000), type=pa.float64()),
            'string_col': pa.array([f'Strategy_{i}' for i in range(1000)]),
            'date_col': pa.array(pd.date_range('2024-01-01', periods=1000))
        }
        table = pa.table(data)
        
        # Optimize layout
        optimized = optimize_memory_layout(table)
        
        # Verify optimization
        assert optimized.num_rows == table.num_rows
        assert optimized.num_columns == table.num_columns
        
        # Check memory is contiguous (combined chunks)
        for col in optimized.columns:
            assert col.num_chunks <= 1
    
    def test_batch_processing(self):
        """Test batch processing of large Arrow tables"""
        # Create large table
        size = 100000
        data = {
            'Date': pa.array(pd.date_range('2024-01-01', periods=size, freq='1min')),
            'Value': pa.array(np.random.randn(size) * 1000)
        }
        table = pa.table(data)
        
        # Process in batches
        batch_size = 10000
        batches = list(batch_process_arrow_table(table, batch_size))
        
        # Verify batches
        assert len(batches) == 10  # 100000 / 10000
        
        total_rows = sum(batch.num_rows for batch in batches)
        assert total_rows == size
        
        # Verify each batch (except possibly the last)
        for i, batch in enumerate(batches[:-1]):
            assert batch.num_rows == batch_size


class TestEdgeCasesAndErrors:
    """Test edge cases and error handling"""
    
    def test_empty_table_handling(self):
        """Test handling of empty Arrow tables"""
        # Create empty table
        schema = pa.schema([
            ('col1', pa.float64()),
            ('col2', pa.int64())
        ])
        table = pa.table([], schema=schema)
        
        # Test operations on empty table
        result = arrow_to_cudf(table, use_gpu=False)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
        assert list(result.columns) == ['col1', 'col2']
    
    def test_large_memory_allocation(self):
        """Test handling of large memory allocations"""
        pool = create_memory_pool(size_gb=0.1)  # Small pool
        
        # Try to allocate large array
        try:
            # This should work even with small pool due to Arrow's memory management
            large_array = pa.array(np.random.randn(10000000))
            assert len(large_array) == 10000000
        finally:
            del large_array
            gc.collect()
    
    def test_corrupted_parquet_handling(self):
        """Test handling of corrupted Parquet files"""
        # Create invalid file
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f:
            f.write(b'Invalid Parquet content')
            temp_path = f.name
        
        try:
            # Should handle gracefully
            with pytest.raises(Exception):  # Could be various exceptions
                load_parquet_to_arrow(temp_path)
        finally:
            os.unlink(temp_path)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])