"""
Unit tests for Parquet Pipeline components
Tests CSV to Parquet conversion with schema detection
"""

import pytest
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
import tempfile
import os

# Add parent directory to path for imports
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from backend.lib.parquet_pipeline.csv_to_parquet import (
    detect_csv_schema, 
    csv_to_parquet,
    validate_parquet_output,
    MANDATORY_COLUMNS,
    ENHANCED_COLUMNS
)


class TestCSVSchemaDetection:
    """Test CSV schema detection functionality"""
    
    def test_detect_legacy_csv_schema(self):
        """Test schema detection for legacy CSV format (Date + strategies only)"""
        # Create sample legacy CSV
        data = {
            'Date': pd.date_range('2024-01-01', periods=10),
            'Strategy_1': np.random.randn(10) * 1000,
            'Strategy_2': np.random.randn(10) * 1000,
            'Strategy_3': np.random.randn(10) * 1000
        }
        df = pd.DataFrame(data)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            schema, strategy_cols = detect_csv_schema(f.name)
            
        # Verify schema
        assert 'Date' in schema
        assert schema['Date'] == 'date'
        assert len(strategy_cols) == 3
        assert all(col.startswith('Strategy_') for col in strategy_cols)
        
        # Cleanup
        os.unlink(f.name)
    
    def test_detect_enhanced_csv_schema(self):
        """Test schema detection for enhanced CSV format with all new columns"""
        # Create sample enhanced CSV
        data = {
            'Date': pd.date_range('2024-01-01', periods=10),
            'start_time': pd.date_range('2024-01-01 09:00:00', periods=10, freq='1D'),
            'end_time': pd.date_range('2024-01-01 16:00:00', periods=10, freq='1D'),
            'market_regime': ['Bull', 'Bear', 'Neutral'] * 3 + ['Bull'],
            'Regime_Confidence_%': np.random.uniform(60, 95, 10),
            'Market_regime_transition_threshold': np.random.uniform(0.7, 0.9, 10),
            'capital': np.full(10, 100000.0),
            'zone': ['Zone_A', 'Zone_B'] * 5,
            'Strategy_1': np.random.randn(10) * 1000,
            'Strategy_2': np.random.randn(10) * 1000
        }
        df = pd.DataFrame(data)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            schema, strategy_cols = detect_csv_schema(f.name)
            
        # Verify schema
        assert schema['Date'] == 'date'
        assert schema['start_time'] == 'datetime64[ns]'
        assert schema['end_time'] == 'datetime64[ns]'
        assert schema['market_regime'] == 'string'
        assert schema['Regime_Confidence_%'] == 'float64'
        assert schema['zone'] == 'string'
        assert len(strategy_cols) == 2
        
        # Cleanup
        os.unlink(f.name)
    
    def test_skip_symbol_columns(self):
        """Test that symbol columns are correctly skipped"""
        data = {
            'Date': pd.date_range('2024-01-01', periods=5),
            'Symbol': ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META'],
            'Strategy_1': np.random.randn(5) * 1000
        }
        df = pd.DataFrame(data)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            schema, strategy_cols = detect_csv_schema(f.name)
            
        # Verify symbol column is excluded
        assert 'Symbol' not in schema
        assert len(strategy_cols) == 1
        assert 'Strategy_1' in strategy_cols
        
        # Cleanup
        os.unlink(f.name)
    
    def test_handle_non_numeric_columns(self):
        """Test handling of non-numeric columns that aren't enhanced columns"""
        data = {
            'Date': pd.date_range('2024-01-01', periods=5),
            'Strategy_1': np.random.randn(5) * 1000,
            'InvalidColumn': ['A', 'B', 'C', 'D', 'E'],  # Non-numeric, not enhanced
            'Strategy_2': np.random.randn(5) * 1000
        }
        df = pd.DataFrame(data)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            schema, strategy_cols = detect_csv_schema(f.name)
            
        # Verify invalid column is excluded
        assert 'InvalidColumn' not in schema
        assert len(strategy_cols) == 2
        
        # Cleanup
        os.unlink(f.name)


class TestCSVToParquetConversion:
    """Test CSV to Parquet conversion functionality"""
    
    def test_convert_legacy_csv_to_parquet(self):
        """Test conversion of legacy CSV format to Parquet"""
        # Create sample data
        num_strategies = 100
        num_days = 30
        
        data = {'Date': pd.date_range('2024-01-01', periods=num_days)}
        for i in range(num_strategies):
            data[f'Strategy_{i+1}'] = np.random.randn(num_days) * 1000
            
        df = pd.DataFrame(data)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, 'test_legacy.csv')
            parquet_path = os.path.join(tmpdir, 'test_legacy.parquet')
            
            # Save CSV
            df.to_csv(csv_path, index=False)
            
            # Convert to Parquet
            success = csv_to_parquet(csv_path, parquet_path)
            assert success is True
            
            # Validate Parquet file
            assert os.path.exists(parquet_path)
            
            # Read back and compare
            df_parquet = pd.read_parquet(parquet_path)
            assert len(df_parquet) == num_days
            assert len(df_parquet.columns) == num_strategies + 1
            
            # Check data integrity
            pd.testing.assert_frame_equal(
                df.set_index('Date'),
                df_parquet.set_index('Date'),
                check_dtype=False,
                rtol=1e-5
            )
    
    def test_convert_enhanced_csv_to_parquet(self):
        """Test conversion of enhanced CSV format to Parquet"""
        # Create enhanced sample data
        num_days = 20
        
        data = {
            'Date': pd.date_range('2024-01-01', periods=num_days),
            'start_time': pd.date_range('2024-01-01 09:00:00', periods=num_days, freq='1D'),
            'end_time': pd.date_range('2024-01-01 16:00:00', periods=num_days, freq='1D'),
            'market_regime': np.random.choice(['Bull', 'Bear', 'Neutral'], num_days),
            'Regime_Confidence_%': np.random.uniform(60, 95, num_days),
            'Market_regime_transition_threshold': np.random.uniform(0.7, 0.9, num_days),
            'capital': np.full(num_days, 100000.0),
            'zone': np.random.choice(['Zone_A', 'Zone_B', 'Zone_C'], num_days),
            'Strategy_1': np.random.randn(num_days) * 1000,
            'Strategy_2': np.random.randn(num_days) * 1000,
            'Strategy_3': np.random.randn(num_days) * 1000
        }
        df = pd.DataFrame(data)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, 'test_enhanced.csv')
            parquet_path = os.path.join(tmpdir, 'test_enhanced.parquet')
            
            # Save CSV
            df.to_csv(csv_path, index=False)
            
            # Convert to Parquet
            success = csv_to_parquet(csv_path, parquet_path)
            assert success is True
            
            # Validate Parquet file
            table = pq.read_table(parquet_path)
            
            # Check schema types
            schema = table.schema
            assert schema.field('Date').type == pa.date32()
            assert schema.field('market_regime').type == pa.string()
            assert schema.field('Regime_Confidence_%').type == pa.float64()
            
            # Check row groups
            parquet_file = pq.ParquetFile(parquet_path)
            assert parquet_file.num_row_groups >= 1
    
    def test_compression_options(self):
        """Test different compression algorithms"""
        data = {
            'Date': pd.date_range('2024-01-01', periods=100),
            'Strategy_1': np.random.randn(100) * 1000
        }
        df = pd.DataFrame(data)
        
        compressions = ['snappy', 'gzip', 'lz4', 'zstd']
        
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, 'test.csv')
            df.to_csv(csv_path, index=False)
            
            for compression in compressions:
                parquet_path = os.path.join(tmpdir, f'test_{compression}.parquet')
                success = csv_to_parquet(csv_path, parquet_path, compression=compression)
                assert success is True
                
                # Verify compression
                parquet_file = pq.ParquetFile(parquet_path)
                metadata = parquet_file.metadata
                assert metadata.row_group(0).column(0).compression == compression.upper()
    
    def test_row_group_size_configuration(self):
        """Test custom row group size configuration"""
        # Create large dataset
        num_rows = 150000
        data = {
            'Date': pd.date_range('2024-01-01', periods=num_rows, freq='1min'),
            'Strategy_1': np.random.randn(num_rows) * 1000
        }
        df = pd.DataFrame(data)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, 'test_large.csv')
            parquet_path = os.path.join(tmpdir, 'test_large.parquet')
            
            df.to_csv(csv_path, index=False)
            
            # Convert with custom row group size
            row_group_size = 25000
            success = csv_to_parquet(csv_path, parquet_path, row_group_size=row_group_size)
            assert success is True
            
            # Verify row groups
            parquet_file = pq.ParquetFile(parquet_path)
            expected_groups = (num_rows + row_group_size - 1) // row_group_size
            assert parquet_file.num_row_groups == expected_groups
    
    def test_parquet_validation(self):
        """Test Parquet output validation"""
        data = {
            'Date': pd.date_range('2024-01-01', periods=10),
            'Strategy_1': np.random.randn(10) * 1000,
            'Strategy_2': np.random.randn(10) * 1000
        }
        df = pd.DataFrame(data)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, 'test.csv')
            parquet_path = os.path.join(tmpdir, 'test.parquet')
            
            df.to_csv(csv_path, index=False)
            success = csv_to_parquet(csv_path, parquet_path)
            assert success is True
            
            # Validate output
            is_valid, message = validate_parquet_output(parquet_path)
            assert is_valid is True
            assert message == "Parquet file is valid"


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_empty_csv_handling(self):
        """Test handling of empty CSV files"""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, 'empty.csv')
            parquet_path = os.path.join(tmpdir, 'empty.parquet')
            
            # Create empty CSV with headers only
            with open(csv_path, 'w') as f:
                f.write('Date,Strategy_1,Strategy_2\\n')
            
            success = csv_to_parquet(csv_path, parquet_path)
            assert success is False
    
    def test_missing_mandatory_columns(self):
        """Test handling of CSV without mandatory Date column"""
        data = {
            'Strategy_1': np.random.randn(10) * 1000,
            'Strategy_2': np.random.randn(10) * 1000
        }
        df = pd.DataFrame(data)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, 'no_date.csv')
            parquet_path = os.path.join(tmpdir, 'no_date.parquet')
            
            df.to_csv(csv_path, index=False)
            success = csv_to_parquet(csv_path, parquet_path)
            assert success is False
    
    def test_all_nan_strategy_columns(self):
        """Test handling of strategy columns with all NaN values"""
        data = {
            'Date': pd.date_range('2024-01-01', periods=10),
            'Strategy_1': np.random.randn(10) * 1000,
            'Strategy_2': np.full(10, np.nan)  # All NaN
        }
        df = pd.DataFrame(data)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, 'nan_strategy.csv')
            parquet_path = os.path.join(tmpdir, 'nan_strategy.parquet')
            
            df.to_csv(csv_path, index=False)
            schema, strategy_cols = detect_csv_schema(csv_path)
            
            # Should skip the all-NaN column
            assert 'Strategy_2' not in strategy_cols
            assert len(strategy_cols) == 1
    
    def test_memory_efficient_chunking(self):
        """Test memory-efficient chunked processing for large files"""
        # Create a moderately large dataset
        num_rows = 100000
        num_strategies = 50
        
        data = {'Date': pd.date_range('2024-01-01', periods=num_rows, freq='1min')}
        for i in range(num_strategies):
            data[f'Strategy_{i+1}'] = np.random.randn(num_rows) * 1000
            
        df = pd.DataFrame(data)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, 'large.csv')
            parquet_path = os.path.join(tmpdir, 'large.parquet')
            
            df.to_csv(csv_path, index=False)
            
            # Convert with chunking
            success = csv_to_parquet(csv_path, parquet_path, chunk_size=10000)
            assert success is True
            
            # Verify all data was converted
            df_parquet = pd.read_parquet(parquet_path)
            assert len(df_parquet) == num_rows
            assert len(df_parquet.columns) == num_strategies + 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])