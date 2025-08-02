"""
Integration tests for CSV to Parquet conversion pipeline
Part of Epic 1 - Parquet/Arrow/cuDF Implementation
"""

import pytest
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
from pathlib import Path
import tempfile
import numpy as np
from datetime import datetime, timedelta


class TestCSVToParquetIntegration:
    """Test suite for CSV to Parquet conversion with schema detection"""
    
    @pytest.fixture
    def sample_legacy_csv(self, tmp_path):
        """Create a sample legacy CSV file for testing"""
        dates = pd.date_range('2023-01-01', periods=100)
        strategies = [f'Strategy_{i}' for i in range(1, 11)]
        
        data = []
        for date in dates:
            for strategy in strategies:
                data.append({
                    'Date': date.strftime('%Y-%m-%d'),
                    strategy: np.random.uniform(-1000, 5000)
                })
        
        df = pd.DataFrame(data)
        csv_path = tmp_path / "legacy_test.csv"
        df.to_csv(csv_path, index=False)
        return csv_path
    
    @pytest.fixture
    def sample_enhanced_csv(self, tmp_path):
        """Create a sample enhanced CSV file with new columns"""
        dates = pd.date_range('2023-01-01', periods=100)
        strategies = [f'Strategy_{i}' for i in range(1, 11)]
        
        data = []
        for date in dates:
            base_row = {
                'Date': date.strftime('%Y-%m-%d'),
                'start_time': '09:30:00',
                'end_time': '16:00:00',
                'market_regime': np.random.choice(['Bullish_Vol_Expansion', 'Bearish_Panic', 'Neutral']),
                'Regime_Confidence_%': np.random.uniform(60, 95),
                'Market_regime_transition_threshold': np.random.uniform(0.3, 0.7),
                'capital': 100000,
                'zone': np.random.choice(['US', 'EU', 'ASIA'])
            }
            for strategy in strategies:
                base_row[strategy] = np.random.uniform(-1000, 5000)
            data.append(base_row)
        
        df = pd.DataFrame(data)
        csv_path = tmp_path / "enhanced_test.csv"
        df.to_csv(csv_path, index=False)
        return csv_path
    
    def test_legacy_csv_to_parquet_conversion(self, sample_legacy_csv, tmp_path):
        """Test conversion of legacy CSV format to Parquet"""
        from backend.lib.data_pipeline.csv_to_parquet import CSVToParquetConverter
        
        converter = CSVToParquetConverter()
        parquet_path = tmp_path / "output.parquet"
        
        # Convert CSV to Parquet
        metadata = converter.convert(sample_legacy_csv, parquet_path)
        
        # Verify Parquet file exists
        assert parquet_path.exists()
        
        # Verify metadata
        assert metadata['format'] == 'legacy'
        assert metadata['num_strategies'] == 10
        assert metadata['num_rows'] == 100
        
        # Read back and verify data integrity
        table = pq.read_table(parquet_path)
        df_original = pd.read_csv(sample_legacy_csv)
        df_parquet = table.to_pandas()
        
        pd.testing.assert_frame_equal(df_original, df_parquet)
    
    def test_enhanced_csv_to_parquet_conversion(self, sample_enhanced_csv, tmp_path):
        """Test conversion of enhanced CSV format to Parquet"""
        from backend.lib.data_pipeline.csv_to_parquet import CSVToParquetConverter
        
        converter = CSVToParquetConverter()
        parquet_path = tmp_path / "output.parquet"
        
        # Convert CSV to Parquet
        metadata = converter.convert(sample_enhanced_csv, parquet_path)
        
        # Verify metadata
        assert metadata['format'] == 'enhanced'
        assert metadata['has_regime_data'] == True
        assert metadata['has_zone_data'] == True
        
        # Verify Parquet schema includes new columns
        schema = pq.read_schema(parquet_path)
        assert 'market_regime' in schema.names
        assert 'zone' in schema.names
        assert 'capital' in schema.names
    
    def test_schema_detection(self, sample_legacy_csv, sample_enhanced_csv):
        """Test automatic schema detection for different CSV formats"""
        from backend.lib.data_pipeline.schema_detector import SchemaDetector
        
        detector = SchemaDetector()
        
        # Test legacy format detection
        legacy_schema = detector.detect(sample_legacy_csv)
        assert legacy_schema['format'] == 'legacy'
        assert 'Date' in legacy_schema['columns']
        assert 'market_regime' not in legacy_schema['columns']
        
        # Test enhanced format detection
        enhanced_schema = detector.detect(sample_enhanced_csv)
        assert enhanced_schema['format'] == 'enhanced'
        assert 'market_regime' in enhanced_schema['columns']
        assert 'zone' in enhanced_schema['columns']
    
    def test_compression_efficiency(self, sample_legacy_csv, tmp_path):
        """Test Parquet compression efficiency"""
        from backend.lib.data_pipeline.csv_to_parquet import CSVToParquetConverter
        
        converter = CSVToParquetConverter()
        parquet_path = tmp_path / "compressed.parquet"
        
        # Convert with different compression algorithms
        for compression in ['snappy', 'gzip', 'brotli']:
            output_path = tmp_path / f"output_{compression}.parquet"
            converter.convert(sample_legacy_csv, output_path, compression=compression)
            
            # Check file sizes
            csv_size = sample_legacy_csv.stat().st_size
            parquet_size = output_path.stat().st_size
            compression_ratio = csv_size / parquet_size
            
            assert compression_ratio > 1.5  # Expect at least 1.5x compression
    
    def test_partitioning_by_date(self, sample_enhanced_csv, tmp_path):
        """Test Parquet partitioning by date for query optimization"""
        from backend.lib.data_pipeline.csv_to_parquet import CSVToParquetConverter
        
        converter = CSVToParquetConverter()
        parquet_dir = tmp_path / "partitioned"
        
        # Convert with date partitioning
        converter.convert(
            sample_enhanced_csv, 
            parquet_dir,
            partition_cols=['Date']
        )
        
        # Verify partition structure
        date_dirs = list(parquet_dir.glob('Date=*'))
        assert len(date_dirs) > 0
        
        # Verify we can read specific partitions
        table = pq.read_table(parquet_dir, filters=[('Date', '=', '2023-01-01')])
        assert len(table) > 0
    
    def test_large_file_handling(self, tmp_path):
        """Test handling of large CSV files (>1GB)"""
        from backend.lib.data_pipeline.csv_to_parquet import CSVToParquetConverter
        
        # Create a large CSV file
        large_csv = tmp_path / "large_test.csv"
        
        # Generate data in chunks to avoid memory issues
        chunk_size = 100000
        num_chunks = 20  # ~2GB file
        
        with open(large_csv, 'w') as f:
            # Write header
            f.write('Date,' + ','.join([f'Strategy_{i}' for i in range(1, 101)]) + '\n')
            
            # Write data chunks
            for chunk in range(num_chunks):
                dates = pd.date_range('2023-01-01', periods=chunk_size)
                for date in dates:
                    row = [date.strftime('%Y-%m-%d')]
                    row.extend([str(np.random.uniform(-1000, 5000)) for _ in range(100)])
                    f.write(','.join(row) + '\n')
        
        # Convert to Parquet
        converter = CSVToParquetConverter()
        parquet_path = tmp_path / "large_output.parquet"
        
        metadata = converter.convert(large_csv, parquet_path, chunk_size=50000)
        
        # Verify conversion completed
        assert parquet_path.exists()
        assert metadata['num_rows'] == chunk_size * num_chunks
    
    def test_error_handling_invalid_csv(self, tmp_path):
        """Test error handling for invalid CSV files"""
        from backend.lib.data_pipeline.csv_to_parquet import CSVToParquetConverter
        
        # Create invalid CSV
        invalid_csv = tmp_path / "invalid.csv"
        invalid_csv.write_text("Date,Strategy1\n2023-01-01,not_a_number\n")
        
        converter = CSVToParquetConverter()
        parquet_path = tmp_path / "output.parquet"
        
        with pytest.raises(ValueError, match="Invalid numeric data"):
            converter.convert(invalid_csv, parquet_path)
    
    @pytest.mark.benchmark
    def test_conversion_performance(self, sample_legacy_csv, tmp_path, benchmark):
        """Benchmark CSV to Parquet conversion performance"""
        from backend.lib.data_pipeline.csv_to_parquet import CSVToParquetConverter
        
        converter = CSVToParquetConverter()
        parquet_path = tmp_path / "benchmark.parquet"
        
        # Benchmark the conversion
        result = benchmark(converter.convert, sample_legacy_csv, parquet_path)
        
        # Performance assertions
        assert benchmark.stats['mean'] < 1.0  # Should complete in < 1 second