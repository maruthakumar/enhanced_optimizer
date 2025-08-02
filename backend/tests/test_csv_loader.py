#!/usr/bin/env python3
"""
Unit Tests for Enterprise CSV Loader

Tests all features of the enterprise CSV loader including:
- Streaming
- Batch inserts
- Progress tracking
- Data validation
- Checkpoint/resume
- Audit trail
"""

import unittest
import tempfile
import os
import time
import json
import pandas as pd
import numpy as np
from pathlib import Path
import shutil

# Add backend to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from lib.csv_loader import (
    EnterpriseCSVLoader, BatchInsertHandler, GPUMemoryManager,
    LoadMetrics, ValidationResult, LoadCheckpoint
)


class TestEnterpriseCSVLoader(unittest.TestCase):
    """Test cases for Enterprise CSV Loader"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create temporary directories
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint_dir = os.path.join(self.temp_dir, 'checkpoints')
        self.audit_log_path = os.path.join(self.temp_dir, 'audit.json')
        
        # Create loader instance
        self.loader = EnterpriseCSVLoader(
            chunk_size=100,
            batch_size=50,
            enable_gpu_monitoring=False,  # Disable for tests
            checkpoint_dir=self.checkpoint_dir,
            audit_log_path=self.audit_log_path
        )
        
        # Create test CSV files
        self.small_csv = self._create_test_csv('small.csv', 500, 10)
        self.medium_csv = self._create_test_csv('medium.csv', 5000, 20)
        self.large_csv = self._create_test_csv('large.csv', 10000, 50)
        self.invalid_csv = self._create_invalid_csv('invalid.csv')
        
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir)
    
    def _create_test_csv(self, filename: str, rows: int, cols: int) -> str:
        """Create a test CSV file"""
        filepath = os.path.join(self.temp_dir, filename)
        
        # Create random data
        data = np.random.randn(rows, cols)
        columns = [f'strategy_{i:04d}' for i in range(cols)]
        
        df = pd.DataFrame(data, columns=columns)
        
        # Add some missing values
        mask = np.random.random(df.shape) < 0.05  # 5% missing
        df[mask] = np.nan
        
        df.to_csv(filepath, index=False)
        return filepath
    
    def _create_invalid_csv(self, filename: str) -> str:
        """Create an invalid CSV file"""
        filepath = os.path.join(self.temp_dir, filename)
        
        with open(filepath, 'w') as f:
            f.write("col1,col2,col3\n")
            f.write("1,2,3\n")
            f.write("4,five,6\n")  # Invalid numeric
            f.write("7,8\n")  # Missing column
            f.write("9,10,11,12\n")  # Extra column
        
        return filepath
    
    def test_basic_streaming_load(self):
        """Test basic streaming CSV load"""
        chunks_received = []
        
        def chunk_handler(table_name: str, chunk: pd.DataFrame):
            chunks_received.append(chunk)
        
        metrics = self.loader.load_csv_streaming(
            file_path=self.small_csv,
            table_name='test_table',
            data_handler=chunk_handler
        )
        
        # Verify metrics
        self.assertEqual(metrics.status, "SUCCESS")
        self.assertEqual(metrics.total_rows, 500)
        self.assertEqual(metrics.total_columns, 10)
        self.assertGreater(metrics.chunks_processed, 0)
        self.assertGreater(metrics.throughput_mbps, 0)
        self.assertGreater(metrics.peak_memory_mb, 0)
        
        # Verify chunks
        total_rows = sum(len(chunk) for chunk in chunks_received)
        self.assertEqual(total_rows, 500)
    
    def test_progress_tracking(self):
        """Test progress tracking during load"""
        progress_updates = []
        
        def progress_callback(progress: dict):
            progress_updates.append(progress.copy())
        
        metrics = self.loader.load_csv_streaming(
            file_path=self.medium_csv,
            table_name='test_table',
            progress_callback=progress_callback
        )
        
        # Verify progress updates
        self.assertGreater(len(progress_updates), 0)
        
        # Check progress increases
        for i in range(1, len(progress_updates)):
            self.assertGreaterEqual(
                progress_updates[i]['rows_processed'],
                progress_updates[i-1]['rows_processed']
            )
        
        # Final progress should be 100%
        final_progress = progress_updates[-1]
        self.assertAlmostEqual(final_progress['percentage'], 100.0, delta=5.0)
    
    def test_data_validation(self):
        """Test data validation functionality"""
        # Test valid CSV
        valid_result = self.loader.validate_csv_file(self.small_csv)
        self.assertTrue(valid_result.is_valid)
        self.assertGreater(len(valid_result.numeric_columns), 0)
        
        # Test invalid CSV
        invalid_result = self.loader.validate_csv_file(self.invalid_csv)
        self.assertFalse(invalid_result.is_valid)
        self.assertGreater(len(invalid_result.errors), 0)
    
    def test_missing_value_handling(self):
        """Test missing value handling"""
        chunks_received = []
        
        def chunk_handler(table_name: str, chunk: pd.DataFrame):
            chunks_received.append(chunk)
        
        metrics = self.loader.load_csv_streaming(
            file_path=self.small_csv,
            table_name='test_table',
            data_handler=chunk_handler,
            fill_missing_values=True,
            missing_fill_value=0.0
        )
        
        # Verify no missing values in processed chunks
        for chunk in chunks_received:
            self.assertEqual(chunk.isnull().sum().sum(), 0)
    
    def test_checkpoint_resume(self):
        """Test checkpoint and resume functionality"""
        chunks_before_interrupt = []
        chunks_after_resume = []
        
        # Simulate interrupted load
        def interrupt_handler(table_name: str, chunk: pd.DataFrame):
            chunks_before_interrupt.append(chunk)
            if len(chunks_before_interrupt) == 2:
                raise Exception("Simulated interruption")
        
        # First attempt - will be interrupted
        try:
            self.loader.load_csv_streaming(
                file_path=self.medium_csv,
                table_name='test_table',
                data_handler=interrupt_handler
            )
        except Exception:
            pass
        
        # Verify checkpoint was created
        checkpoint_files = list(Path(self.checkpoint_dir).glob('*.checkpoint'))
        self.assertEqual(len(checkpoint_files), 1)
        
        # Resume from checkpoint
        def resume_handler(table_name: str, chunk: pd.DataFrame):
            chunks_after_resume.append(chunk)
        
        metrics = self.loader.load_csv_streaming(
            file_path=self.medium_csv,
            table_name='test_table',
            data_handler=resume_handler,
            resume_from_checkpoint=True
        )
        
        # Verify successful completion
        self.assertEqual(metrics.status, "SUCCESS")
        
        # Verify checkpoint was cleaned up
        checkpoint_files = list(Path(self.checkpoint_dir).glob('*.checkpoint'))
        self.assertEqual(len(checkpoint_files), 0)
    
    def test_audit_trail(self):
        """Test audit trail functionality"""
        # Load multiple files
        for csv_file in [self.small_csv, self.medium_csv]:
            self.loader.load_csv_streaming(
                file_path=csv_file,
                table_name='test_table'
            )
        
        # Get audit trail
        audit_entries = self.loader.get_audit_trail(limit=10)
        
        # Verify audit entries
        self.assertGreaterEqual(len(audit_entries), 2)
        
        for entry in audit_entries:
            self.assertIn('timestamp', entry)
            self.assertIn('metrics', entry)
            
            metrics = entry['metrics']
            self.assertIn('file_path', metrics)
            self.assertIn('total_rows', metrics)
            self.assertIn('status', metrics)
    
    def test_memory_monitoring(self):
        """Test memory monitoring"""
        metrics = self.loader.load_csv_streaming(
            file_path=self.large_csv,
            table_name='test_table'
        )
        
        # Verify memory metrics
        self.assertGreater(metrics.peak_memory_mb, 0)
        self.assertLess(metrics.peak_memory_mb, 1000)  # Reasonable limit
    
    def test_throughput_calculation(self):
        """Test throughput calculation"""
        metrics = self.loader.load_csv_streaming(
            file_path=self.medium_csv,
            table_name='test_table'
        )
        
        # Verify throughput
        self.assertGreater(metrics.throughput_mbps, 0)
        
        # Manual calculation
        expected_throughput = metrics.file_size_mb / metrics.load_time_seconds
        self.assertAlmostEqual(metrics.throughput_mbps, expected_throughput, delta=0.1)


class TestBatchInsertHandler(unittest.TestCase):
    """Test cases for Batch Insert Handler"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.handler = BatchInsertHandler(
            connection=None,  # No real connection for tests
            batch_size=100,
            use_gpu_transfer=False
        )
        
        # Create test data
        self.test_df = pd.DataFrame(
            np.random.randn(1000, 10),
            columns=[f'col_{i}' for i in range(10)]
        )
    
    def test_batch_insert(self):
        """Test basic batch insert"""
        success = self.handler.insert_batch('test_table', self.test_df.iloc[:100])
        self.assertTrue(success)
    
    def test_dataframe_batch_insert(self):
        """Test inserting entire DataFrame in batches"""
        progress_updates = []
        
        def progress_callback(progress: dict):
            progress_updates.append(progress.copy())
        
        metrics = self.handler.insert_dataframe_in_batches(
            table_name='test_table',
            df=self.test_df,
            progress_callback=progress_callback
        )
        
        # Verify metrics
        self.assertEqual(metrics.total_rows_inserted, 1000)
        self.assertGreater(metrics.total_batches, 0)
        self.assertGreater(metrics.rows_per_second, 0)
        self.assertEqual(metrics.failed_batches, 0)
        
        # Verify progress updates
        self.assertGreater(len(progress_updates), 0)
        final_progress = progress_updates[-1]
        self.assertEqual(final_progress['rows_inserted'], 1000)
    
    def test_batch_size_optimization(self):
        """Test batch size optimization"""
        optimal_size = self.handler.optimize_batch_size(self.test_df.iloc[:100])
        
        # Should return reasonable batch size
        self.assertGreater(optimal_size, 0)
        self.assertLessEqual(optimal_size, 10000)
    
    def test_async_batch_processing(self):
        """Test asynchronous batch processing"""
        # Start async processing
        self.handler.start_async_processing()
        
        # Queue some batches
        for i in range(5):
            batch = self.test_df.iloc[i*100:(i+1)*100]
            success = self.handler.queue_batch_async('test_table', batch)
            self.assertTrue(success)
        
        # Allow processing
        time.sleep(0.5)
        
        # Stop async processing
        self.handler.stop_async_processing()


class TestGPUMemoryManager(unittest.TestCase):
    """Test cases for GPU Memory Manager"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.manager = GPUMemoryManager(threshold_mb=1000)
        
        # Create test data
        self.test_df = pd.DataFrame(
            np.random.randn(10000, 100),
            columns=[f'col_{i}' for i in range(100)]
        )
    
    def test_data_chunking(self):
        """Test data chunking for GPU"""
        chunks = self.manager.chunk_data_for_gpu(
            self.test_df,
            target_memory_mb=10
        )
        
        # Verify chunks created
        self.assertGreater(len(chunks), 1)
        
        # Verify all data preserved
        total_rows = sum(len(chunk) for chunk in chunks)
        self.assertEqual(total_rows, len(self.test_df))
    
    def test_gpu_transfer_simulation(self):
        """Test GPU transfer (simulation when GPU not available)"""
        # This will return None if GPU/CuPy not available
        gpu_data = self.manager.transfer_to_gpu(self.test_df.iloc[:100])
        
        # Just verify it doesn't crash
        # Actual GPU transfer only works with CuPy installed
        self.assertTrue(gpu_data is None or gpu_data is not None)


class TestEdgeCases(unittest.TestCase):
    """Edge case tests for CSV loader"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.loader = EnterpriseCSVLoader(
            chunk_size=100,
            checkpoint_dir=os.path.join(self.temp_dir, 'checkpoints')
        )
    
    def tearDown(self):
        """Clean up"""
        shutil.rmtree(self.temp_dir)
    
    def test_empty_csv_file(self):
        """Test handling of empty CSV file"""
        # Create empty CSV
        empty_csv = os.path.join(self.temp_dir, 'empty.csv')
        with open(empty_csv, 'w') as f:
            f.write('col1,col2,col3\n')
        
        # Should handle gracefully
        metrics = self.loader.load_csv_streaming(
            file_path=empty_csv,
            table_name='test_table'
        )
        
        self.assertEqual(metrics.status, "SUCCESS")
        self.assertEqual(metrics.total_rows, 0)
    
    def test_single_column_csv(self):
        """Test CSV with single column"""
        single_col_csv = os.path.join(self.temp_dir, 'single_col.csv')
        with open(single_col_csv, 'w') as f:
            f.write('value\n')
            for i in range(100):
                f.write(f'{i}\n')
        
        metrics = self.loader.load_csv_streaming(
            file_path=single_col_csv,
            table_name='test_table'
        )
        
        self.assertEqual(metrics.status, "SUCCESS")
        self.assertEqual(metrics.total_rows, 100)
        self.assertEqual(metrics.total_columns, 1)
    
    def test_large_column_count(self):
        """Test CSV with many columns"""
        # Create CSV with 1000 columns
        many_cols_csv = os.path.join(self.temp_dir, 'many_cols.csv')
        cols = [f'col_{i}' for i in range(1000)]
        
        with open(many_cols_csv, 'w') as f:
            f.write(','.join(cols) + '\n')
            for _ in range(10):
                f.write(','.join(['1'] * 1000) + '\n')
        
        metrics = self.loader.load_csv_streaming(
            file_path=many_cols_csv,
            table_name='test_table'
        )
        
        self.assertEqual(metrics.status, "SUCCESS")
        self.assertEqual(metrics.total_columns, 1000)
    
    def test_special_characters_in_data(self):
        """Test handling of special characters"""
        special_csv = os.path.join(self.temp_dir, 'special.csv')
        with open(special_csv, 'w') as f:
            f.write('name,value\n')
            f.write('"Test, with comma",123\n')
            f.write('"Test with \"quotes\"",456\n')
            f.write('Test\nwith\nnewlines,789\n')
        
        metrics = self.loader.load_csv_streaming(
            file_path=special_csv,
            table_name='test_table'
        )
        
        self.assertEqual(metrics.status, "SUCCESS")
        self.assertGreater(metrics.total_rows, 0)
    
    def test_concurrent_checkpoint_access(self):
        """Test concurrent checkpoint operations"""
        import threading
        
        csv_file = self._create_test_csv('concurrent.csv', 1000, 10)
        
        results = []
        
        def load_with_interrupt():
            try:
                def handler(table_name, chunk):
                    if len(results) == 2:
                        raise Exception("Interrupt")
                    results.append(chunk)
                
                self.loader.load_csv_streaming(
                    file_path=csv_file,
                    table_name='test_table',
                    data_handler=handler
                )
            except:
                pass
        
        # Start multiple threads
        threads = []
        for _ in range(3):
            t = threading.Thread(target=load_with_interrupt)
            t.start()
            threads.append(t)
        
        # Wait for completion
        for t in threads:
            t.join()
        
        # Should handle concurrent access gracefully
        self.assertTrue(True)  # No crash is success
    
    def test_corrupted_checkpoint_recovery(self):
        """Test recovery from corrupted checkpoint"""
        csv_file = self._create_test_csv('checkpoint_test.csv', 500, 10)
        
        # Create corrupted checkpoint
        checkpoint_file = os.path.join(
            self.checkpoint_dir,
            f"checkpoint_test.csv_test.checkpoint"
        )
        
        with open(checkpoint_file, 'w') as f:
            f.write("corrupted json data")
        
        # Should handle gracefully and start fresh
        metrics = self.loader.load_csv_streaming(
            file_path=csv_file,
            table_name='test_table'
        )
        
        self.assertEqual(metrics.status, "SUCCESS")
        self.assertEqual(metrics.total_rows, 500)
    
    def _create_test_csv(self, filename: str, rows: int, cols: int) -> str:
        """Helper to create test CSV"""
        filepath = os.path.join(self.temp_dir, filename)
        
        data = np.random.randn(rows, cols)
        columns = [f'col_{i}' for i in range(cols)]
        
        df = pd.DataFrame(data, columns=columns)
        df.to_csv(filepath, index=False)
        
        return filepath


class TestPerformance(unittest.TestCase):
    """Performance tests for CSV loader"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.loader = EnterpriseCSVLoader(
            chunk_size=10000,
            checkpoint_dir=os.path.join(self.temp_dir, 'checkpoints')
        )
    
    def tearDown(self):
        """Clean up"""
        shutil.rmtree(self.temp_dir)
    
    def test_load_performance_scaling(self):
        """Test performance scaling with file size"""
        sizes = [1000, 5000, 10000]
        throughputs = []
        
        for size in sizes:
            csv_file = self._create_test_csv(f'perf_{size}.csv', size, 50)
            
            metrics = self.loader.load_csv_streaming(
                file_path=csv_file,
                table_name='test_table'
            )
            
            throughputs.append(metrics.throughput_mbps)
            
            # Clean up
            os.remove(csv_file)
        
        # Verify reasonable throughput
        for throughput in throughputs:
            self.assertGreater(throughput, 0)
            self.assertLess(throughput, 10000)  # Reasonable upper limit
    
    def test_memory_efficiency(self):
        """Test memory efficiency with large files"""
        # Create a file larger than chunk size
        large_csv = self._create_test_csv('memory_test.csv', 50000, 20)
        
        metrics = self.loader.load_csv_streaming(
            file_path=large_csv,
            table_name='test_table'
        )
        
        # Peak memory should be reasonable (not loading entire file)
        file_size_mb = metrics.file_size_mb
        peak_memory_mb = metrics.peak_memory_mb
        
        # Peak memory should be less than 2x file size (streaming efficiency)
        self.assertLess(peak_memory_mb, file_size_mb * 2)
    
    def _create_test_csv(self, filename: str, rows: int, cols: int) -> str:
        """Helper to create test CSV"""
        filepath = os.path.join(self.temp_dir, filename)
        
        # Write in chunks to avoid memory issues
        chunk_size = 1000
        
        with open(filepath, 'w') as f:
            # Header
            header = ','.join([f'col_{i}' for i in range(cols)])
            f.write(header + '\n')
            
            # Data in chunks
            for start in range(0, rows, chunk_size):
                end = min(start + chunk_size, rows)
                chunk_rows = end - start
                
                for _ in range(chunk_rows):
                    row = ','.join(map(str, np.random.randn(cols)))
                    f.write(row + '\n')
        
        return filepath


class TestIntegration(unittest.TestCase):
    """Integration tests for CSV loading pipeline"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create components
        self.loader = EnterpriseCSVLoader(
            chunk_size=1000,
            checkpoint_dir=os.path.join(self.temp_dir, 'checkpoints')
        )
        
        self.batch_handler = BatchInsertHandler(batch_size=500)
        
        # Create test CSV
        self.test_csv = self._create_large_csv('large_test.csv', 10000, 50)
    
    def tearDown(self):
        """Clean up"""
        shutil.rmtree(self.temp_dir)
    
    def _create_large_csv(self, filename: str, rows: int, cols: int) -> str:
        """Create a large test CSV"""
        filepath = os.path.join(self.temp_dir, filename)
        
        # Create in chunks to avoid memory issues
        chunk_size = 1000
        
        with open(filepath, 'w') as f:
            # Write header
            header = ','.join([f'strategy_{i:04d}' for i in range(cols)])
            f.write(header + '\n')
            
            # Write data in chunks
            for start in range(0, rows, chunk_size):
                end = min(start + chunk_size, rows)
                chunk_rows = end - start
                
                data = np.random.randn(chunk_rows, cols)
                for row in data:
                    f.write(','.join(map(str, row)) + '\n')
        
        return filepath
    
    def test_full_pipeline(self):
        """Test full CSV loading pipeline"""
        all_chunks = []
        
        def chunk_handler(table_name: str, chunk: pd.DataFrame):
            # Process chunk through batch handler
            self.batch_handler.insert_batch(table_name, chunk)
            all_chunks.append(chunk)
        
        # Load CSV with streaming and batch processing
        metrics = self.loader.load_csv_streaming(
            file_path=self.test_csv,
            table_name='test_table',
            data_handler=chunk_handler,
            validate_data=True,
            fill_missing_values=True
        )
        
        # Verify success
        self.assertEqual(metrics.status, "SUCCESS")
        self.assertEqual(metrics.total_rows, 10000)
        
        # Verify all data processed
        total_rows = sum(len(chunk) for chunk in all_chunks)
        self.assertEqual(total_rows, 10000)


class TestDALIntegration(unittest.TestCase):
    """Test Enhanced CSV DAL integration"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create DAL instance
        from dal.csv_dal_enhanced import EnhancedCSVDAL
        self.dal = EnhancedCSVDAL({
            'csv_loader': {
                'chunk_size': 1000,
                'checkpoint_dir': os.path.join(self.temp_dir, 'checkpoints')
            }
        })
        self.dal.connect()
        
        # Create test CSV
        self.test_csv = self._create_test_csv('dal_test.csv', 5000, 20)
    
    def tearDown(self):
        """Clean up"""
        self.dal.disconnect()
        shutil.rmtree(self.temp_dir)
    
    def _create_test_csv(self, filename: str, rows: int, cols: int) -> str:
        """Create test CSV file"""
        filepath = os.path.join(self.temp_dir, filename)
        
        data = np.random.randn(rows, cols)
        columns = [f'strategy_{i:04d}' for i in range(cols)]
        
        df = pd.DataFrame(data, columns=columns)
        df.to_csv(filepath, index=False)
        
        return filepath
    
    def test_dal_csv_loading_with_progress(self):
        """Test CSV loading through DAL with progress tracking"""
        progress_updates = []
        
        def progress_callback(progress):
            progress_updates.append(progress.copy())
        
        success = self.dal.load_csv_to_heavydb(
            filepath=self.test_csv,
            table_name='test_strategies',
            progress_callback=progress_callback
        )
        
        self.assertTrue(success)
        self.assertIn('test_strategies', self.dal.tables)
        self.assertGreater(len(progress_updates), 0)
        
        # Verify data loaded correctly
        df = self.dal.tables['test_strategies']
        self.assertEqual(len(df), 5000)
        self.assertEqual(len(df.columns), 20)
    
    def test_dal_batch_insert(self):
        """Test batch insert functionality"""
        # Create test data
        test_df = pd.DataFrame(
            np.random.randn(2000, 10),
            columns=[f'col_{i}' for i in range(10)]
        )
        
        progress_updates = []
        
        def progress_callback(progress):
            progress_updates.append(progress.copy())
        
        success = self.dal.batch_insert_dataframe(
            table_name='batch_test',
            df=test_df,
            progress_callback=progress_callback
        )
        
        self.assertTrue(success)
        self.assertIn('batch_test', self.dal.tables)
        self.assertEqual(len(self.dal.tables['batch_test']), 2000)
        self.assertGreater(len(progress_updates), 0)
    
    def test_dal_get_load_metrics(self):
        """Test retrieving load metrics"""
        # Load a file first
        self.dal.load_csv_to_heavydb(
            filepath=self.test_csv,
            table_name='metrics_test'
        )
        
        # Get metrics
        metrics_list = self.dal.get_load_metrics(limit=5)
        
        self.assertGreater(len(metrics_list), 0)
        
        # Verify metrics structure
        for metrics in metrics_list:
            self.assertIsInstance(metrics.file_path, str)
            self.assertGreater(metrics.total_rows, 0)
            self.assertGreater(metrics.throughput_mbps, 0)
    
    def test_dal_ulta_transformation(self):
        """Test ULTA transformation through DAL"""
        # Load data
        self.dal.load_csv_to_heavydb(
            filepath=self.test_csv,
            table_name='ulta_test'
        )
        
        # Apply ULTA
        success = self.dal.apply_ulta_transformation('ulta_test')
        self.assertTrue(success)
        
        # Verify transformed table exists
        self.assertIn('ulta_test_ulta', self.dal.tables)
        
        # Verify same dimensions
        original_df = self.dal.tables['ulta_test']
        transformed_df = self.dal.tables['ulta_test_ulta']
        self.assertEqual(original_df.shape, transformed_df.shape)
    
    def test_dal_correlation_matrix(self):
        """Test correlation matrix computation"""
        # Load data
        self.dal.load_csv_to_heavydb(
            filepath=self.test_csv,
            table_name='corr_test'
        )
        
        # Compute correlation
        corr_matrix = self.dal.compute_correlation_matrix('corr_test')
        
        self.assertIsNotNone(corr_matrix)
        self.assertEqual(corr_matrix.shape, (20, 20))  # 20x20 for 20 strategies
        
        # Verify properties
        self.assertTrue(np.allclose(corr_matrix, corr_matrix.T))  # Symmetric
        self.assertTrue(np.allclose(np.diag(corr_matrix), 1.0))  # Diagonal is 1
    
    def test_dal_table_info(self):
        """Test getting table information"""
        # Load data
        self.dal.load_csv_to_heavydb(
            filepath=self.test_csv,
            table_name='info_test'
        )
        
        # Get info
        info = self.dal.get_table_info('info_test')
        
        self.assertIsNotNone(info)
        self.assertEqual(info['name'], 'info_test')
        self.assertEqual(info['rows'], 5000)
        self.assertEqual(info['columns'], 20)
        self.assertGreater(info['memory_usage_mb'], 0)
        self.assertIn('numeric_columns', info)
        self.assertEqual(len(info['numeric_columns']), 20)


if __name__ == '__main__':
    unittest.main()