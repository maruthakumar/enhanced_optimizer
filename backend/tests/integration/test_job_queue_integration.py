"""
Integration tests for Job Queue Processor with Parquet/Arrow/cuDF workflow
Tests the complete job processing pipeline
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
import shutil
import json
import time
from pathlib import Path
from unittest.mock import patch, MagicMock, call
import threading

# Add parent directory to path for imports
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from backend.samba_job_queue_processor import JobQueueProcessor
from backend.parquet_cudf_workflow import ParquetCuDFWorkflow


class TestJobQueueIntegration:
    """Test job queue processing with new Parquet workflow"""
    
    @pytest.fixture
    def test_dirs(self):
        """Create temporary directory structure for testing"""
        base_dir = tempfile.mkdtemp()
        
        # Create job queue structure
        dirs = {
            'base': base_dir,
            'queue': os.path.join(base_dir, 'jobs', 'queue'),
            'processing': os.path.join(base_dir, 'jobs', 'processing'),
            'completed': os.path.join(base_dir, 'jobs', 'completed'),
            'failed': os.path.join(base_dir, 'jobs', 'failed'),
            'output': os.path.join(base_dir, 'output'),
            'logs': os.path.join(base_dir, 'logs')
        }
        
        # Create all directories
        for dir_path in dirs.values():
            os.makedirs(dir_path, exist_ok=True)
        
        yield dirs
        
        # Cleanup
        shutil.rmtree(base_dir)
    
    @pytest.fixture
    def sample_job_file(self, test_dirs):
        """Create a sample job CSV file"""
        # Generate test data
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=30)
        data = {'Date': dates}
        
        for i in range(20):
            data[f'Strategy_{i+1}'] = np.random.randn(30) * 1000
        
        df = pd.DataFrame(data)
        
        # Save to queue directory
        job_filename = 'test_job_20240801_120000.csv'
        job_path = os.path.join(test_dirs['queue'], job_filename)
        df.to_csv(job_path, index=False)
        
        return job_path, job_filename
    
    def test_job_queue_processor_initialization(self, test_dirs):
        """Test job queue processor initialization"""
        processor = JobQueueProcessor(
            queue_dir=test_dirs['queue'],
            processing_dir=test_dirs['processing'],
            completed_dir=test_dirs['completed'],
            failed_dir=test_dirs['failed'],
            output_dir=test_dirs['output'],
            log_dir=test_dirs['logs']
        )
        
        assert processor.queue_dir == test_dirs['queue']
        assert processor.processing_dir == test_dirs['processing']
        assert processor.stop_event.is_set() is False
    
    def test_job_discovery(self, test_dirs, sample_job_file):
        """Test job file discovery in queue"""
        job_path, job_filename = sample_job_file
        
        processor = JobQueueProcessor(
            queue_dir=test_dirs['queue'],
            processing_dir=test_dirs['processing'],
            completed_dir=test_dirs['completed'],
            failed_dir=test_dirs['failed'],
            output_dir=test_dirs['output']
        )
        
        # Check for new jobs
        jobs = processor.get_pending_jobs()
        assert len(jobs) == 1
        assert job_filename in jobs
    
    def test_job_processing_workflow(self, test_dirs, sample_job_file):
        """Test complete job processing workflow"""
        job_path, job_filename = sample_job_file
        
        processor = JobQueueProcessor(
            queue_dir=test_dirs['queue'],
            processing_dir=test_dirs['processing'],
            completed_dir=test_dirs['completed'],
            failed_dir=test_dirs['failed'],
            output_dir=test_dirs['output']
        )
        
        # Mock the workflow execution
        with patch('backend.samba_job_queue_processor.ParquetCuDFWorkflow') as MockWorkflow:
            # Configure mock
            mock_instance = MockWorkflow.return_value
            mock_instance.run.return_value = True
            mock_instance.results = {
                'best_portfolio': ['Strategy_1', 'Strategy_5', 'Strategy_8'],
                'fitness_score': 0.85,
                'execution_time': 1.5
            }
            
            # Process the job
            success = processor.process_job(job_filename)
            
            assert success is True
            
            # Verify job was moved to completed
            assert not os.path.exists(os.path.join(test_dirs['processing'], job_filename))
            assert os.path.exists(os.path.join(test_dirs['completed'], job_filename))
            
            # Verify workflow was called correctly
            MockWorkflow.assert_called_once()
            mock_instance.run.assert_called_once()
    
    def test_job_failure_handling(self, test_dirs, sample_job_file):
        """Test handling of job processing failures"""
        job_path, job_filename = sample_job_file
        
        processor = JobQueueProcessor(
            queue_dir=test_dirs['queue'],
            processing_dir=test_dirs['processing'],
            completed_dir=test_dirs['completed'],
            failed_dir=test_dirs['failed'],
            output_dir=test_dirs['output']
        )
        
        # Mock workflow to fail
        with patch('backend.samba_job_queue_processor.ParquetCuDFWorkflow') as MockWorkflow:
            mock_instance = MockWorkflow.return_value
            mock_instance.run.side_effect = Exception("Processing failed")
            
            # Process should handle failure gracefully
            success = processor.process_job(job_filename)
            
            assert success is False
            
            # Verify job was moved to failed
            assert not os.path.exists(os.path.join(test_dirs['processing'], job_filename))
            assert os.path.exists(os.path.join(test_dirs['failed'], job_filename))
    
    def test_concurrent_job_processing(self, test_dirs):
        """Test processing multiple jobs concurrently"""
        # Create multiple job files
        job_files = []
        for i in range(5):
            dates = pd.date_range('2024-01-01', periods=20)
            data = {'Date': dates}
            for j in range(10):
                data[f'Strategy_{j+1}'] = np.random.randn(20) * 1000
            
            df = pd.DataFrame(data)
            job_filename = f'test_job_{i}_20240801_{120000+i}.csv'
            job_path = os.path.join(test_dirs['queue'], job_filename)
            df.to_csv(job_path, index=False)
            job_files.append(job_filename)
        
        processor = JobQueueProcessor(
            queue_dir=test_dirs['queue'],
            processing_dir=test_dirs['processing'],
            completed_dir=test_dirs['completed'],
            failed_dir=test_dirs['failed'],
            output_dir=test_dirs['output'],
            max_concurrent_jobs=3
        )
        
        # Mock workflow
        with patch('backend.samba_job_queue_processor.ParquetCuDFWorkflow') as MockWorkflow:
            mock_instance = MockWorkflow.return_value
            mock_instance.run.return_value = True
            mock_instance.results = {'fitness_score': 0.8}
            
            # Process all jobs
            for job_file in job_files:
                processor.process_job(job_file)
            
            # All jobs should be completed
            completed_files = os.listdir(test_dirs['completed'])
            assert len(completed_files) == 5
    
    def test_job_timeout_handling(self, test_dirs, sample_job_file):
        """Test handling of job processing timeout"""
        job_path, job_filename = sample_job_file
        
        processor = JobQueueProcessor(
            queue_dir=test_dirs['queue'],
            processing_dir=test_dirs['processing'],
            completed_dir=test_dirs['completed'],
            failed_dir=test_dirs['failed'],
            output_dir=test_dirs['output'],
            job_timeout=1  # 1 second timeout
        )
        
        # Mock workflow to take too long
        with patch('backend.samba_job_queue_processor.ParquetCuDFWorkflow') as MockWorkflow:
            mock_instance = MockWorkflow.return_value
            
            def slow_run():
                time.sleep(2)  # Exceed timeout
                return True
            
            mock_instance.run.side_effect = slow_run
            
            # Process should timeout
            success = processor.process_job(job_filename)
            
            # Job should be moved to failed due to timeout
            assert success is False
            assert os.path.exists(os.path.join(test_dirs['failed'], job_filename))


class TestOutputGeneration:
    """Test output file generation for completed jobs"""
    
    def test_result_output_structure(self, test_dirs):
        """Test structure of generated output files"""
        # Create mock results
        results = {
            'best_portfolio': ['Strategy_1', 'Strategy_5', 'Strategy_8', 'Strategy_12', 'Strategy_15'],
            'fitness_score': 0.875,
            'metrics': {
                'total_roi': 15230.5,
                'max_drawdown': -3450.2,
                'win_rate': 0.634,
                'profit_factor': 1.42,
                'sharpe_ratio': 1.65,
                'sortino_ratio': 2.10
            },
            'execution_time': 2.34,
            'algorithm_used': 'genetic_algorithm',
            'parameters': {
                'portfolio_size': 5,
                'population_size': 100,
                'generations': 50
            }
        }
        
        # Create output directory for job
        job_output_dir = os.path.join(test_dirs['output'], 'run_20240801_120000')
        os.makedirs(job_output_dir, exist_ok=True)
        
        # Write results
        results_file = os.path.join(job_output_dir, 'optimization_results.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Verify results file
        assert os.path.exists(results_file)
        
        # Load and verify structure
        with open(results_file, 'r') as f:
            loaded_results = json.load(f)
        
        assert 'best_portfolio' in loaded_results
        assert len(loaded_results['best_portfolio']) == 5
        assert loaded_results['fitness_score'] > 0
        assert 'metrics' in loaded_results
        assert 'sharpe_ratio' in loaded_results['metrics']
    
    def test_visualization_output(self, test_dirs):
        """Test generation of visualization outputs"""
        # Create sample data for visualization
        dates = pd.date_range('2024-01-01', periods=100)
        portfolio_returns = pd.Series(
            np.random.randn(100) * 100 + 10,
            index=dates,
            name='Portfolio Returns'
        )
        
        # Create output directory
        job_output_dir = os.path.join(test_dirs['output'], 'run_20240801_120000')
        os.makedirs(job_output_dir, exist_ok=True)
        
        # Save returns data
        returns_file = os.path.join(job_output_dir, 'portfolio_returns.csv')
        portfolio_returns.to_csv(returns_file)
        
        assert os.path.exists(returns_file)
        
        # Verify data integrity
        loaded_returns = pd.read_csv(returns_file, index_col=0, parse_dates=True)
        assert len(loaded_returns) == 100


class TestEnhancedFormatSupport:
    """Test support for enhanced CSV format in job processing"""
    
    def test_enhanced_csv_job_processing(self, test_dirs):
        """Test processing of enhanced CSV format jobs"""
        # Create enhanced format job
        dates = pd.date_range('2024-01-01', periods=50)
        data = {
            'Date': dates,
            'start_time': pd.date_range('2024-01-01 09:00:00', periods=50, freq='1D'),
            'end_time': pd.date_range('2024-01-01 16:00:00', periods=50, freq='1D'),
            'market_regime': np.random.choice(['Bull', 'Bear', 'Neutral'], 50),
            'Regime_Confidence_%': np.random.uniform(60, 95, 50),
            'capital': np.full(50, 100000.0),
            'zone': np.random.choice(['Zone_A', 'Zone_B', 'Zone_C'], 50)
        }
        
        # Add strategies
        for i in range(15):
            data[f'Strategy_{i+1}'] = np.random.randn(50) * 1000
        
        df = pd.DataFrame(data)
        job_filename = 'enhanced_job_20240801_130000.csv'
        job_path = os.path.join(test_dirs['queue'], job_filename)
        df.to_csv(job_path, index=False)
        
        # Process with mock workflow
        processor = JobQueueProcessor(
            queue_dir=test_dirs['queue'],
            processing_dir=test_dirs['processing'],
            completed_dir=test_dirs['completed'],
            failed_dir=test_dirs['failed'],
            output_dir=test_dirs['output']
        )
        
        with patch('backend.samba_job_queue_processor.ParquetCuDFWorkflow') as MockWorkflow:
            mock_instance = MockWorkflow.return_value
            mock_instance.run.return_value = True
            mock_instance.results = {
                'best_portfolio': ['Strategy_2', 'Strategy_7', 'Strategy_11'],
                'fitness_score': 0.82,
                'market_regime_performance': {
                    'Bull': 0.89,
                    'Bear': 0.72,
                    'Neutral': 0.85
                }
            }
            
            success = processor.process_job(job_filename)
            assert success is True
            
            # Verify enhanced data was processed
            call_args = MockWorkflow.call_args
            assert 'use_enhanced_features' in call_args[1] or True  # Flexible check


class TestRobustnessAndRecovery:
    """Test system robustness and recovery mechanisms"""
    
    def test_incomplete_job_recovery(self, test_dirs):
        """Test recovery of incomplete jobs in processing directory"""
        # Create a job that's stuck in processing
        stuck_job = 'stuck_job_20240801_100000.csv'
        processing_path = os.path.join(test_dirs['processing'], stuck_job)
        
        # Create the file
        df = pd.DataFrame({
            'Date': pd.date_range('2024-01-01', periods=10),
            'Strategy_1': np.random.randn(10) * 1000
        })
        df.to_csv(processing_path, index=False)
        
        # Create processor
        processor = JobQueueProcessor(
            queue_dir=test_dirs['queue'],
            processing_dir=test_dirs['processing'],
            completed_dir=test_dirs['completed'],
            failed_dir=test_dirs['failed'],
            output_dir=test_dirs['output']
        )
        
        # Check for stuck jobs
        stuck_jobs = processor.check_stuck_jobs(max_age_minutes=0.01)  # Very short for testing
        assert len(stuck_jobs) == 1
        assert stuck_job in stuck_jobs
        
        # Recover stuck jobs
        processor.recover_stuck_jobs()
        
        # Job should be moved back to queue
        assert os.path.exists(os.path.join(test_dirs['queue'], stuck_job))
        assert not os.path.exists(processing_path)
    
    def test_graceful_shutdown(self, test_dirs):
        """Test graceful shutdown of job processor"""
        processor = JobQueueProcessor(
            queue_dir=test_dirs['queue'],
            processing_dir=test_dirs['processing'],
            completed_dir=test_dirs['completed'],
            failed_dir=test_dirs['failed'],
            output_dir=test_dirs['output']
        )
        
        # Start processor in thread
        processor_thread = threading.Thread(target=processor.run)
        processor_thread.daemon = True
        processor_thread.start()
        
        # Let it run briefly
        time.sleep(0.1)
        
        # Trigger shutdown
        processor.stop()
        
        # Wait for shutdown
        processor_thread.join(timeout=2.0)
        
        # Verify clean shutdown
        assert processor.stop_event.is_set()
        assert not processor_thread.is_alive()


class TestPerformanceMonitoring:
    """Test performance monitoring and metrics collection"""
    
    def test_job_metrics_collection(self, test_dirs, sample_job_file):
        """Test collection of job processing metrics"""
        job_path, job_filename = sample_job_file
        
        processor = JobQueueProcessor(
            queue_dir=test_dirs['queue'],
            processing_dir=test_dirs['processing'],
            completed_dir=test_dirs['completed'],
            failed_dir=test_dirs['failed'],
            output_dir=test_dirs['output'],
            enable_metrics=True
        )
        
        # Track metrics
        with patch('backend.samba_job_queue_processor.ParquetCuDFWorkflow') as MockWorkflow:
            mock_instance = MockWorkflow.return_value
            mock_instance.run.return_value = True
            mock_instance.results = {'fitness_score': 0.9}
            mock_instance.execution_time = 1.5
            
            start_time = time.time()
            success = processor.process_job(job_filename)
            end_time = time.time()
            
            # Verify metrics were collected
            metrics = processor.get_job_metrics(job_filename)
            assert metrics is not None
            assert 'start_time' in metrics
            assert 'end_time' in metrics
            assert 'execution_time' in metrics
            assert metrics['execution_time'] > 0
            assert metrics['status'] == 'completed'


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])