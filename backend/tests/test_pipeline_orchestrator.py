#!/usr/bin/env python3
"""
Unit tests for Pipeline Orchestrator
"""

import unittest
import os
import sys
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
import pandas as pd

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline_orchestrator import PipelineOrchestrator


class TestPipelineOrchestrator(unittest.TestCase):
    """Test cases for Pipeline Orchestrator"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_dir = tempfile.mkdtemp()
        self.test_csv = os.path.join(self.test_dir, "test_data.csv")
        
        # Create test CSV
        test_data = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=10),
            'strategy1': [100 + i for i in range(10)],
            'strategy2': [200 + i*2 for i in range(10)],
            'strategy3': [300 + i*3 for i in range(10)]
        })
        test_data.to_csv(self.test_csv, index=False)
        
    def tearDown(self):
        """Clean up test environment"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
            
    def test_initialization(self):
        """Test orchestrator initialization"""
        orchestrator = PipelineOrchestrator()
        
        self.assertIsNotNone(orchestrator.config)
        self.assertIsNotNone(orchestrator.logger)
        self.assertEqual(len(orchestrator.pipeline_steps), 8)
        
    def test_configuration_validation(self):
        """Test configuration validation"""
        orchestrator = PipelineOrchestrator()
        validation = orchestrator.validate_configuration()
        
        self.assertTrue(validation['is_valid'])
        self.assertEqual(len(validation['issues']), 0)
        
    def test_supported_execution_modes(self):
        """Test getting supported execution modes"""
        orchestrator = PipelineOrchestrator()
        modes = orchestrator.get_supported_execution_modes()
        
        self.assertIn('full', modes)
        self.assertIn('zone-wise', modes)
        self.assertIn('dry-run', modes)
        
    def test_dry_run_execution(self):
        """Test dry-run mode execution"""
        orchestrator = PipelineOrchestrator()
        
        results = orchestrator.execute_pipeline(
            input_csv=self.test_csv,
            portfolio_size=2,
            execution_mode='dry-run'
        )
        
        self.assertEqual(results['status'], 'completed')
        self.assertEqual(results['execution_mode'], 'dry-run')
        self.assertEqual(len(results['step_timings']), 8)
        
        # Verify all steps were marked as dry-run
        for step_name, result in results['step_results'].items():
            self.assertEqual(result['status'], 'dry-run')
            
    def test_pipeline_step_order(self):
        """Test that pipeline steps are in correct order"""
        orchestrator = PipelineOrchestrator()
        
        expected_order = [
            "load_csv_to_database",
            "validate_data",
            "apply_ulta_transformation",
            "compute_correlation_matrix",
            "run_optimization_algorithms",
            "select_final_portfolio",
            "calculate_final_metrics",
            "generate_output_files"
        ]
        
        actual_order = [step[0] for step in orchestrator.pipeline_steps]
        self.assertEqual(actual_order, expected_order)
        
    def test_output_directory_creation(self):
        """Test output directory creation"""
        orchestrator = PipelineOrchestrator()
        
        with patch('os.makedirs') as mock_makedirs:
            output_dir = orchestrator._create_output_directory()
            
            self.assertTrue(output_dir.startswith('/mnt/optimizer_share/output/run_'))
            # Note: makedirs is called via Path.mkdir()
            
    @patch('backend.dal.dal_factory.DALFactory.create_dal')
    def test_dal_initialization(self, mock_create_dal):
        """Test DAL initialization"""
        orchestrator = PipelineOrchestrator()
        
        mock_dal = Mock()
        mock_create_dal.return_value = mock_dal
        
        context = {}
        orchestrator._initialize_dal(context)
        
        self.assertEqual(orchestrator.dal, mock_dal)
        self.assertIsNotNone(orchestrator.zone_optimizer)
        mock_create_dal.assert_called_once_with('csv', orchestrator.config_path)
        
    def test_error_handling(self):
        """Test error handling in pipeline execution"""
        orchestrator = PipelineOrchestrator()
        
        # Test with non-existent file
        with self.assertRaises(Exception):
            orchestrator.execute_pipeline(
                input_csv='/non/existent/file.csv',
                portfolio_size=10,
                execution_mode='full'
            )
            
    def test_invalid_execution_mode(self):
        """Test handling of invalid execution mode"""
        orchestrator = PipelineOrchestrator()
        
        with self.assertRaises(ValueError) as context:
            orchestrator.execute_pipeline(
                input_csv=self.test_csv,
                portfolio_size=10,
                execution_mode='invalid_mode'
            )
            
        self.assertIn('Invalid execution mode', str(context.exception))
        
    @patch('backend.pipeline_orchestrator.PipelineOrchestrator._step_load_csv')
    def test_step_execution_tracking(self, mock_load_csv):
        """Test that step execution is properly tracked"""
        orchestrator = PipelineOrchestrator()
        
        # Mock the step to return success
        mock_load_csv.return_value = {
            'status': 'success',
            'rows_loaded': 100,
            'columns_loaded': 10
        }
        
        # Execute only the first step by mocking the pipeline steps
        orchestrator.pipeline_steps = [("load_csv_to_database", mock_load_csv)]
        
        results = orchestrator.execute_pipeline(
            input_csv=self.test_csv,
            portfolio_size=10,
            execution_mode='full'
        )
        
        # Verify step was called and tracked
        self.assertIn('load_csv_to_database', results['step_timings'])
        self.assertIn('load_csv_to_database', results['step_results'])
        self.assertEqual(
            results['step_results']['load_csv_to_database']['status'], 
            'success'
        )


class TestPipelineOrchestratorIntegration(unittest.TestCase):
    """Integration tests for Pipeline Orchestrator"""
    
    def setUp(self):
        """Set up integration test environment"""
        self.test_csv = "/mnt/optimizer_share/input/Python_Multi_Consolidated_20250726_161921.csv"
        
    def test_real_csv_dry_run(self):
        """Test with real CSV file in dry-run mode"""
        if not os.path.exists(self.test_csv):
            self.skipTest(f"Test CSV not found: {self.test_csv}")
            
        orchestrator = PipelineOrchestrator()
        
        results = orchestrator.execute_pipeline(
            input_csv=self.test_csv,
            portfolio_size=5,
            execution_mode='dry-run'
        )
        
        self.assertEqual(results['status'], 'completed')
        self.assertIsNotNone(results['output_directory'])
        self.assertTrue(os.path.exists(results['output_directory']))


if __name__ == '__main__':
    unittest.main()