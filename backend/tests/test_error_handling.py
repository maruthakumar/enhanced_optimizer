#!/usr/bin/env python3
"""
Test Suite for Error Handling Implementation
Validates all error handling components work correctly
"""

import os
import sys
import time
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import error handling components
from lib.error_handling import (
    CheckpointManager, retry, RetryError,
    ErrorNotifier, ContextLogger, ErrorRecoveryManager,
    DataProcessingError, AlgorithmTimeoutError, CSVLoadError
)

class TestErrorHandling:
    """Test error handling components"""
    
    def __init__(self):
        self.test_dir = Path(tempfile.mkdtemp())
        self.results = {
            'passed': 0,
            'failed': 0,
            'tests': []
        }
        print(f"Test directory: {self.test_dir}")
    
    def cleanup(self):
        """Cleanup test directory"""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    def run_test(self, test_name, test_func):
        """Run a single test"""
        print(f"\n{'='*60}")
        print(f"Running: {test_name}")
        print('='*60)
        
        try:
            test_func()
            self.results['passed'] += 1
            self.results['tests'].append({
                'name': test_name,
                'status': 'PASSED',
                'error': None
            })
            print(f"✅ {test_name}: PASSED")
            
        except Exception as e:
            self.results['failed'] += 1
            self.results['tests'].append({
                'name': test_name,
                'status': 'FAILED',
                'error': str(e)
            })
            print(f"❌ {test_name}: FAILED - {e}")
            import traceback
            traceback.print_exc()
    
    def test_checkpoint_manager(self):
        """Test CheckpointManager functionality"""
        cm = CheckpointManager(base_dir=str(self.test_dir / "checkpoints"))
        
        # Test save checkpoint
        test_state = {
            'data': [1, 2, 3],
            'config': {'key': 'value'},
            'timestamp': datetime.now().isoformat()
        }
        
        success = cm.save_checkpoint(test_state, 'test_checkpoint', 'Test checkpoint')
        assert success, "Failed to save checkpoint"
        
        # Test load checkpoint
        loaded_state = cm.load_checkpoint('test_checkpoint')
        assert loaded_state is not None, "Failed to load checkpoint"
        assert loaded_state['data'] == test_state['data'], "Checkpoint data mismatch"
        
        # Test list checkpoints
        checkpoints = cm.list_checkpoints()
        assert len(checkpoints) > 0, "No checkpoints found"
        assert checkpoints[0]['name'] == 'test_checkpoint', "Checkpoint name mismatch"
        
        # Test get latest checkpoint
        latest = cm.get_latest_checkpoint()
        assert latest == 'test_checkpoint', "Latest checkpoint mismatch"
        
        # Test delete checkpoint
        success = cm.delete_checkpoint('test_checkpoint')
        assert success, "Failed to delete checkpoint"
        
        # Verify deletion
        loaded = cm.load_checkpoint('test_checkpoint')
        assert loaded is None, "Checkpoint not deleted"
        
        print("CheckpointManager tests completed successfully")
    
    def test_retry_decorator(self):
        """Test retry decorator functionality"""
        attempt_count = 0
        
        @retry(max_attempts=3, delay=0.1, backoff="exponential")
        def failing_function():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise ValueError("Simulated failure")
            return "Success"
        
        # Test successful retry
        result = failing_function()
        assert result == "Success", "Retry did not succeed"
        assert attempt_count == 3, f"Expected 3 attempts, got {attempt_count}"
        
        # Test retry exhaustion
        attempt_count = 0
        
        @retry(max_attempts=2, delay=0.1)
        def always_failing():
            nonlocal attempt_count
            attempt_count += 1
            raise ValueError("Always fails")
        
        try:
            always_failing()
            assert False, "Should have raised RetryError"
        except RetryError as e:
            assert e.attempts == 2, f"Expected 2 attempts, got {e.attempts}"
            assert attempt_count == 2, "Incorrect attempt count"
        
        print("Retry decorator tests completed successfully")
    
    def test_context_logger(self):
        """Test ContextLogger functionality"""
        logger = ContextLogger("test_logger", str(self.test_dir / "logs"))
        
        # Test context setting
        logger.set_context(job_id="test_job", user="test_user")
        
        # Test logging with context
        logger.error("Test error message", include_locals=True)
        
        # Verify log file exists
        log_files = list((self.test_dir / "logs").glob("*.log"))
        assert len(log_files) > 0, "No log files created"
        
        # Test error detail file creation
        error_files = list((self.test_dir / "logs" / "errors").glob("*.json"))
        assert len(error_files) > 0, "No error detail files created"
        
        # Read and verify error detail
        with open(error_files[0], 'r') as f:
            error_detail = json.load(f)
        
        assert error_detail['message'] == "Test error message", "Error message mismatch"
        assert 'context' in error_detail, "Context missing from error detail"
        assert error_detail['context']['custom_context']['job_id'] == "test_job", "Context data mismatch"
        
        print("ContextLogger tests completed successfully")
    
    def test_error_recovery_manager(self):
        """Test ErrorRecoveryManager functionality"""
        cm = CheckpointManager(base_dir=str(self.test_dir / "checkpoints"))
        rm = ErrorRecoveryManager(checkpoint_manager=cm)
        
        # Test recoverable error handling
        error = DataProcessingError("Test data error", recovery_action="Test recovery")
        context = {'data': 'test_data'}
        
        result = rm.handle_error(error, context, allow_recovery=True)
        
        # Verify recovery was attempted
        assert len(rm.recovery_history) > 0, "No recovery history"
        assert rm.recovery_history[0]['error_type'] == 'DataProcessingError', "Wrong error type in history"
        
        # Test non-recoverable error
        from lib.error_handling.error_types import CriticalError
        critical_error = CriticalError("Critical test error")
        
        result = rm.handle_error(critical_error, context)
        assert result is None, "Critical error should not recover"
        
        # Test recovery report
        report = rm.create_recovery_report()
        assert report['total_attempts'] > 0, "No recovery attempts in report"
        
        print("ErrorRecoveryManager tests completed successfully")
    
    def test_error_notifier(self):
        """Test ErrorNotifier functionality"""
        # Create test config
        config_file = self.test_dir / "notifier_config.json"
        config = {
            'email': {
                'enabled': False,  # Disable actual sending
                'recipients': ['test@example.com']
            },
            'slack': {
                'enabled': False,  # Disable actual sending
                'webhook_url': 'https://hooks.slack.com/test'
            },
            'rate_limit': 10
        }
        
        with open(config_file, 'w') as f:
            json.dump(config, f)
        
        notifier = ErrorNotifier(str(config_file))
        
        # Test notification preparation
        test_error = ValueError("Test error for notification")
        context = {
            'job_id': 'test_job_123',
            'timestamp': datetime.now().isoformat()
        }
        
        # This should not actually send notifications (disabled in config)
        success = notifier.notify_critical_error(test_error, context, "CRITICAL")
        
        # Verify notification would have been sent
        assert len(notifier.notification_history) == 0, "Notification sent when disabled"
        
        # Test rate limiting
        for i in range(12):
            notifier.notification_history.append({
                'timestamp': datetime.now(),
                'error_type': 'TestError',
                'severity': 'CRITICAL'
            })
        
        # Should be rate limited now
        assert not notifier._check_rate_limit(), "Rate limiting not working"
        
        print("ErrorNotifier tests completed successfully")
    
    def test_custom_exceptions(self):
        """Test custom exception types"""
        # Test CSVLoadError
        error = CSVLoadError("Failed to load CSV", "/path/to/file.csv", 42)
        assert error.context['file_path'] == "/path/to/file.csv", "File path not in context"
        assert error.context['line_number'] == 42, "Line number not in context"
        assert error.recoverable, "CSVLoadError should be recoverable"
        
        # Test AlgorithmTimeoutError
        error = AlgorithmTimeoutError("TestAlgorithm", 30.0)
        assert "TestAlgorithm" in str(error), "Algorithm name not in message"
        assert error.context['timeout'] == 30.0, "Timeout not in context"
        
        print("Custom exceptions tests completed successfully")
    
    def test_integration(self):
        """Test integration of all components"""
        # Create a mock workflow with error handling
        cm = CheckpointManager(base_dir=str(self.test_dir / "checkpoints"))
        logger = ContextLogger("integration_test", str(self.test_dir / "logs"))
        rm = ErrorRecoveryManager(checkpoint_manager=cm)
        
        # Simulate workflow with checkpoints
        workflow_state = {'stage': 'start', 'data': []}
        
        # Save initial checkpoint
        cm.save_checkpoint(workflow_state, 'workflow_start', 'Initial state')
        
        # Simulate error and recovery
        try:
            # Simulate processing that fails
            workflow_state['stage'] = 'processing'
            raise DataProcessingError("Simulated processing error")
            
        except DataProcessingError as e:
            logger.error("Processing failed", exc_info=True)
            
            # Attempt recovery
            recovery_result = rm.handle_error(e, {'state': workflow_state})
            
            # Load from checkpoint
            if not recovery_result:
                recovered_state = cm.load_checkpoint('workflow_start')
                assert recovered_state is not None, "Failed to load checkpoint"
                assert recovered_state['stage'] == 'start', "Wrong checkpoint loaded"
        
        print("Integration tests completed successfully")
    
    def run_all_tests(self):
        """Run all tests"""
        print("\n" + "="*60)
        print("ERROR HANDLING TEST SUITE")
        print("="*60)
        
        tests = [
            ("CheckpointManager", self.test_checkpoint_manager),
            ("Retry Decorator", self.test_retry_decorator),
            ("ContextLogger", self.test_context_logger),
            ("ErrorRecoveryManager", self.test_error_recovery_manager),
            ("ErrorNotifier", self.test_error_notifier),
            ("Custom Exceptions", self.test_custom_exceptions),
            ("Integration", self.test_integration)
        ]
        
        for test_name, test_func in tests:
            self.run_test(test_name, test_func)
        
        # Print summary
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        print(f"Total Tests: {self.results['passed'] + self.results['failed']}")
        print(f"Passed: {self.results['passed']}")
        print(f"Failed: {self.results['failed']}")
        print(f"Success Rate: {(self.results['passed'] / (self.results['passed'] + self.results['failed']) * 100):.1f}%")
        
        # Save test report
        report_file = self.test_dir.parent / "error_handling_test_report.json"
        with open(report_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\nTest report saved to: {report_file}")
        
        return self.results['failed'] == 0


def main():
    """Main test runner"""
    tester = TestErrorHandling()
    
    try:
        success = tester.run_all_tests()
        
        if success:
            print("\n✅ All error handling tests passed!")
            exit_code = 0
        else:
            print("\n❌ Some error handling tests failed!")
            exit_code = 1
            
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
        exit_code = 2
        
    finally:
        tester.cleanup()
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()