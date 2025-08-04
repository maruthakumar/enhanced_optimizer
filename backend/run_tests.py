#!/usr/bin/env python3
"""
Comprehensive test runner for Heavy Optimizer Platform
Runs all test suites with proper reporting
"""

import sys
import subprocess
import time
from pathlib import Path
from datetime import datetime
import argparse
import json


class TestRunner:
    """Run all test suites with reporting"""
    
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'suites': {},
            'summary': {
                'total_suites': 0,
                'passed_suites': 0,
                'failed_suites': 0,
                'total_time': 0
            }
        }
        self.backend_dir = Path(__file__).parent
    
    def run_command(self, cmd, suite_name):
        """Run a command and capture output"""
        print(f"\n{'='*60}")
        print(f"Running {suite_name}...")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        if self.verbose:
            # Run with output visible
            result = subprocess.run(cmd, shell=True, cwd=self.backend_dir)
        else:
            # Capture output
            result = subprocess.run(
                cmd, 
                shell=True, 
                cwd=self.backend_dir,
                capture_output=True,
                text=True
            )
        
        elapsed_time = time.time() - start_time
        
        # Record results
        self.results['suites'][suite_name] = {
            'command': cmd,
            'return_code': result.returncode,
            'elapsed_time': elapsed_time,
            'success': result.returncode == 0
        }
        
        if result.returncode == 0:
            print(f"✅ {suite_name} PASSED ({elapsed_time:.2f}s)")
            self.results['summary']['passed_suites'] += 1
        else:
            print(f"❌ {suite_name} FAILED ({elapsed_time:.2f}s)")
            self.results['summary']['failed_suites'] += 1
            if not self.verbose and result.stderr:
                print(f"Error output:\n{result.stderr}")
        
        self.results['summary']['total_suites'] += 1
        self.results['summary']['total_time'] += elapsed_time
        
        return result.returncode == 0
    
    def run_unit_tests(self):
        """Run unit tests"""
        return self.run_command(
            "python -m pytest tests/unit/ -v --cov=lib --cov-report=term-missing",
            "Unit Tests"
        )
    
    def run_integration_tests(self):
        """Run integration tests"""
        return self.run_command(
            "python -m pytest tests/integration/ -v --timeout=300",
            "Integration Tests"
        )
    
    def run_algorithm_tests(self):
        """Run algorithm-specific tests"""
        return self.run_command(
            "python -m pytest tests/test_all_algorithms.py -v",
            "Algorithm Tests"
        )
    
    def run_performance_tests(self):
        """Run performance benchmarks"""
        return self.run_command(
            "python -m pytest tests/performance/ -v --benchmark-only",
            "Performance Tests"
        )
    
    def run_gpu_tests(self):
        """Run GPU-specific tests (if GPU available)"""
        # Check if GPU is available
        try:
            import cudf
            import cupy
            gpu_available = True
        except ImportError:
            gpu_available = False
        
        if gpu_available:
            return self.run_command(
                "python -m pytest tests/gpu/ -v -m gpu",
                "GPU Tests"
            )
        else:
            print("\n⚠️  Skipping GPU tests (CUDA/cuDF not available)")
            return True
    
    def run_code_quality_checks(self):
        """Run code quality checks"""
        checks_passed = True
        
        # Black formatting check
        print("\nChecking code formatting with Black...")
        if not self.run_command(
            "python -m black --check . --exclude='tests/data|lib/legacy'",
            "Black Formatting"
        ):
            checks_passed = False
            print("  💡 Run 'black .' to auto-format code")
        
        # Flake8 linting
        print("\nChecking code style with Flake8...")
        if not self.run_command(
            "python -m flake8 . --config=../setup.cfg",
            "Flake8 Linting"
        ):
            checks_passed = False
        
        # isort import sorting
        print("\nChecking import sorting with isort...")
        if not self.run_command(
            "python -m isort --check-only . --settings-path=../setup.cfg",
            "Import Sorting"
        ):
            checks_passed = False
            print("  💡 Run 'isort .' to fix import ordering")
        
        # MyPy type checking (optional, often has false positives)
        print("\nChecking types with MyPy...")
        self.run_command(
            "python -m mypy . --config-file=../setup.cfg --ignore-missing-imports",
            "MyPy Type Checking"
        )  # Don't fail on MyPy errors
        
        return checks_passed
    
    def run_security_scan(self):
        """Run security vulnerability scan"""
        return self.run_command(
            "python -m bandit -r . -ll -x tests/",
            "Security Scan"
        )
    
    def generate_report(self):
        """Generate test report"""
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        
        summary = self.results['summary']
        print(f"Total Test Suites: {summary['total_suites']}")
        print(f"Passed: {summary['passed_suites']}")
        print(f"Failed: {summary['failed_suites']}")
        print(f"Total Time: {summary['total_time']:.2f}s")
        
        if summary['failed_suites'] > 0:
            print("\nFailed Suites:")
            for suite_name, result in self.results['suites'].items():
                if not result['success']:
                    print(f"  - {suite_name}")
        
        # Save detailed report
        report_file = self.backend_dir / 'test_report.json'
        with open(report_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\nDetailed report saved to: {report_file}")
        
        return summary['failed_suites'] == 0
    
    def run_all(self, skip_quality=False, skip_performance=False, skip_gpu=False):
        """Run all test suites"""
        print("Starting Heavy Optimizer Platform Test Suite")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Run test suites
        self.run_unit_tests()
        self.run_integration_tests()
        self.run_algorithm_tests()
        
        if not skip_performance:
            self.run_performance_tests()
        
        if not skip_gpu:
            self.run_gpu_tests()
        
        if not skip_quality:
            self.run_code_quality_checks()
        
        self.run_security_scan()
        
        # Generate report
        all_passed = self.generate_report()
        
        if all_passed:
            print("\n✅ All tests passed!")
        else:
            print("\n❌ Some tests failed!")
        
        return all_passed


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Run Heavy Optimizer tests')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Show detailed test output')
    parser.add_argument('--skip-quality', action='store_true',
                        help='Skip code quality checks')
    parser.add_argument('--skip-performance', action='store_true',
                        help='Skip performance tests')
    parser.add_argument('--skip-gpu', action='store_true',
                        help='Skip GPU tests')
    parser.add_argument('--suite', choices=['unit', 'integration', 'algorithm', 
                                           'performance', 'gpu', 'quality'],
                        help='Run only specific test suite')
    
    args = parser.parse_args()
    
    runner = TestRunner(verbose=args.verbose)
    
    if args.suite:
        # Run specific suite
        suite_methods = {
            'unit': runner.run_unit_tests,
            'integration': runner.run_integration_tests,
            'algorithm': runner.run_algorithm_tests,
            'performance': runner.run_performance_tests,
            'gpu': runner.run_gpu_tests,
            'quality': runner.run_code_quality_checks
        }
        
        success = suite_methods[args.suite]()
        runner.generate_report()
        return 0 if success else 1
    else:
        # Run all suites
        all_passed = runner.run_all(
            skip_quality=args.skip_quality,
            skip_performance=args.skip_performance,
            skip_gpu=args.skip_gpu
        )
        
        return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())