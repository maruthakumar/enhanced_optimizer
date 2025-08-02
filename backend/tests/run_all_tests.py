#!/usr/bin/env python3
"""
Test Runner for Algorithm Test Suite

This script runs all algorithm tests and generates a comprehensive report.
"""

import unittest
import sys
import time
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent.parent))


def run_test_suite():
    """Run the complete test suite"""
    print("="*80)
    print("Heavy Optimizer Platform - Algorithm Test Suite")
    print("="*80)
    print()
    
    # Create test loader
    loader = unittest.TestLoader()
    
    # Discover all tests
    start_dir = Path(__file__).parent
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    # Run tests with verbosity
    runner = unittest.TextTestRunner(verbosity=2)
    
    start_time = time.time()
    result = runner.run(suite)
    end_time = time.time()
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    print(f"Total tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}")
    
    # Return exit code
    return 0 if result.wasSuccessful() else 1


def run_specific_test_module(module_name):
    """Run tests from a specific module"""
    print(f"Running tests from {module_name}...")
    
    loader = unittest.TestLoader()
    
    try:
        # Import the module
        module = __import__(f'tests.{module_name}', fromlist=[''])
        suite = loader.loadTestsFromModule(module)
        
        # Run tests
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        return 0 if result.wasSuccessful() else 1
        
    except ImportError as e:
        print(f"Error: Could not import {module_name}: {e}")
        return 1


def list_available_tests():
    """List all available test modules"""
    print("Available test modules:")
    print("-" * 40)
    
    test_dir = Path(__file__).parent
    test_files = sorted(test_dir.glob('test_*.py'))
    
    for test_file in test_files:
        module_name = test_file.stem
        print(f"  {module_name}")
    
    print("\nUsage:")
    print("  python run_all_tests.py              # Run all tests")
    print("  python run_all_tests.py <module>     # Run specific module")
    print("  python run_all_tests.py --list       # List available modules")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        if sys.argv[1] == '--list':
            list_available_tests()
            sys.exit(0)
        else:
            # Run specific test module
            module_name = sys.argv[1]
            if not module_name.startswith('test_'):
                module_name = f'test_{module_name}'
            exit_code = run_specific_test_module(module_name)
    else:
        # Run all tests
        exit_code = run_test_suite()
    
    sys.exit(exit_code)