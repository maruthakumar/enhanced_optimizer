#!/usr/bin/env python3
"""
Test Runner for Configuration Management Tests

Runs all configuration-related tests with proper output formatting.
"""

import unittest
import sys
import os
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent.parent))

def run_config_tests():
    """Run all configuration management tests"""
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all config tests
    from test_config_manager import (
        TestConfigurationManager,
        TestConfigValidation,
        TestConfigurationIntegration
    )
    
    test_classes = [
        TestConfigurationManager,
        TestConfigValidation,
        TestConfigurationIntegration
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    print("CONFIGURATION MANAGEMENT TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    if result.wasSuccessful():
        print("\n✅ All configuration tests passed!")
        return 0
    else:
        print("\n❌ Some tests failed. See details above.")
        return 1


if __name__ == '__main__':
    sys.exit(run_config_tests())