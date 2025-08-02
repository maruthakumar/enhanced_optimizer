#!/usr/bin/env python3
"""
QA Test Script for ACO Algorithm and Pandas Warning Fixes
"""

import subprocess
import sys
import re

def test_aco_algorithm():
    """Test that ACO algorithm runs without probability errors"""
    print("=" * 60)
    print("TEST 1: ACO Algorithm Probability Fix")
    print("=" * 60)
    
    # Run the optimizer
    cmd = [sys.executable, "csv_only_heavydb_workflow.py", "--test", "--portfolio-size", "5"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Check for ACO success
    aco_success = False
    aco_error = False
    
    for line in result.stdout.split('\n'):
        if "ACO: Starting real execution" in line:
            print(f"✓ Found: {line}")
        if "✅ ACO:" in line and "Fitness=" in line:
            aco_success = True
            print(f"✓ ACO Success: {line}")
        if "❌ ACO:" in line or "ACO failed" in line:
            aco_error = True
            print(f"✗ ACO Error: {line}")
    
    # Check stderr for probability errors
    if "probabilities are not non-negative" in result.stderr:
        aco_error = True
        print("✗ ACO Error: Probability calculation failed")
    
    if aco_success and not aco_error:
        print("✅ PASS: ACO algorithm runs successfully without probability errors")
        return True
    else:
        print("❌ FAIL: ACO algorithm still has issues")
        return False

def test_pandas_warning():
    """Test that pandas deprecation warning is fixed"""
    print("\n" + "=" * 60)
    print("TEST 2: Pandas Deprecation Warning Fix")
    print("=" * 60)
    
    # Run with all warnings enabled
    cmd = [sys.executable, "-W", "all", "csv_only_heavydb_workflow.py", "--test", "--portfolio-size", "3"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Check for infer_datetime_format warning
    warning_found = False
    
    warning_patterns = [
        "infer_datetime_format",
        "FutureWarning.*infer_datetime_format",
        "deprecated.*infer_datetime_format"
    ]
    
    for pattern in warning_patterns:
        if re.search(pattern, result.stderr, re.IGNORECASE):
            warning_found = True
            print(f"✗ Found warning pattern: {pattern}")
            break
    
    if not warning_found:
        print("✅ PASS: No pandas infer_datetime_format deprecation warning found")
        return True
    else:
        print("❌ FAIL: Pandas deprecation warning still present")
        print("Stderr output:", result.stderr[:500])
        return False

def test_multiple_runs():
    """Test stability over multiple runs"""
    print("\n" + "=" * 60)
    print("TEST 3: Stability Test (3 runs)")
    print("=" * 60)
    
    success_count = 0
    
    for i in range(3):
        print(f"\nRun {i+1}/3:")
        cmd = [sys.executable, "csv_only_heavydb_workflow.py", "--test", "--portfolio-size", "5"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Count successful algorithms
        success_match = re.search(r"Successful Algorithms: (\d+)/8", result.stdout)
        if success_match:
            num_success = int(success_match.group(1))
            print(f"  Successful algorithms: {num_success}/8")
            if num_success == 8:
                success_count += 1
    
    if success_count == 3:
        print(f"\n✅ PASS: All {success_count} runs completed with 8/8 algorithms successful")
        return True
    else:
        print(f"\n⚠️  PARTIAL: {success_count}/3 runs had all algorithms successful")
        return success_count > 0

def main():
    """Run all QA tests"""
    print("QA TEST SUITE: ACO Algorithm & Pandas Warning Fixes")
    print("=" * 60)
    
    tests = [
        ("ACO Algorithm Fix", test_aco_algorithm),
        ("Pandas Warning Fix", test_pandas_warning),
        ("Stability Test", test_multiple_runs)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            print(f"\n❌ ERROR in {test_name}: {str(e)}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("QA TEST SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ ALL TESTS PASSED - Fixes verified successfully!")
    else:
        print("❌ SOME TESTS FAILED - Please review the issues above")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())