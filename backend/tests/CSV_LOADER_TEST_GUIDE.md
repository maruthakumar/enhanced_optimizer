# CSV Loader Test Guide

## Overview

This guide covers the comprehensive test suite for the Enterprise CSV Loader component of the Heavy Optimizer Platform. The tests ensure that all CSV loading features work correctly including streaming, validation, progress tracking, batch inserts, and GPU memory management.

## Test Structure

### Test Files
- `test_csv_loader.py` - Main test suite with all test classes
- `run_csv_loader_tests.py` - Test runner script

### Test Classes

#### 1. TestEnterpriseCSVLoader
Tests core CSV loading functionality:
- Basic streaming load
- Progress tracking
- Data validation
- Missing value handling
- Checkpoint/resume functionality
- Audit trail
- Memory monitoring
- Throughput calculation

#### 2. TestBatchInsertHandler
Tests batch insert operations:
- Basic batch insert
- DataFrame batch insert with progress
- Batch size optimization
- Asynchronous batch processing

#### 3. TestGPUMemoryManager
Tests GPU memory management:
- Data chunking for GPU
- GPU transfer simulation

#### 4. TestEdgeCases
Tests edge cases and error handling:
- Empty CSV files
- Single column CSV
- Large column count (1000+ columns)
- Special characters in data
- Concurrent checkpoint access
- Corrupted checkpoint recovery

#### 5. TestPerformance
Tests performance characteristics:
- Load performance scaling
- Memory efficiency with large files

#### 6. TestIntegration
Tests full CSV loading pipeline integration

#### 7. TestDALIntegration
Tests Enhanced CSV DAL integration:
- CSV loading through DAL with progress
- Batch insert via DAL
- Load metrics retrieval
- ULTA transformation
- Correlation matrix computation
- Table information retrieval

## Running Tests

### Run All CSV Loader Tests
```bash
cd /mnt/optimizer_share/backend/tests
python3 run_csv_loader_tests.py
```

### Run Specific Test Class
```bash
python3 -m unittest test_csv_loader.TestEnterpriseCSVLoader
```

### Run Specific Test Method
```bash
python3 -m unittest test_csv_loader.TestEnterpriseCSVLoader.test_basic_streaming_load
```

### Run with Coverage
```bash
coverage run -m unittest test_csv_loader
coverage report -m
```

## Test Data

The tests automatically create temporary test data including:
- Small CSV files (500 rows)
- Medium CSV files (5,000 rows)  
- Large CSV files (10,000+ rows)
- Invalid CSV files for error testing
- CSV files with special characters
- CSV files with missing values

All test data is cleaned up automatically after tests complete.

## Key Test Scenarios

### 1. Streaming Large Files
- Verifies chunked reading works correctly
- Ensures memory usage stays within bounds
- Tests progress tracking accuracy

### 2. Data Validation
- Tests numeric column detection
- Validates error handling for invalid data
- Checks missing value detection

### 3. Checkpoint/Resume
- Simulates interrupted loads
- Verifies checkpoint creation
- Tests successful resume from checkpoint
- Ensures checkpoint cleanup after success

### 4. Batch Processing
- Tests efficient batch inserts
- Verifies progress tracking during batches
- Tests async batch queue processing

### 5. GPU Integration
- Tests GPU memory monitoring (when available)
- Verifies data chunking for GPU transfers
- Tests graceful fallback when GPU not available

### 6. DAL Integration
- Tests complete workflow through Enhanced CSV DAL
- Verifies all DAL methods work with enterprise loader
- Tests ULTA transformation and correlation matrix

## Expected Test Output

Successful test run should show:
```
test_basic_streaming_load (test_csv_loader.TestEnterpriseCSVLoader) ... ok
test_progress_tracking (test_csv_loader.TestEnterpriseCSVLoader) ... ok
...
======================================================================
CSV LOADER TEST SUMMARY
======================================================================
Tests run: 43
Failures: 0
Errors: 0
Skipped: 0

✅ All CSV loader tests passed!
```

## Troubleshooting

### Common Issues

1. **Import Errors**
   - Ensure you're in the correct directory
   - Check that backend is in Python path

2. **File Permission Errors**
   - Ensure temp directories are writable
   - Check audit log path permissions

3. **Memory Errors**
   - Reduce test data sizes
   - Increase system memory

4. **GPU Tests Failing**
   - GPU tests gracefully skip if GPU not available
   - Install GPUtil for GPU monitoring: `pip install gputil`

### Debug Mode
Run tests with more verbose output:
```bash
python3 -m unittest test_csv_loader -v
```

## Extending Tests

To add new tests:

1. Add test method to appropriate test class
2. Follow naming convention: `test_feature_description`
3. Include docstring describing what is tested
4. Use setUp/tearDown for test fixtures
5. Clean up any created files/resources

Example:
```python
def test_new_feature(self):
    """Test description of new feature"""
    # Arrange
    test_data = self._create_test_csv('test.csv', 100, 5)
    
    # Act
    result = self.loader.new_feature(test_data)
    
    # Assert
    self.assertTrue(result.success)
    self.assertEqual(result.value, expected_value)
```

## Performance Benchmarks

Expected performance on test system:
- Small files (< 10MB): > 50 MB/s throughput
- Medium files (10-100MB): > 30 MB/s throughput  
- Large files (> 100MB): > 20 MB/s throughput

Memory usage should stay under 2x file size for streaming efficiency.