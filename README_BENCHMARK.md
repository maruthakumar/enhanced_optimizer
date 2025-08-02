# Legacy vs HeavyDB Benchmark Suite

## Overview

This benchmark suite provides comprehensive performance and accuracy validation between the legacy zone optimization system and the new HeavyDB-accelerated Heavy Optimizer Platform.

## Quick Start

### Run Complete Benchmark Suite
```bash
cd /mnt/optimizer_share/backend
python3 legacy_vs_heavydb_benchmark.py
```

### Run Specific Test Case
```bash
# Micro test (500 strategies) - fastest validation
python3 legacy_vs_heavydb_benchmark.py --test-case micro

# Full production test (25,544 strategies) - complete validation  
python3 legacy_vs_heavydb_benchmark.py --test-case full
```

## Test Scale Matrix

| Test Scale | Strategies | Dataset % | Purpose | Target Time |
|------------|------------|-----------|---------|-------------|
| **Micro** | 500 | 2% | Algorithm accuracy | < 10s |
| **Small** | 2,500 | 10% | Performance baseline | < 60s |
| **Medium** | 5,000 | 20% | Scaling validation | < 180s |
| **Large** | 12,500 | 50% | Memory optimization | < 600s |
| **Full** | 25,544 | 100% | Production readiness | < 1200s |

## Output Structure

```
/output/benchmark_YYYYMMDD_HHMMSS/
├── executive_summary.md           # Management overview
├── performance_analysis.md        # Detailed performance metrics
├── accuracy_validation.md         # Mathematical consistency
├── resource_analysis.md           # Memory/CPU analysis  
├── benchmark_visualizations.png   # 9-panel chart suite
├── benchmark_report.html          # Combined HTML report
├── detailed_benchmark_data.json   # Complete raw data
└── test_data_*.csv                # Generated test datasets
```

## Key Validation Criteria

### Performance Requirements
- **Target**: ≥ 2x speedup vs legacy system
- **Memory**: < 4GB peak usage
- **Accuracy**: Within 5% of legacy results
- **Success Rate**: ≥ 80% tests must pass

### Automated Validation
- ✅ **Performance**: Execution time comparison
- ✅ **Accuracy**: Mathematical result validation  
- ✅ **Memory**: Resource constraint checking
- ✅ **Quality**: Portfolio optimization validation

## Core Components

### 1. Main Benchmark (`legacy_vs_heavydb_benchmark.py`)
- Comprehensive test suite execution
- Real-time resource monitoring
- Automated result validation
- Dataset sampling and generation

### 2. Monitoring Utils (`benchmark_utils.py`)  
- Resource tracking (memory, CPU, I/O)
- High-precision timing
- Data processing utilities
- Result validation framework

### 3. Report Generator (`benchmark_report_generator.py`)
- Executive summary generation
- Performance analysis charts
- Resource utilization reports
- Multi-format output (MD, HTML, JSON, PNG)

## Production Migration Criteria

The benchmark suite provides **go/no-go** recommendations based on:

1. **Performance**: ≥ 2x improvement across all scales
2. **Accuracy**: ≥ 80% mathematical consistency 
3. **Resources**: 100% memory constraint compliance
4. **Quality**: Valid portfolio optimization results

## Example Results Interpretation

### ✅ Production Ready
```
Overall Performance Improvement: 3.2x
Accuracy Validations Passed: 5/5
Memory Efficiency Tests Passed: 5/5
Status: ✅ READY FOR PRODUCTION
```

### ⚠️ Needs Optimization
```
Overall Performance Improvement: 1.8x  
Accuracy Validations Passed: 4/5
Memory Efficiency Tests Passed: 3/5
Status: ❌ REQUIRES OPTIMIZATION
```

## Troubleshooting

### Common Issues

**Dataset Not Found**
```bash
# Ensure test dataset exists
ls -la /mnt/optimizer_share/input/Python_Multi_Consolidated_20250726_161921.csv
```

**Memory Errors**
```bash
# Run smaller test first
python3 legacy_vs_heavydb_benchmark.py --test-case micro
```

**Permission Issues**
```bash
# Check output directory permissions
mkdir -p /mnt/optimizer_share/output/
chmod 755 /mnt/optimizer_share/output/
```

### Monitoring During Execution
```bash
# Monitor system resources
htop

# Watch memory usage
watch -n 1 'free -h'

# Check benchmark logs
tail -f /mnt/optimizer_share/output/benchmark_*/benchmark_report.html
```

## Advanced Usage

### Custom Test Configuration
Modify test cases in `legacy_vs_heavydb_benchmark.py`:
```python
self.test_cases = [
    TestCase("custom", 1000, 5.0, "Custom validation", 30),
    # Add more test cases as needed
]
```

### Integration with CI/CD
```bash
#!/bin/bash
# Automated benchmark execution
cd /mnt/optimizer_share/backend
python3 legacy_vs_heavydb_benchmark.py > benchmark_results.log 2>&1

# Check exit code for pass/fail
if [ $? -eq 0 ]; then
    echo "✅ Benchmark PASSED - Ready for deployment"
else  
    echo "❌ Benchmark FAILED - Requires optimization"
    exit 1
fi
```

## Dependencies

- Python 3.10+
- NumPy, Pandas, Matplotlib
- psutil (resource monitoring)
- seaborn (visualizations)

## Related Documentation

- **Story Implementation**: `/docs/stories/story_legacy_vs_heavydb_benchmark.md`
- **Architecture**: `/docs/Complete_Financial_Architecture.md`  
- **Workflow Guide**: `/docs/Complete_Workflow_Guide.md`
- **Backend Status**: `/backend/README.md`