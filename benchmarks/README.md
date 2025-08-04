# Parquet Pipeline Benchmark Framework

A comprehensive performance benchmarking framework for the Heavy Optimizer Platform's Parquet/Arrow/cuDF pipeline, designed to validate the <3s SLA requirement and measure 10× speedup improvements over legacy CSV processing.

## 🎯 Overview

This framework provides:
- **Automated Performance Testing**: End-to-end pipeline benchmarking with multiple scenarios
- **SLA Validation**: Ensures <3s processing time requirement compliance
- **Legacy Comparison**: Measures speedup factors vs traditional CSV processing
- **CI Integration**: Automated regression detection and build failure on SLA violations
- **Comprehensive Reporting**: HTML dashboards, CSV exports, and JSON artifacts

## 📁 Directory Structure

```
benchmarks/
├── config/
│   └── benchmark_config.json          # Test scenarios and thresholds
├── scripts/
│   ├── generate_test_data.py          # Test dataset generator
│   ├── parquet_pipeline_benchmark.py  # Core benchmark engine
│   ├── generate_report.py             # Report generation system
│   └── templates/                     # HTML report templates
├── data/                              # Generated test datasets
├── reports/                           # Generated reports and artifacts
│   └── ci_artifacts/                  # CI-specific outputs
├── run_benchmark.py                   # Main runner script
└── README.md                          # This file
```

## 🚀 Quick Start

### 1. Run All Benchmark Scenarios
```bash
cd /mnt/optimizer_share/benchmarks
python run_benchmark.py
```

### 2. Run Specific Scenario
```bash
python run_benchmark.py --scenario production_baseline
```

### 3. CI Mode (Fail on SLA Violations)
```bash
python run_benchmark.py --ci-mode
```

## 📊 Benchmark Scenarios

The framework includes four predefined scenarios:

| Scenario | Strategies | Trading Days | Target Time | Description |
|----------|------------|--------------|-------------|-------------|
| `micro_dataset` | 500 | 50 | 1000ms | Quick validation dataset |
| `production_baseline` | 2500 | 82 | 2500ms | Representative production workload |
| `large_portfolio` | 10000 | 250 | 3000ms | Large dataset stress test |
| `sla_boundary` | 3000 | 100 | 2900ms | SLA boundary testing |

## 🔧 Configuration

### Benchmark Configuration (`config/benchmark_config.json`)

```json
{
  "performance_thresholds": {
    "max_runtime_seconds": 3.0,
    "min_speedup_factor": 10.0,
    "max_memory_gb": 8.0,
    "min_gpu_utilization": 0.7
  },
  "benchmark_scenarios": [
    {
      "name": "production_baseline",
      "strategy_count": 2500,
      "trading_days": 82,
      "target_time_ms": 2500,
      "description": "Representative production workload"
    }
  ]
}
```

### Environment Variables

- `BENCHMARK_OUTPUT_DIR`: Custom output directory for reports
- `BENCHMARK_TIMEOUT`: Timeout in seconds (default: 1800)
- `FAIL_ON_SLA`: Exit with error on SLA violations (CI use)

## 📈 Performance Metrics

### Pipeline Stage Breakdown
- **CSV to Parquet**: Conversion time with compression analysis
- **Parquet to cuDF**: Loading time with memory optimization
- **Algorithm Processing**: Full optimization workflow execution
- **Total Pipeline**: End-to-end processing time

### System Metrics
- **Memory Usage**: Peak RAM and GPU memory consumption
- **CPU Utilization**: Processing efficiency metrics
- **I/O Performance**: File read/write throughput
- **Compression Ratios**: Storage optimization effectiveness

### SLA Validation
- **Response Time**: <3s requirement for standard datasets
- **Throughput**: Strategies processed per second
- **Reliability**: Success rate across multiple runs
- **Scalability**: Performance consistency across dataset sizes

## 🛠️ CLI Usage

### Main Runner Script

```bash
python run_benchmark.py [OPTIONS]

Options:
  --scenario NAME         Run specific scenario only
  --fail-on-sla          Exit with error code if SLA violations detected
  --no-reports           Skip report generation for faster execution
  --output-dir PATH      Custom output directory
  --verbose, -v          Enable verbose logging
  --ci-mode              Run in CI mode (fail-on-sla + minimal output)
```

### Individual Components

#### Generate Test Data
```bash
cd scripts/
python generate_test_data.py --all-scenarios
python generate_test_data.py --strategies 5000 --days 200 --correlation medium
```

#### Run Benchmarks Only
```bash
cd scripts/
python parquet_pipeline_benchmark.py --scenario production_baseline --fail-on-sla
```

#### Generate Reports
```bash
cd scripts/
python generate_report.py --results ../reports/benchmark_results_20250804_120000.json --format html
```

## 📋 Report Formats

### HTML Dashboard
Interactive dashboard with:
- Performance charts and trend analysis
- SLA compliance visualization
- System resource utilization
- Pipeline stage breakdown
- Historical comparison

### CSV Export
Raw data export for external analysis:
```csv
scenario_name,strategy_count,trading_days,actual_time_ms,target_time_ms,sla_compliance,speedup_factor
production_baseline,2500,82,2100,2500,true,12.5
```

### JSON Artifacts
Machine-readable results for automation:
```json
{
  "scenario_name": "production_baseline",
  "validation": {
    "sla_compliance": true,
    "speedup_factor": 12.5,
    "actual_time_ms": 2100,
    "target_time_ms": 2500
  }
}
```

## 🔄 CI Integration

### GitHub Actions Workflow

The framework includes automated CI integration (`.github/workflows/benchmark.yml`):

```yaml
# Runs on: push, PR, daily schedule, manual trigger
# Features: GPU support, artifact upload, PR comments, SLA enforcement
```

### CI Features
- **Automated Execution**: Triggered on code changes
- **Build Failure**: On SLA violations (configurable)
- **Artifact Storage**: 30-day retention of benchmark results
- **PR Comments**: Inline performance feedback
- **Performance Regression**: Detection vs baseline metrics

### CI Usage Examples

```bash
# Manual trigger with specific scenario
gh workflow run benchmark.yml -f scenario=production_baseline -f fail_on_sla=true

# Check latest benchmark results
gh run list --workflow=benchmark.yml --limit=5
```

## 🎛️ Advanced Usage

### Custom Scenarios

Add new scenarios to `config/benchmark_config.json`:

```json
{
  "name": "custom_scenario",
  "strategy_count": 5000,
  "trading_days": 120,
  "target_time_ms": 2800,
  "description": "Custom test scenario"
}
```

### Performance Tuning

Monitor and optimize using benchmark insights:

```python
# Example: Analyze bottlenecks
results = load_benchmark_results("benchmark_results.json")
bottlenecks = analyze_pipeline_stages(results)
print(f"Slowest stage: {bottlenecks['slowest_stage']}")
```

### Historical Tracking

Store baseline performance for regression detection:

```bash
# Store current results as baseline
cp benchmark_results.json data/performance_baselines.json

# Compare against baseline
python scripts/compare_performance.py --baseline data/performance_baselines.json --current benchmark_results.json
```

## 🐛 Troubleshooting

### Common Issues

1. **"No cuDF available" Warning**
   ```bash
   # Install cuDF for GPU acceleration (optional)
   pip install cudf-cu11 --extra-index-url=https://pypi.nvidia.com
   ```

2. **Memory Limitations**
   ```bash
   # Reduce dataset size or enable chunking
   python run_benchmark.py --scenario micro_dataset
   ```

3. **SLA Violations**
   ```bash
   # Check system resources and GPU availability
   python -c "from scripts.parquet_pipeline_benchmark import *; log_gpu_environment()"
   ```

### Debug Mode

Enable verbose logging for detailed execution traces:

```bash
python run_benchmark.py --verbose --scenario production_baseline
```

### Log Locations

- **Benchmark logs**: Console output with timestamps
- **Error details**: Included in JSON results under `errors` key
- **System info**: Automatically captured in all reports

## 📊 Performance Baselines

### Expected Results (Production Hardware)

| Scenario | Expected Time | Speedup Factor | Memory Usage |
|----------|---------------|----------------|--------------|
| micro_dataset | ~800ms | 8-12x | <1GB |
| production_baseline | ~2000ms | 10-15x | 2-4GB |
| large_portfolio | ~2500ms | 12-20x | 4-6GB |
| sla_boundary | ~2400ms | 10-14x | 3-5GB |

### Hardware Requirements

- **Minimum**: 8GB RAM, 4 CPU cores
- **Recommended**: 16GB RAM, 8 CPU cores, GPU
- **Optimal**: 32GB RAM, 16 CPU cores, NVIDIA GPU with CUDA

## 🔗 Integration with Existing Workflows

### Job Queue Integration

Monitor production performance:

```python
# Add to samba_job_queue_processor.py
from benchmarks.scripts.parquet_pipeline_benchmark import BenchmarkResults
# Collect metrics during job processing
```

### Automated Monitoring

Schedule regular benchmarks:

```bash
# Add to crontab for daily performance monitoring
0 2 * * * cd /mnt/optimizer_share/benchmarks && python run_benchmark.py --ci-mode
```

## 📚 API Reference

### Core Classes

- `ParquetPipelineBenchmark`: Main benchmark executor
- `BenchmarkResults`: Results container with validation
- `BenchmarkDataGenerator`: Test dataset generator
- `BenchmarkReportGenerator`: Report generation engine

### Key Methods

```python
# Run single scenario benchmark
results = benchmark.run_scenario_benchmark(scenario_config)

# Generate comprehensive reports
report_paths = generator.generate_all_reports(results_file)

# Validate SLA compliance
is_compliant = benchmark.check_sla_compliance(results)
```

## 🤝 Contributing

### Adding New Benchmarks

1. Update `benchmark_config.json` with new scenarios
2. Extend `ParquetPipelineBenchmark` class with new test methods
3. Update report templates to include new metrics
4. Add documentation and examples

### Testing Framework Changes

```bash
# Validate benchmark framework
python run_benchmark.py --scenario micro_dataset --verbose

# Test report generation
python scripts/generate_report.py --results test_results.json --format all
```

## 📄 License

Part of the Heavy Optimizer Platform - Internal use only.

## 🔄 Changelog

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-08-04 | Initial benchmark framework implementation |

---

For support and questions, contact the Heavy Optimizer Platform development team.