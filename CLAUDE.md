# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

The Heavy Optimizer Platform is a high-performance portfolio optimization system that processes trading strategy data using GPU-accelerated algorithms. It serves Windows clients via Samba network shares and provides comprehensive financial analysis with professional-grade metrics.

## Core Architecture

### Job Processing Flow
1. Windows clients submit CSV files via `\\204.12.223.93\optimizer_share`
2. Job queue processor (`/backend/samba_job_queue_processor.py`) monitors `/jobs/queue/`
3. Core workflow (`/backend/csv_only_heavydb_workflow.py`) processes optimization
4. Results are written to `/output/run_YYYYMMDD_HHMMSS/`
5. Clients access results via Samba share

### Key Components
- **8 Optimization Algorithms** in `/backend/algorithms/`: GA, PSO, SA, DE, ACO, HC, BO, RS
- **ULTA Strategy Inversion**: Automatically optimizes poor-performing strategies
- **Correlation-Based Diversification**: Portfolio risk management through correlation analysis
- **GPU Acceleration**: Optional HeavyDB/CuPy for 2-5x performance improvement

## Common Commands

### Running the System
```bash
# Start job queue processor (server-side daemon)
cd /mnt/optimizer_share/backend
python3 samba_job_queue_processor.py

# Monitor job processor logs
tail -f /mnt/optimizer_share/logs/job_processor_*.log

# Test CSV workflow directly
python3 csv_only_heavydb_workflow.py --input ../input/SENSEX_test_dataset.csv --portfolio-size 35

# Run comprehensive A100 GPU testing
python3 bin/a100_comprehensive_testing.py
```

### Debugging Commands
```bash
# Check job queue status
ls -la /mnt/optimizer_share/jobs/queue/
ls -la /mnt/optimizer_share/jobs/processing/

# View recent error logs
tail -n 100 /mnt/optimizer_share/logs/job_processor_*.log | grep ERROR

# Check output directory for specific run
ls -la /mnt/optimizer_share/output/run_*/

# Find stuck jobs (older than 1 hour)
find /mnt/optimizer_share/jobs/processing/ -mmin +60 -type f

# Check system health
ps aux | grep -E "samba_job_queue_processor|heavydb_workflow"
```

## Key File Locations

### Core Processing
- Main workflow: `/backend/csv_only_heavydb_workflow.py`
- Job processor: `/backend/samba_job_queue_processor.py`
- Algorithms: `/backend/algorithms/*.py`
- Configuration: `/config/production_config.ini`, `/backend/config/algorithm_timeouts.json`

### Input/Output
- Windows launcher: `/input/Samba_Only_HeavyDB_Launcher.bat`
- Test dataset: `input/Python_Multi_Consolidated_20250726_161921.csv`
- Results: `/output/run_YYYYMMDD_HHMMSS/`
- Job queues: `/jobs/{queue,processing,completed,failed}/`

### Documentation
- Architecture: `/docs/Complete_Financial_Architecture.md`
- User guide: `/docs/Complete_Workflow_Guide.md`
- Backend status: `/backend/README.md`

## Financial Metrics

The system calculates 6 professional metrics:
1. **ROI/Drawdown Ratio**: Primary risk-adjusted return metric
2. **Total ROI**: Absolute return analysis
3. **Maximum Drawdown**: Risk assessment (-418,710 to -608,217 typical range)
4. **Win Rate**: Success rate (60.2% to 65.9% typical)
5. **Profit Factor**: Profitability ratio (1.131 to 1.389 typical)
6. **Drawdown Minimization**: Risk control optimization

## Development Notes

### Dependencies
- Python 3.10+ with NumPy, Pandas, Matplotlib
- Optional: HeavyDB, CuPy for GPU acceleration
- No formal package management (no requirements.txt)
- No formal testing framework (no pytest, unittest)

### Testing
```bash
# Test with sample dataset
cd /mnt/optimizer_share/backend
python3 csv_only_heavydb_workflow.py --input ../input/SENSEX_test_dataset.csv --portfolio-size 10

# Test specific algorithm
python3 -m algorithms.genetic_algorithm

# Run A100 GPU performance tests
python3 bin/a100_comprehensive_testing.py
```

### Performance Benchmarks
- Processing speed: 14,388 strategies/second
- Memory usage: ~175MB peak
- Algorithm execution: 0.025s average
- Full workflow: 1.22-1.78s for test dataset
- Production data: 25,544 strategies � 82 trading days

### Adding New Features

#### New Algorithm
1. Create file in `/backend/algorithms/`
2. Implement `optimize(data, portfolio_size, metrics)` method
3. Add to algorithm registry in workflow files
4. Update `/backend/config/algorithm_timeouts.json`

#### New Financial Metric
1. Add calculation to workflow metric functions
2. Update output generation for new metric
3. Modify visualization code if needed
4. Document metric interpretation

## Troubleshooting

### Common Issues
- **Job stuck in queue**: Ensure job processor is running
- **Network drive access**: Check credentials (opt_admin/Chetti@123)
- **Memory errors**: Reduce portfolio size or dataset
- **Missing outputs**: Check error logs in output directory

### Error Patterns
```bash
# Search for common errors
grep -E "MemoryError|TimeoutError|FileNotFoundError|PermissionError" /mnt/optimizer_share/logs/job_processor_*.log

# Check for algorithm failures
grep "Algorithm.*failed" /mnt/optimizer_share/logs/job_processor_*.log

# Find incomplete runs
find /mnt/optimizer_share/output/run_* -name "error_log.txt" -exec grep -l "ERROR" {} \;
```

### Log Locations
- Job processor: `/logs/job_processor_*.log`
- Error logs: `/output/run_*/error_log.txt`
- System logs: `/logs/`

## Windows Client Access
- Network path: `\\204.12.223.93\optimizer_share`
- Credentials: Username `opt_admin`, Password `Chetti@123`
- Launcher: `input\Samba_Only_HeavyDB_Launcher.bat`

## Important Configuration Files

### Production Config (`/config/production_config.ini`)
Contains settings for:
- Algorithm parameters
- Portfolio constraints
- Risk thresholds
- Output preferences

### Algorithm Timeouts (`/backend/config/algorithm_timeouts.json`)
Defines maximum execution time for each algorithm

### Monitoring Config (`/backend/config/monitoring_config.json`)
Contains performance tracking and alerting thresholds

## System Maintenance

### Cleanup Old Results
```bash
# Remove output directories older than 30 days
find /mnt/optimizer_share/output/ -name "run_*" -type d -mtime +30 -exec rm -rf {} \;

# Clean failed jobs older than 7 days
find /mnt/optimizer_share/jobs/failed/ -type f -mtime +7 -delete

# Archive completed jobs
tar -czf /mnt/optimizer_share/archive/jobs_$(date +%Y%m%d).tar.gz /mnt/optimizer_share/jobs/completed/
```

### Monitor System Health
```bash
# Check disk usage
df -h /mnt/optimizer_share

# Monitor memory usage during processing
watch -n 1 'free -h'

# Check active Python processes
ps aux | grep python3 | grep -v grep
```

## Notes

- System optimized for CSV input only (Excel dependencies removed)
- All processing happens server-side with results accessible via Samba
- Portfolio sizes typically range from 10-100 strategies
- Real production data includes 25,544+ strategies across 82 trading days
- No automated tests - use sample datasets for validation
- No CI/CD pipeline - manual deployment only