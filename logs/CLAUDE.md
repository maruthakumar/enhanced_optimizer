# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

The Heavy Optimizer Platform is a portfolio optimization system that uses multiple algorithms with GPU acceleration for financial strategy optimization. It processes CSV data containing trading strategies and produces optimized portfolio selections.

## Key Architecture Components

### 1. Client-Server Architecture
- **Windows Clients**: Access via Samba share `\\204.12.223.93\optimizer_share`
- **Server Processing**: Linux server with HeavyDB GPU acceleration
- **Job Queue System**: Asynchronous job processing via Samba-based queue

### 2. Main Entry Points
- **Windows Users**: `/input/Samba_Only_HeavyDB_Launcher.bat` - Main launcher for portfolio optimization
- **Server Processing**: `/backend/samba_job_queue_processor.py` - Job queue daemon
- **Core Workflow**: `/backend/csv_only_heavydb_workflow.py` - Main optimization logic

### 3. Optimization Algorithms (8 total)
Located in `/backend/algorithms/`:
- Genetic Algorithm (GA)
- Particle Swarm Optimization (PSO)
- Simulated Annealing (SA)
- Differential Evolution (DE)
- Ant Colony Optimization (ACO)
- Hill Climbing (HC)
- Bayesian Optimization (BO)
- Random Search (RS)

### 4. Key Features
- **Financial Metrics**: ROI/Drawdown Ratio, Win Rate, Profit Factor, Max Drawdown
- **ULTA Strategy Inversion**: Automatic optimization of poor-performing strategies
- **Correlation-Based Diversification**: Portfolio risk management
- **GPU Acceleration**: HeavyDB/CuPy for high-performance computing

## Common Development Commands

### Running the System
```bash
# Start job queue processor (server-side)
cd /mnt/optimizer_share/backend
python3 samba_job_queue_processor.py

# Monitor logs
tail -f /mnt/optimizer_share/logs/job_processor_*.log
```

### Testing Individual Components
```bash
# Test CSV workflow
cd /mnt/optimizer_share/backend
python3 csv_only_heavydb_workflow.py --input ../input/SENSEX_test_dataset.csv --portfolio-size 35

# Test specific algorithm
python3 -m algorithms.genetic_algorithm
```

### Windows Client Access
- Map network drive to `\\204.12.223.93\optimizer_share`
- Username: `opt_admin`
- Password: `Chetti@123`
- Run: `input\Samba_Only_HeavyDB_Launcher.bat`

## Important File Locations

### Input/Output
- Input CSV files: `/input/`
- Output results: `/output/run_YYYYMMDD_HHMMSS/`
- Job queue: `/jobs/queue/`, `/jobs/processing/`, `/jobs/completed/`, `/jobs/failed/`

### Configuration
- Production config: `/config/production_config.ini`
- Algorithm timeouts: `/config/algorithm_timeouts.json`

### Documentation
- Architecture details: `/docs/Complete_Financial_Architecture.md`
- User guide: `/docs/Complete_Workflow_Guide.md`

## Development Guidelines

### Adding New Algorithms
1. Create new file in `/backend/algorithms/`
2. Implement standard interface with `optimize()` method
3. Add to algorithm registry in workflow files
4. Update timeout configuration if needed

### Modifying Financial Metrics
- Core metric calculations in workflow files
- Ensure compatibility with output generation engine
- Update visualization code for new metrics

### Performance Considerations
- Use NumPy/Pandas vectorized operations
- Leverage HeavyDB acceleration when available
- Monitor memory usage with large datasets (25,000+ strategies)

## System Dependencies
- Python 3.x with NumPy, Pandas, Matplotlib
- HeavyDB for GPU acceleration (optional but recommended)
- Samba for Windows client access
- Job queue system for asynchronous processing

## Troubleshooting

### Common Issues
1. **Network drive mapping fails**: Check credentials and network connectivity
2. **Job stuck in queue**: Verify job processor is running
3. **Out of memory**: Reduce portfolio size or use smaller dataset
4. **Missing output files**: Check `/output/run_*/error_log.txt`

### Log Locations
- System logs: `/logs/`
- Job processor logs: `/logs/job_processor_*.log`
- Error logs: `/output/run_*/error_log.txt`

## Notes
- System optimized for CSV input only (Excel dependencies removed)
- All processing happens server-side with results accessible via Samba
- Portfolio sizes typically range from 10-100 strategies
- Real production data includes 25,544+ strategies across 82 trading days