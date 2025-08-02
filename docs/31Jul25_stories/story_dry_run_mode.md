# Story: Dry Run Mode for Testing

**Status: 📋 PLANNED**

**As a** Developer,
**I want to** add a "dry run" mode to the Pipeline Orchestrator,
**So that** I can test the data flow and configuration of the pipeline without actually performing the expensive computations.

**Production Data Context**:
- **Target Dataset**: `input/Python_Multi_Consolidated_20250726_161921.csv`
- **File Size**: 39.2 MB (39,201,813 bytes)
- **Strategy Count**: 25,544 SENSEX trading strategies
- **Trading Days**: 82 days (January 4 - July 26, 2024)
- **Data Volume**: 2,094,608 P&L data points
- **Processing Time**: Full execution ~300 seconds, dry run target <5 seconds

**Acceptance Criteria**:

1. **Pipeline Orchestrator Dry Run Support**:
   - `start_optimization_job()` accepts `dry_run=True` parameter
   - Simulates processing of 39.2 MB production file without loading
   - Validates CSV structure and 25,544-column format
   - Estimates memory requirements for full production dataset
   - Logs expected processing steps for 82-day analysis

2. **Data Flow Simulation**:
   - **CSV Loading**: Simulate reading 39.2 MB file, validate headers for 25,544 strategies
   - **Data Preprocessing**: Mock transformation of 2,094,608 data points
   - **Correlation Analysis**: Estimate 25,544×25,544 correlation matrix operations
   - **Algorithm Execution**: Simulate 8 algorithms on production-scale data
   - **Output Generation**: Mock creation of all output formats

3. **Production Data Validation**:
   - Verify SENSEX strategy naming convention compatibility
   - Validate date range processing (January-July 2024)
   - Check Stop Loss/Take Profit parameter parsing
   - Confirm P&L data type and range expectations
   - Validate production file structure integrity

4. **Resource Estimation**:
   - **Memory Requirements**: Estimate peak usage for 25,544 strategies
   - **Processing Time**: Project execution time for production dataset
   - **Disk Space**: Calculate output storage requirements
   - **Network Bandwidth**: Estimate Samba transfer requirements

5. **Configuration Testing**:
   - Test portfolio size configurations (10-100 strategies)
   - Validate algorithm selection for production workload
   - Test correlation threshold settings
   - Verify output format configurations

6. **Error Scenario Simulation**:
   - Simulate corrupted production data scenarios
   - Test memory overflow conditions with 25,544 strategies
   - Mock network interruption during large file processing
   - Simulate disk space exhaustion scenarios

**Dry Run Output Requirements**:
- **Execution Plan**: Detailed steps for processing 25,544 strategies
- **Resource Estimates**: Memory, CPU, disk requirements
- **Time Projections**: Expected duration for each pipeline stage
- **Validation Results**: Production data compatibility check
- **Configuration Summary**: Applied settings and parameters
- **Risk Assessment**: Potential bottlenecks and failure points

**Performance Targets**:
- **Dry Run Execution**: < 5 seconds for complete simulation
- **Memory Usage**: < 100 MB during dry run mode
- **Validation Speed**: < 2 seconds for production file structure check
- **Logging Overhead**: Minimal impact on simulation performance

**Production-Specific Validations**:
1. **SENSEX Strategy Format**: Verify "SENSEX [range] SL[%]TP[%]" pattern compatibility
2. **Date Format**: Validate 2024-01-04 to 2024-07-26 date parsing
3. **Numerical Data**: Check P&L value ranges and formats
4. **File Structure**: Confirm 82 rows × 25,544 columns layout
5. **Memory Scaling**: Estimate requirements for full dataset

**Unit Test Coverage**:
- Dry run mode parameter handling
- Production data structure validation
- Resource estimation accuracy
- Error scenario simulation
- Configuration validation
- Output format simulation

**Integration with Production Workflow**:
- Compatible with `csv_only_heavydb_workflow.py`
- Supports Samba job queue processor dry run
- Integrates with performance monitoring (simulation mode)
- Compatible with all 8 optimization algorithms

**Technical Notes**:
- Dry run must process actual production file metadata
- Resource estimates must reflect 25,544-strategy complexity
- Simulation must account for SENSEX strategy characteristics
- Performance projections must be based on production data patterns
