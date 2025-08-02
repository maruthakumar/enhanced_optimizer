# Epic 1 Story: Parquet/Arrow/cuDF Implementation

## Status
To Do

## Story
**As a** System Architect,
**I want** to replace HeavyDB with a Parquet/Arrow/cuDF data pipeline,
**so that** the platform can scale beyond 32GB limitations and process 100k+ strategies with improved performance

## Acceptance Criteria
1. Reads legacy CSV → writes Parquet with optimized schema and partitioning
2. Loads Parquet as Arrow → cuDF for GPU-accelerated operations
3. Maintains result accuracy within ±0.001% tolerance across all algorithms and metrics

## Tasks / Subtasks

### Task 1: Parquet Data Pipeline Implementation (AC: 1)
- [ ] Subtask 1.1: Create CSV to Parquet converter with schema detection
  - [ ] Auto-detect strategy columns (any column not in mandatory/enhanced lists)
  - [ ] Support both legacy CSV (Date + strategies) and enhanced CSV formats
  - [ ] Implement date partitioning for query optimization
  - [ ] Add compression settings (SNAPPY/GZIP) for storage efficiency
- [ ] Subtask 1.2: Implement enhanced CSV format support
  - [ ] Add support for new columns: start_time, end_time, market_regime, Regime_Confidence_%, capital, zone
  - [ ] Create configuration-driven column inclusion/exclusion
  - [ ] Maintain backward compatibility with existing CSV workflows
  - [ ] Validate data types and handle missing values gracefully
- [ ] Subtask 1.3: Create Parquet storage optimization
  - [ ] Implement column pruning for unused strategy columns
  - [ ] Add row group sizing for optimal GPU memory transfers
  - [ ] Create metadata caching for faster schema access
  - [ ] Implement data validation and integrity checks

### Task 2: Arrow/cuDF GPU Transfer Pipeline (AC: 2)
- [ ] Subtask 2.1: Replace HeavyDB data loading with Arrow/cuDF
  - [ ] Create Arrow memory pools for zero-copy GPU transfers
  - [ ] Implement cuDF DataFrame operations replacing SQL queries
  - [ ] Add GPU memory management and optimization
  - [ ] Handle large datasets exceeding GPU memory via chunking
- [ ] Subtask 2.2: Implement cuDF-based correlation calculations
  - [ ] Replace `calculate_correlations_gpu()` with cuDF operations
  - [ ] Optimize correlation matrix computation for 100k+ strategies
  - [ ] Add memory-efficient correlation algorithms
  - [ ] Implement correlation caching and reuse mechanisms
- [ ] Subtask 2.3: Create cuDF-based fitness calculations
  - [ ] Replace HeavyDB aggregations with cuDF groupby operations
  - [ ] Implement ROI, drawdown, win rate calculations on GPU
  - [ ] Add support for enhanced financial metrics (Sharpe, Sortino, Calmar)
  - [ ] Optimize memory usage for large strategy portfolios

### Task 3: HeavyDB Replacement and Results Validation (AC: 3)
- [ ] Subtask 3.1: Replace all HeavyDB connection points
  - [ ] Remove HeavyDB imports from `csv_only_heavydb_workflow.py`
  - [ ] Replace `lib.heavydb_connector` with Parquet/Arrow/cuDF modules
  - [ ] Update `load_strategy_data()` to use Parquet instead of HeavyDB
  - [ ] Remove SQL query dependencies throughout the codebase
- [ ] Subtask 3.2: Create comprehensive validation framework
  - [ ] Establish reference results from current production data
  - [ ] Execute new Parquet/Arrow/cuDF workflow on same dataset
  - [ ] Validate fitness scores across all 8 algorithms with ±0.001% tolerance
  - [ ] Validate correlation calculations match HeavyDB output exactly
- [ ] Subtask 3.3: Performance validation and optimization
  - [ ] Measure processing speed improvement (target: 10x faster)
  - [ ] Validate memory scaling beyond 32GB HeavyDB limit
  - [ ] Test with 100k+ strategy datasets
  - [ ] Document performance characteristics and bottlenecks

### Task 4: Enhanced Financial Optimization Integration
- [ ] Subtask 4.1: Implement Kelly Criterion position sizing
  - [ ] Add Kelly Criterion calculations for optimal capital allocation
  - [ ] Integrate with market regime confidence scoring
  - [ ] Create position sizing based on win/loss probabilities
  - [ ] Add capital utilization optimization logic
- [ ] Subtask 4.2: Add risk-adjusted return metrics
  - [ ] Implement Sharpe ratio calculations using cuDF
  - [ ] Add Sortino ratio with downside deviation focus
  - [ ] Create Calmar ratio (return/max drawdown) optimization
  - [ ] Implement VaR/CVaR calculations with 95%/99% confidence levels
- [ ] Subtask 4.3: Market regime optimization
  - [ ] Add regime-based strategy filtering using Regime_Confidence_%
  - [ ] Implement dynamic weighting based on confidence scores
  - [ ] Create transition logic for regime uncertainty periods
  - [ ] Add zone-based intraday optimization capabilities

### Task 5: Configuration and Testing Framework
- [ ] Subtask 5.1: Create configuration management system
  - [ ] Add Parquet/Arrow/cuDF settings to production config
  - [ ] Create format detection and column mapping configuration
  - [ ] Implement algorithm-specific GPU memory settings
  - [ ] Add performance monitoring and alerting thresholds
- [ ] Subtask 5.2: Comprehensive testing and validation
  - [ ] Test with production dataset: `Python_Multi_Consolidated_20250726_161921.csv`
  - [ ] Validate all 8 algorithms produce identical results
  - [ ] Test enhanced CSV format with all new columns
  - [ ] Verify scaling to 100k+ strategies without memory issues
- [ ] Subtask 5.3: Migration and deployment preparation
  - [ ] Create migration script from HeavyDB to Parquet format
  - [ ] Update job queue processor for new data pipeline
  - [ ] Prepare rollback procedures in case of issues
  - [ ] Document new architecture and operational procedures

## Dev Notes

### Previous Story Context
This Epic represents a fundamental architecture migration from the HeavyDB-based system implemented in Stories 1.1-1.4 to a modern Parquet/Arrow/cuDF stack. Key context:

**From Story 1.1**: Real algorithm integration now works with HeavyDB
**From Story 1.2**: Legacy system integration provides baseline for validation
**From Story 1.3**: HeavyDB GPU acceleration shows the performance patterns to preserve
**From Story 1.4**: Complete workflow integration demonstrates current data flow

### Data Models
**Current HeavyDB Schema**:
```sql
CREATE TABLE strategies (
    date_col DATE,
    strategy_1 FLOAT,
    strategy_2 FLOAT,
    ...
);
```

**Target Parquet Schema**:
```python
# Legacy format
{
    'Date': 'date',
    'strategy_columns': 'float64[]'  # Auto-detected
}

# Enhanced format
{
    'Date': 'date',
    'start_time': 'datetime64[ns]',
    'end_time': 'datetime64[ns]',
    'market_regime': 'string',
    'Regime_Confidence_%': 'float64',
    'Market_regime_transition_threshold': 'float64',
    'capital': 'float64',
    'zone': 'string',
    'strategy_columns': 'float64[]'  # Auto-detected
}
```

### API Specifications
**New Data Pipeline Interface**:
```python
# Parquet operations
def csv_to_parquet(csv_path: str, parquet_path: str, format_config: Dict) -> bool
def load_parquet_to_cudf(parquet_path: str, columns: List[str] = None) -> cudf.DataFrame
def optimize_parquet_storage(parquet_path: str, partitions: List[str]) -> None

# cuDF operations replacement for HeavyDB
def calculate_correlations_cudf(df: cudf.DataFrame, strategy_cols: List[str]) -> cudf.DataFrame
def calculate_fitness_cudf(df: cudf.DataFrame, portfolio: List[str]) -> Dict[str, float]
def load_strategy_data_cudf(parquet_path: str, date_range: Tuple) -> cudf.DataFrame
```

### Component Specifications
**Parquet Storage Layer**:
- Format: Apache Parquet with SNAPPY compression
- Partitioning: By date for temporal queries
- Schema: Auto-detection with fallback to configuration
- Location: `/data/parquet/strategies/`

**Arrow Memory Management**:
- Memory pools for GPU transfers
- Zero-copy operations where possible
- Chunked processing for large datasets
- Memory monitoring and garbage collection

**cuDF Processing Engine**:
- GPU-accelerated DataFrame operations
- Correlation matrix calculations
- Aggregation and groupby operations
- Mathematical functions for fitness calculations

### File Locations
**New Components**:
- Parquet converter: `/backend/lib/parquet_pipeline/csv_to_parquet.py`
- Arrow interface: `/backend/lib/arrow_connector/memory_manager.py`
- cuDF operations: `/backend/lib/cudf_engine/gpu_calculator.py`
- Configuration: `/config/parquet_arrow_config.ini`

**Modified Components**:
- Main workflow: `/backend/csv_only_heavydb_workflow.py` → `/backend/parquet_cudf_workflow.py`
- Data loading: Replace all `lib.heavydb_connector` imports
- Correlation engine: `/backend/lib/correlation_optimizer.py`

### Testing Requirements
**Baseline Validation**:
- Execute HeavyDB workflow with test dataset
- Record all fitness scores, correlation matrices, execution times
- Use as ground truth for new implementation validation

**Performance Testing**:
- Memory usage: Monitor GPU memory consumption
- Processing speed: Target 10x improvement over HeavyDB
- Scalability: Test with 100k+ strategy datasets
- Accuracy: Ensure ±0.001% fitness score tolerance

**Integration Testing**:
- End-to-end workflow with both CSV formats
- Algorithm execution with new data pipeline
- Output generation and report creation
- Job queue processor integration

### Technical Constraints
**Hardware Requirements**:
- RAPIDS cuDF 24.x installation
- CUDA 12.x compatible GPU drivers
- Sufficient GPU memory for dataset size
- Apache Arrow/Parquet libraries

**Compatibility Requirements**:
- Preserve existing CSV input/output formats [Source: PRD#L21-24]
- Maintain ULTA algorithm logic without modification [Source: Architecture#L42]
- Support both legacy and enhanced CSV formats [Source: PRD#L44-51]
- Ensure result accuracy within ±0.001% tolerance [Source: Architecture#L76-79]

**Migration Requirements**:
- Zero downtime migration capability
- Rollback procedures in case of issues
- Data integrity validation throughout migration
- Preserve all historical optimization results

### Enhanced Features Implementation
**Kelly Criterion Capital Allocation**:
```python
def kelly_position_sizing(win_rate: float, avg_win: float, avg_loss: float) -> float:
    """Calculate optimal position size using Kelly Criterion"""
    kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
    return max(0, min(kelly_fraction, 0.25))  # Cap at 25% for safety
```

**Market Regime Integration**:
```python
def regime_weighted_selection(strategies: cudf.DataFrame, 
                             confidence_threshold: float = 0.7) -> cudf.DataFrame:
    """Filter and weight strategies based on market regime confidence"""
    valid_strategies = strategies[strategies['Regime_Confidence_%'] >= confidence_threshold]
    valid_strategies['regime_weight'] = valid_strategies['Regime_Confidence_%'] / 100.0
    return valid_strategies
```

## Change Log
| Date | Version | Description | Author |
|------|---------|-------------|--------|
| 2025-08-02 | 1.0 | Initial epic story creation for Parquet/Arrow/cuDF migration | PM Agent |

## Dev Agent Record

### Agent Model Used
[To be filled by Dev Agent]

### Debug Log References
[To be updated during implementation]

### Completion Notes List
[To be updated during implementation]

### File List
[To be updated during implementation]

## QA Results

### QA Review Summary
[To be completed by QA Agent after implementation]

### Acceptance Criteria Verification
[To be completed by QA Agent]

### Test Results
[To be completed by QA Agent]

### Code Quality Assessment
[To be completed by QA Agent]

### Issues Found
[To be completed by QA Agent]

### Compliance Check
[To be completed by QA Agent]

### Final QA Verdict
[To be completed by QA Agent]