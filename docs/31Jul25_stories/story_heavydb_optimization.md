# Story: HeavyDB Table Schema Optimization

**Status: ✅ COMPLETED**

**As a** Database Architect,
**I want to** optimize the HeavyDB table schema for analytical query performance,
**So that** all subsequent operations (ULTA, Correlation, Algorithms) run as fast as possible.

**Production Data Specifications**:
- **Input File**: `input/Python_Multi_Consolidated_20250726_161921.csv`
- **File Size**: 39.2 MB (39,201,813 bytes)
- **Trading Days**: 82 days (January 2024 - July 2024)
- **Strategies**: 25,544 unique SENSEX trading strategies
- **Data Points**: 2,094,608 total data points (82 × 25,544)
- **Strategy Types**: Various SENSEX configurations with Stop Loss (SL) and Take Profit (TP) parameters
- **Date Range**: 2024-01-04 to 2024-07-26
- **Column Structure**: Date, Day, followed by 25,544 strategy columns with numerical P&L values

**Acceptance Criteria**:

1.  **Columnar Storage**: Ensure the table is created using HeavyDB's optimal columnar storage format for 25,544 strategy columns.
2.  **Partitioning**: Implement date-based partitioning strategy on 82 trading days to improve time-series query performance.
3.  **Data Encoding**: Analyze and apply optimal encoding for:
    - Date columns (DATE type)
    - Strategy name columns (TEXT with dictionary encoding)
    - P&L value columns (FLOAT/DOUBLE with appropriate precision)
4.  **Index Strategy**: Evaluate indexing for:
    - Date column for temporal queries
    - High-volume strategy columns for portfolio optimization
5.  **Performance Validation**: Benchmark against production data:
    - Load time for 39.2 MB file
    - Query performance for correlation analysis across 25,544 strategies
    - Memory usage optimization for large dataset operations
    - ULTA analysis performance on actual strategy data

**Production Benchmarks**:
- **Expected Load Time**: < 30 seconds for 39.2 MB
- **Memory Usage**: < 2 GB for full dataset
- **Query Performance**: < 5 seconds for correlation matrix of top 1000 strategies
- **Optimization Performance**: < 300 seconds for full portfolio optimization

**Technical Notes**:
- This story directly addresses the "Columnar Storage Optimized for analytics" block in the architecture diagram.
- This should be implemented as part of the `Dynamic Table Creation` process.
- Must handle SENSEX strategy naming conventions and P&L value ranges in production data.
