# 3. Technical Architecture & Integration Strategy

## Migration Architecture
- **Legacy**: CSV → HeavyDB → GPU operations (32GB limit)
- **Target**: CSV → Parquet → Arrow → cuDF → Parallel workers (unlimited)

## Enhanced Integration Pipeline

The complete data processing pipeline implements the following stages:

1. **CSV → Parquet**: Enhanced CSV (with regime/timing columns) converts to compressed columnar storage
2. **Parquet → Arrow**: Zero-copy memory mapping for efficient GPU transfer
3. **Arrow → GPU**: cuDF DataFrames loaded into GPU memory
4. **Parallel Workers**: Multi-process optimization execution layer
5. **ULTA Logic**: Strategy inversion algorithm (preserving existing implementation)
6. **Correlation Matrix**: GPU-accelerated pairwise correlation calculations
7. **8 Optimization Techniques**: GA, PSO, SA, DE, ACO, BO, RS, HC
8. **Advanced Financial Metrics**: Kelly Criterion, VaR/CVaR, Sharpe/Sortino ratios, Calmar ratio
9. **Enhanced Output Generation**: Original 8 outputs + regime analysis + capital optimization reports

## Enhanced CSV Format Integration
```ini
[CSV_FORMAT]
format_detection = auto
legacy_support = true
enhanced_columns = start_time,end_time,market_regime,capital,zone
mandatory_columns = Date
strategy_pattern = auto_detect  # Any column not in enhanced/mandatory list
optional_columns = configurable
```

## Strategy Detection Logic
- **Auto-detection**: Any column not in mandatory/enhanced lists = strategy column
- **Flexible naming**: Supports strategy_1, momentum_algo, arb_strategy_v2, etc.
- **Validation**: Numeric data type confirmation for strategy columns
