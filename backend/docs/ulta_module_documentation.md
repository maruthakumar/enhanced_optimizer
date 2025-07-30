# ULTA Calculator Module Documentation

## Overview

The ULTA (Ultra Low Trading Algorithm) Calculator module implements the core strategy inversion logic from the Heavy Optimizer Platform. This module provides a clean, testable implementation that can work with both in-memory data (pandas/numpy) and GPU-accelerated databases (HeavyDB).

## Purpose

ULTA logic inverts poorly performing trading strategies to potentially improve their performance. The key insight is that some strategies that lose money might become profitable if their trading signals are reversed (buying when they would sell, and vice versa).

## Core Algorithm

### 1. ROI Calculation
For each strategy, calculate the Return on Investment:
```
ROI = sum(daily_returns)
```

### 2. Drawdown Calculation
Calculate the maximum drawdown:
```
Drawdown = min(daily_returns)
```

### 3. Ratio Calculation
Calculate the ROI/Drawdown ratio:
```
Ratio = ROI / abs(Drawdown)
```

### 4. Inversion Decision
- If `Ratio < 0` (negative), calculate the ratio for the inverted strategy
- Invert the strategy by multiplying all returns by -1
- Keep the inversion **only if** `inverted_ratio > original_ratio`

## Module Structure

### Classes

#### `ULTACalculator`
Main class implementing the ULTA logic for in-memory processing.

**Key Methods:**
- `calculate_roi(returns)`: Calculate ROI from daily returns
- `calculate_drawdown(returns)`: Calculate maximum drawdown
- `calculate_ratio(roi, drawdown)`: Calculate ROI/Drawdown ratio
- `invert_strategy(returns)`: Invert strategy returns
- `should_invert_strategy(returns)`: Determine if strategy should be inverted
- `apply_ulta_logic(data)`: Apply ULTA to entire DataFrame
- `generate_inversion_report(format)`: Generate report in markdown or JSON

#### `HeavyDBULTACalculator`
Extended class for HeavyDB GPU-accelerated processing.

**Additional Methods:**
- `create_ulta_metadata_table()`: Create metadata tracking table
- `apply_ulta_to_table()`: Apply ULTA using HeavyDB queries
- GPU-optimized batch processing methods

#### `ULTAStrategyMetrics`
Data class storing metrics for each strategy:
- `strategy_name`: Name of the strategy
- `original_roi`: ROI before inversion
- `inverted_roi`: ROI after inversion
- `original_drawdown`: Drawdown before inversion
- `inverted_drawdown`: Drawdown after inversion
- `original_ratio`: Ratio before inversion
- `inverted_ratio`: Ratio after inversion
- `improvement_percentage`: Percentage improvement from inversion
- `was_inverted`: Boolean indicating if strategy was inverted

## Usage Examples

### Basic Usage

```python
from ulta_calculator import ULTACalculator
import pandas as pd

# Create calculator instance
calculator = ULTACalculator()

# Load your data
data = pd.read_csv('strategies.csv')

# Apply ULTA logic
processed_data, inverted_strategies = calculator.apply_ulta_logic(data)

# Generate report
report = calculator.generate_inversion_report('markdown')
print(report)
```

### HeavyDB Usage

```python
from ulta_calculator import HeavyDBULTACalculator
import heavydb

# Connect to HeavyDB
conn = heavydb.connect(host='localhost', port=6274, user='admin', password='password')

# Create calculator with connection
calculator = HeavyDBULTACalculator(conn)

# Apply ULTA to table
inverted_metrics = calculator.apply_ulta_to_table(
    input_table='strategies_data',
    output_table='strategies_ulta_processed',
    metadata_table='ulta_inversions'
)
```

### Backward Compatibility

For compatibility with legacy code:

```python
from ulta_calculator import apply_ulta_logic

# Works exactly like the legacy function
processed_data, inverted_dict = apply_ulta_logic(data)
```

## Integration with Heavy Optimizer Platform

### Input Requirements
- DataFrame with columns: Date, Zone, Day, followed by strategy columns
- Strategy columns contain daily returns as numeric values
- Non-numeric values are converted to 0

### Output Format
- Processed DataFrame with inverted strategies renamed with "_inv" suffix
- Original poorly performing strategies are removed
- Metadata dictionary with detailed metrics for each inverted strategy

### Report Generation

The module can generate two types of reports:

1. **Markdown Report**: Human-readable format with:
   - Summary statistics
   - Detailed metrics for each inverted strategy
   - Improvement percentages

2. **JSON Report**: Machine-readable format with:
   - Summary object with aggregate statistics
   - Detailed metrics dictionary
   - Average improvement calculations

## Performance Considerations

### Memory Efficiency
- Processes strategies one at a time to minimize memory usage
- Uses numpy arrays for efficient numerical operations
- Drops original columns before adding inverted ones

### GPU Acceleration (HeavyDB)
- Batch processing of strategies
- SQL/CASE statements for efficient GPU operations
- Configurable batch size for memory management

### Benchmarks
- In-memory processing: <1 second for 50 strategies × 100 days
- HeavyDB processing: Scales to millions of data points
- Memory usage: Proportional to data size, not strategy count

## Testing

### Unit Tests
Comprehensive test suite covering:
- Individual calculation methods
- Edge cases (zero returns, infinite ratios)
- Backward compatibility
- Report generation
- Numeric precision

### Integration Tests
- Large dataset performance
- Consistency across multiple runs
- Comparison with legacy implementation

### Running Tests
```bash
cd /mnt/optimizer_share/backend
python -m pytest tests/test_ulta_calculator.py -v
```

## Error Handling

The module includes robust error handling:
- Non-numeric values converted to 0
- Division by zero returns infinity
- Failed inversions logged but don't stop processing
- Detailed error messages for debugging

## Future Enhancements

1. **Parallel Processing**: Multi-threaded strategy evaluation
2. **Additional Metrics**: Sharpe ratio, volatility-adjusted returns
3. **Machine Learning**: Predict which strategies benefit from inversion
4. **Real-time Processing**: Stream processing capabilities
5. **Visualization**: Built-in plotting of before/after performance

## Migration from Legacy Code

To migrate from the legacy `apply_ulta_logic()` function:

1. Replace import:
   ```python
   # Old
   from Optimizer_New_patched import apply_ulta_logic
   
   # New
   from ulta_calculator import apply_ulta_logic
   ```

2. The function signature and return format are identical
3. Add error handling for the new exceptions
4. Optionally use the class-based API for more features

## Best Practices

1. **Logging**: Always configure logging for production use
2. **Validation**: Verify input data format before processing
3. **Monitoring**: Track inversion rates and improvements
4. **Testing**: Run validation against known datasets
5. **Documentation**: Keep strategy naming conventions consistent