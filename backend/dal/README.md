# Data Access Layer (DAL) Implementation

## Overview

The DAL provides a clean abstraction for all database operations in the Heavy Optimizer Platform. It supports both HeavyDB GPU-accelerated operations and CSV fallback mode, ensuring the system can operate with or without a database.

## Architecture

```
dal/
├── base_dal.py          # Abstract base class defining the interface
├── heavydb_dal.py       # HeavyDB implementation with GPU acceleration
├── csv_dal.py           # CSV-based fallback implementation
├── dal_factory.py       # Factory for creating appropriate DAL instances
└── __init__.py          # Module exports
```

## Usage

### Basic Usage

```python
from backend.dal import get_dal

# Automatically selects appropriate DAL (HeavyDB if available, CSV otherwise)
dal = get_dal()

# Connect to data source
dal.connect()

# Load CSV data
dal.load_csv_to_heavydb('input/data.csv', 'my_table')

# Apply ULTA transformation
dal.apply_ulta_transformation('my_table')

# Compute correlation matrix
corr_matrix = dal.compute_correlation_matrix('my_table')

# Get specific strategies
subset = dal.get_strategy_subset('my_table', [0, 5, 10, 15])

# Clean up
dal.disconnect()
```

### Context Manager Usage

```python
from backend.dal import get_dal

with get_dal() as dal:
    # Operations are automatically cleaned up
    dal.load_csv_to_heavydb('input/data.csv', 'my_table')
    corr_matrix = dal.compute_correlation_matrix('my_table')
```

### Forcing Specific DAL Type

```python
from backend.dal import get_dal

# Force CSV DAL (no database required)
csv_dal = get_dal('csv')

# Force HeavyDB DAL (will fail if HeavyDB not available)
heavydb_dal = get_dal('heavydb')
```

## Configuration

The DAL reads configuration from `/mnt/optimizer_share/config/production_config.ini`:

```ini
[database]
host = localhost
port = 6274
database = heavyai
user = admin
password = HyperInteractive
protocol = binary

[performance]
batch_size = 10000
gpu_memory_limit = 8589934592  # 8GB

[system]
dal_type = auto  # auto, heavydb, or csv
```

## Features

### HeavyDB DAL
- GPU-accelerated operations when available
- Automatic CPU fallback
- Dynamic schema detection
- Batch data loading for optimal performance
- Connection pooling

### CSV DAL
- No database dependencies
- In-memory operations using pandas
- Full compatibility with HeavyDB DAL interface
- Suitable for development and testing

## Methods

All DAL implementations provide these methods:

- `connect()` - Establish connection
- `disconnect()` - Close connection
- `load_csv_to_heavydb(filepath, table_name)` - Load CSV data
- `apply_ulta_transformation(table_name)` - Apply ULTA logic
- `compute_correlation_matrix(table_name)` - Calculate correlations
- `get_strategy_subset(table_name, strategy_list)` - Get specific strategies
- `execute_gpu_query(sql_query)` - Execute SQL query
- `get_table_schema(table_name)` - Get table structure
- `create_dynamic_table(table_name, df)` - Create table from DataFrame
- `get_table_row_count(table_name)` - Get row count
- `drop_table(table_name)` - Drop table
- `supports_gpu` - Check GPU availability

## Testing

Run unit tests:

```bash
cd /mnt/optimizer_share/backend
python3 -m tests.test_dal
```

## Integration Example

```python
# In csv_only_heavydb_workflow.py
from dal import get_dal

def process_optimization(input_file, portfolio_size):
    with get_dal() as dal:
        # Load data
        dal.load_csv_to_heavydb(input_file, 'strategies')
        
        # Apply ULTA
        dal.apply_ulta_transformation('strategies')
        
        # Get correlation matrix
        corr_matrix = dal.compute_correlation_matrix('strategies_ulta')
        
        # Continue with optimization...
```

## Error Handling

The DAL provides robust error handling:

```python
dal = get_dal()
if not dal.connect():
    print("Failed to connect to data source")
    # System continues with CSV fallback

success = dal.load_csv_to_heavydb('data.csv', 'table')
if not success:
    print("Failed to load data")
```

## Performance Considerations

- HeavyDB DAL uses GPU acceleration for operations on large datasets
- CSV DAL is suitable for smaller datasets (< 1M rows)
- Batch size is configurable for optimal memory usage
- Correlation calculations use chunking for very large matrices

## Future Enhancements

- Support for additional databases (PostgreSQL, MySQL)
- Advanced caching mechanisms
- Distributed processing support
- Real-time data streaming capabilities