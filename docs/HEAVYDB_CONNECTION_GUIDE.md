# HeavyDB Connection Guide

> **DEPRECATION NOTICE**: HeavyDB is being replaced by Parquet/Arrow/cuDF as per the 
> architecture migration. See [docs/architecture.md](./architecture.md) for the new 
> data stack specification. This guide remains for reference during the transition period.

This document provides comprehensive information on how to connect to HeavyDB in the backtester project.

## Overview

The project supports multiple HeavyDB connection configurations across different modules, with fallback mechanisms and GPU acceleration support.

## Connection Parameters

### Default Configuration

The system uses environment variables with fallback defaults:

```bash
HEAVYDB_HOST="127.0.0.1"          # Default: localhost
HEAVYDB_PORT="6274"               # Default: 6274
HEAVYDB_USER="admin"              # Default: admin
HEAVYDB_PASSWORD="HyperInteractive" # Default password
HEAVYDB_DATABASE="heavyai"        # Default database name
```

### Production Server Configuration

For production deployment:

```bash
HEAVYDB_HOST="204.12.223.93"     # Production server IP
HEAVYDB_PORT="6274"
HEAVYDB_USER="admin"
HEAVYDB_PASSWORD=""               # Empty password for production
HEAVYDB_DATABASE="heavyai"
HEAVYDB_PROTOCOL="binary"
```

## Environment Setup

### 1. Set Environment Variables

Create a `.env` file or export these variables:

```bash
# For local development
export HEAVYDB_HOST="127.0.0.1"
export HEAVYDB_PORT="6274"
export HEAVYDB_USER="admin"
export HEAVYDB_PASSWORD="HyperInteractive"
export HEAVYDB_DATABASE="heavyai"

# For production
export HEAVYDB_HOST="204.12.223.93"
export HEAVYDB_PORT="6274"
export HEAVYDB_USER="admin"
export HEAVYDB_PASSWORD=""
export HEAVYDB_DATABASE="heavyai"
export HEAVYDB_PROTOCOL="binary"
```

### 2. Install Required Packages

The system supports two HeavyDB connector libraries with automatic fallback:

```bash
# Option 1: Modern HeavyDB connector (recommended)
pip install heavydb

# Option 2: Legacy pymapd connector (fallback)
pip install pymapd

# For GPU acceleration (optional)
pip install cudf cupy
```

## Connection Methods

### Method 1: Using Core Connection Module

```python
from core.heavydb_connection import get_connection, execute_query

# Get connection
conn = get_connection()

# Execute query with optimizations
df = execute_query(
    query="SELECT * FROM nifty_option_chain LIMIT 10",
    connection=conn,
    return_gpu_df=True,  # Use GPU if available
    optimise=True        # Apply query optimizations
)
```

### Method 2: Using DAL Module

```python
from dal.heavydb_conn import get_conn

# Get connection
conn = get_conn()
if conn:
    # Use connection for queries
    result = conn.execute("SELECT COUNT(*) FROM nifty_option_chain")
    print(result.fetchall())
```

### Method 3: Using Server Database Module

```python
from server.app.core.database import create_connection

# Create new connection
conn = create_connection()
```

### Method 4: Using HeavyDB Client (for Options Data)

```python
from input_sheets.heavydb_client import HeavyDBClient

# Initialize client with automatic connection
client = HeavyDBClient()

# Execute queries with GPU acceleration
df = client.execute_query("SELECT * FROM nifty_option_chain WHERE expiry_date = '2024-01-25'")
```

## Connection Features

### Automatic Library Detection

The system automatically detects and uses available HeavyDB libraries:

1. **heavydb** (modern connector) - preferred
2. **pymapd** (legacy connector) - fallback

### GPU Acceleration

When available, the system uses GPU acceleration:

```python
# GPU libraries detection
try:
    import cudf as pd
    import cupy as np
    GPU_ENABLED = True
except ImportError:
    import pandas as pd
    import numpy as np
    GPU_ENABLED = False
```

### Connection Caching

The system implements connection caching to prevent repeated connections:

```python
# Global connection instance for reuse
_connection_instance = None
_connection_validated = False
```

## Query Execution

### Basic Query Execution

```python
from core.heavydb_connection import execute_query

# Simple query
df = execute_query("SELECT * FROM your_table LIMIT 100")

# Query with connection reuse
conn = get_connection()
df1 = execute_query("SELECT * FROM table1", connection=conn)
df2 = execute_query("SELECT * FROM table2", connection=conn)
```

### Chunked Query Execution

For large datasets:

```python
from core.heavydb_connection import chunked_query

# Execute query in chunks
df = chunked_query(
    query_template="SELECT * FROM large_table WHERE date_column BETWEEN {start} AND {end}",
    chunk_column="date_column",
    start_value="2024-01-01",
    end_value="2024-12-31",
    chunk_size=1000000
)
```

## Configuration Files

The connection parameters are defined in multiple configuration files:

- `core/config.py` - Main configuration with environment variable support
- `server/app/core/config.py` - Server-specific configuration
- `dal/heavydb_connection.py` - DAL layer configuration
- `backtester_v2/ml_triple_rolling_straddle_system/core/database_config.py` - ML system configuration

## Troubleshooting

### Common Issues

1. **Import Error**: Install either `heavydb` or `pymapd`
2. **Connection Timeout**: Check host and port configuration
3. **Authentication Error**: Verify username and password
4. **GPU Issues**: Ensure CUDA drivers are installed for GPU acceleration

### Debug Logging

Enable debug logging to troubleshoot connection issues:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
```

### Connection Testing

Test your connection:

```python
from core.heavydb_connection import get_connection

try:
    conn = get_connection()
    if conn:
        print("✅ Successfully connected to HeavyDB")
        # Test query
        result = conn.execute("SELECT 1 as test")
        print(f"✅ Test query successful: {result.fetchall()}")
    else:
        print("❌ Failed to connect to HeavyDB")
except Exception as e:
    print(f"❌ Connection error: {e}")
```

## Additional Configuration Options

### Query Optimization Settings

```bash
HEAVYDB_QUERY_TIMEOUT="300"       # Query timeout in seconds
HEAVYDB_CHUNK_SIZE="1000000"      # Chunk size for large queries
HEAVYDB_MAX_ROWS="10000000"       # Maximum rows per query
HEAVYDB_USE_GPU="True"            # Enable GPU acceleration
```

### Connection Pool Settings

```bash
HEAVYDB_TIMEOUT="30"              # Connection timeout
HEAVYDB_MAX_RETRIES="3"           # Maximum retry attempts
```

## Best Practices

1. **Use Environment Variables**: Always configure connection parameters via environment variables
2. **Connection Reuse**: Reuse connections when executing multiple queries
3. **Error Handling**: Always wrap connection code in try-catch blocks
4. **GPU Acceleration**: Enable GPU acceleration for large datasets when available
5. **Query Optimization**: Use the built-in query optimization features
6. **Connection Pooling**: Leverage the connection caching mechanisms

## Support

For issues or questions regarding HeavyDB connections, check:

1. Connection logs in the application
2. HeavyDB server status and accessibility
3. Network connectivity to the HeavyDB host
4. Required Python package installations
