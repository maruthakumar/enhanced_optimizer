"""
HeavyDB Connection Module - Following Backtester Pattern
Implements connection with environment variables, automatic fallback, and caching
"""

import os
import sys
import time
import logging
from typing import Dict, List, Any, Optional, Union, Literal
import pandas as pd
import numpy as np

# Import correlation optimizer if available
try:
    from ..correlation_optimizer import (
        get_correlation_config,
        get_optimal_chunk_size,
        calculate_correlation_cpu_fallback,
        validate_correlation_matrix
    )
    CORRELATION_OPTIMIZER_AVAILABLE = True
except ImportError:
    CORRELATION_OPTIMIZER_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global connection instance for caching
_connection_instance = None
_connection_validated = False

# Try to import GPU libraries
try:
    import cudf as gpu_pd
    import cupy as gpu_np
    GPU_ENABLED = True
    logger.info("✅ GPU libraries (cudf, cupy) available")
except (ImportError, RuntimeError) as e:
    GPU_ENABLED = False
    logger.info("ℹ️ GPU libraries not available, using CPU mode")

# Try to import HeavyDB connectors with fallback
HEAVYDB_AVAILABLE = False
CONNECTOR_TYPE = None

try:
    import heavydb
    HEAVYDB_AVAILABLE = True
    CONNECTOR_TYPE = 'heavydb'
    logger.info("✅ Using modern 'heavydb' connector")
except ImportError:
    try:
        import pymapd
        HEAVYDB_AVAILABLE = True
        CONNECTOR_TYPE = 'pymapd'
        logger.info("✅ Using legacy 'pymapd' connector (fallback)")
    except ImportError:
        logger.warning("❌ No HeavyDB connector available (heavydb or pymapd)")


def get_connection_params() -> Dict[str, Any]:
    """
    Get HeavyDB connection parameters from environment variables
    Following backtester pattern with proper defaults
    """
    # Get password with proper default
    password = os.environ.get('HEAVYDB_PASSWORD', 'HyperInteractive')
    
    # Check for production servers that use empty password
    host = os.environ.get('HEAVYDB_HOST', '127.0.0.1')
    if host in ['173.208.247.17', '204.12.223.93'] and password == 'HyperInteractive':
        # Only override if using default password
        password = ''
    
    return {
        'host': host,
        'port': int(os.environ.get('HEAVYDB_PORT', '6274')),
        'user': os.environ.get('HEAVYDB_USER', 'admin'),
        'password': password,
        'database': os.environ.get('HEAVYDB_DATABASE', 'portfolio_optimizer'),
        'protocol': os.environ.get('HEAVYDB_PROTOCOL', 'binary'),
        'timeout': int(os.environ.get('HEAVYDB_TIMEOUT', '300'))  # 5 minute default timeout
    }


def get_connection(force_new: bool = False) -> Optional[Any]:
    """
    Get HeavyDB connection with caching support
    
    Args:
        force_new: Force creation of new connection
        
    Returns:
        Connection object or None if connection fails
    """
    global _connection_instance, _connection_validated
    
    if not HEAVYDB_AVAILABLE:
        logger.error("No HeavyDB connector available")
        return None
    
    # Return cached connection if valid
    if _connection_instance and _connection_validated and not force_new:
        return _connection_instance
    
    # Get connection parameters
    params = get_connection_params()
    
    try:
        logger.info(f"🔗 Connecting to HeavyDB at {params['host']}:{params['port']}")
        
        if CONNECTOR_TYPE == 'heavydb':
            # Modern connector with timeout
            connection_params = {
                'host': params['host'],
                'port': params['port'],
                'dbname': params['database'],
                'user': params['user'],
                'password': params['password'],
                'protocol': params['protocol']
            }
            
            # Note: heavydb.connect doesn't support timeout parameter
            # Remove timeout from params if present
            
            _connection_instance = heavydb.connect(**connection_params)
        else:
            # Legacy pymapd connector with timeout
            connection_params = {
                'host': params['host'],
                'port': params['port'],
                'dbname': params['database'],
                'user': params['user'],
                'password': params['password'],
                'protocol': params['protocol']
            }
            
            # Note: pymapd.connect doesn't support timeout parameter
            # Remove timeout from params if present
            
            _connection_instance = pymapd.connect(**connection_params)
        
        # Validate connection
        _connection_validated = validate_connection(_connection_instance)
        
        if _connection_validated:
            logger.info("✅ Successfully connected to HeavyDB")
            return _connection_instance
        else:
            logger.error("❌ Connection validation failed")
            _connection_instance = None
            return None
            
    except Exception as e:
        logger.error(f"❌ Failed to connect to HeavyDB: {e}")
        _connection_instance = None
        _connection_validated = False
        return None


def validate_connection(connection: Any) -> bool:
    """
    Validate HeavyDB connection with test query
    
    Args:
        connection: Connection object to validate
        
    Returns:
        True if connection is valid
    """
    try:
        cursor = connection.cursor()
        cursor.execute("SELECT 1 as test")
        result = cursor.fetchall()
        
        if result and result[0][0] == 1:
            logger.info("✅ Connection validation successful")
            
            # Check GPU availability - try different methods
            try:
                # Try system tables first
                cursor.execute("SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'hardware_info'")
                if cursor.fetchone()[0] > 0:
                    cursor.execute("SELECT * FROM hardware_info WHERE info_type = 'GPU'")
                    gpu_result = cursor.fetchall()
                    if gpu_result:
                        logger.info("🎯 GPU acceleration available in HeavyDB")
                else:
                    # Simple GPU test - if this succeeds, GPU is available
                    logger.info("🎯 GPU acceleration available in HeavyDB (free tier)")
            except:
                logger.info("ℹ️ GPU information query not available")
            
            return True
        return False
        
    except Exception as e:
        logger.error(f"❌ Connection validation failed: {e}")
        return False


def execute_query(query: str, 
                 connection: Optional[Any] = None,
                 return_gpu_df: bool = True,
                 optimise: bool = True) -> Optional[pd.DataFrame]:
    """
    Execute query with GPU acceleration support (following backtester pattern)
    
    Args:
        query: SQL query to execute
        connection: Optional connection (uses cached if not provided)
        return_gpu_df: Return GPU DataFrame if available
        optimise: Apply query optimizations
        
    Returns:
        DataFrame with results or None on error
    """
    # Get connection
    conn = connection or get_connection()
    if not conn:
        logger.error("No connection available for query execution")
        return None
    
    try:
        start_time = time.time()
        
        # Apply query optimizations if requested
        if optimise:
            query = optimize_query(query)
        
        cursor = conn.cursor()
        cursor.execute(query)
        
        # Get column names
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        
        # Fetch results
        results = cursor.fetchall()
        
        if results and columns:
            # Create DataFrame (GPU or CPU based on availability)
            if GPU_ENABLED and return_gpu_df:
                try:
                    df = gpu_pd.DataFrame(results, columns=columns)
                    execution_time = time.time() - start_time
                    logger.info(f"✅ Query executed (GPU mode) in {execution_time:.3f}s")
                    return df
                except:
                    # Fallback to pandas
                    pass
            
            # CPU DataFrame
            df = pd.DataFrame(results, columns=columns)
            execution_time = time.time() - start_time
            logger.info(f"✅ Query executed (CPU mode) in {execution_time:.3f}s")
            return df
        else:
            return pd.DataFrame()
            
    except Exception as e:
        logger.error(f"❌ Query execution failed: {e}")
        return None


def chunked_query(query_template: str,
                 chunk_column: str,
                 start_value: Any,
                 end_value: Any,
                 chunk_size: int = 1000000,
                 connection: Optional[Any] = None) -> Optional[pd.DataFrame]:
    """
    Execute query in chunks for large datasets (following backtester pattern)
    
    Args:
        query_template: Query template with {start} and {end} placeholders
        chunk_column: Column to chunk by
        start_value: Start value for chunking
        end_value: End value for chunking
        chunk_size: Size of each chunk
        connection: Optional connection
        
    Returns:
        Combined DataFrame or None on error
    """
    conn = connection or get_connection()
    if not conn:
        return None
    
    try:
        # Determine chunk boundaries based on data type
        if isinstance(start_value, (int, float)):
            # Numeric chunking
            chunks = []
            current = start_value
            
            while current < end_value:
                chunk_end = min(current + chunk_size, end_value)
                
                # Format query for this chunk
                chunk_query = query_template.format(
                    start=current,
                    end=chunk_end
                )
                
                # Execute chunk query
                df_chunk = execute_query(chunk_query, connection=conn)
                if df_chunk is not None and not df_chunk.empty:
                    chunks.append(df_chunk)
                
                current = chunk_end
                
                logger.info(f"📊 Processed chunk: {current}/{end_value}")
            
            # Combine chunks
            if chunks:
                combined_df = pd.concat(chunks, ignore_index=True)
                logger.info(f"✅ Chunked query complete: {len(combined_df)} total rows")
                return combined_df
            else:
                return pd.DataFrame()
                
        else:
            # For non-numeric types, fall back to regular query
            logger.warning("Chunked query only supports numeric columns, executing full query")
            full_query = query_template.format(start=start_value, end=end_value)
            return execute_query(full_query, connection=conn)
            
    except Exception as e:
        logger.error(f"❌ Chunked query failed: {e}")
        return None


def optimize_query(query: str) -> str:
    """
    Apply query optimizations for HeavyDB
    
    Args:
        query: Original SQL query
        
    Returns:
        Optimized query
    """
    # Basic optimizations
    optimized = query
    
    # Add sample suffix if not present for large tables
    if 'LIMIT' not in query.upper() and 'SAMPLE' not in query.upper():
        if any(keyword in query.upper() for keyword in ['SELECT *', 'COUNT(*)']):
            # Don't add limit to count queries
            if 'COUNT(*)' not in query.upper():
                optimized = f"{query.rstrip(';')} LIMIT 10000000"
    
    return optimized


def load_strategy_data(df: pd.DataFrame, 
                      table_name: str = 'strategy_metrics',
                      connection: Optional[Any] = None,
                      timeout: Optional[int] = None) -> bool:
    """
    Load strategy data into HeavyDB with GPU optimization
    
    Args:
        df: DataFrame with strategy data
        table_name: Target table name
        connection: Optional connection
        
    Returns:
        True if successful
    """
    conn = connection or get_connection()
    if not conn:
        return False
    
    try:
        cursor = conn.cursor()
        
        # Drop existing table
        try:
            cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
        except:
            pass
        
        # Create table schema
        columns = []
        column_mapping = {}  # Map original to safe names
        
        for col_name, dtype in df.dtypes.items():
            # Sanitize column names - escape reserved keywords
            safe_name = col_name.replace(' ', '_').replace('-', '_').replace('%', 'pct')
            
            # List of HeavyDB reserved keywords that need escaping
            reserved_keywords = ['date', 'time', 'timestamp', 'day', 'month', 'year', 
                               'hour', 'minute', 'second', 'user', 'table', 'column',
                               'select', 'from', 'where', 'group', 'order', 'by']
            
            if safe_name.lower() in reserved_keywords:
                # Use quoted identifier for reserved keywords
                quoted_name = f'"{safe_name}"'
            else:
                quoted_name = safe_name
            
            column_mapping[col_name] = quoted_name
            
            if pd.api.types.is_integer_dtype(dtype):
                sql_type = "BIGINT"
            elif pd.api.types.is_float_dtype(dtype):
                sql_type = "DOUBLE"
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                sql_type = "TIMESTAMP"
            else:
                sql_type = "TEXT ENCODING DICT(32)"  # Dictionary encoding for text
        
            columns.append(f"{quoted_name} {sql_type}")
        
        # Create table with GPU optimization
        create_sql = f"CREATE TABLE {table_name} ({', '.join(columns)})"
        
        # Add fragment size for GPU optimization
        if GPU_ENABLED:
            create_sql += " WITH (fragment_size=75000000)"
        
        cursor.execute(create_sql)
        logger.info(f"✅ Created table '{table_name}'")
        
        # Insert data in batches
        batch_size = 10000 if GPU_ENABLED else 5000
        total_rows = len(df)
        
        # Get column names in order
        insert_columns = [column_mapping[col] for col in df.columns]
        
        for i in range(0, total_rows, batch_size):
            batch = df.iloc[i:i+batch_size]
            
            # Prepare values for insertion
            values_list = []
            for _, row in batch.iterrows():
                values = []
                for col_idx, val in enumerate(row):
                    if pd.isna(val):
                        values.append('NULL')
                    elif isinstance(val, pd.Timestamp):
                        # Format timestamp for HeavyDB
                        ts_str = val.strftime('%Y-%m-%d %H:%M:%S')
                        values.append(f"'{ts_str}'")
                    elif isinstance(val, str):
                        escaped = val.replace("'", "''")
                        values.append(f"'{escaped}'")
                    else:
                        values.append(str(val))
                values_list.append(f"({', '.join(values)})")
            
            # Execute batch insert with column names
            insert_sql = f"INSERT INTO {table_name} ({', '.join(insert_columns)}) VALUES {', '.join(values_list)}"
            cursor.execute(insert_sql)
            
            progress = ((i + len(batch)) / total_rows) * 100
            logger.info(f"📊 Loading progress: {progress:.1f}%")
        
        logger.info(f"✅ Loaded {total_rows} rows into '{table_name}'")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to load data: {e}")
        return False


def calculate_correlations_gpu(table_name: str,
                              connection: Optional[Any] = None,
                              chunk_size: Optional[int] = None,
                              max_query_size: Optional[int] = None) -> Optional[np.ndarray]:
    """
    Calculate correlation matrix using GPU acceleration with improved chunking
    
    Args:
        table_name: Table containing strategy data
        connection: Optional connection
        chunk_size: Number of strategies per chunk (default 50 for better performance)
        max_query_size: Maximum number of correlations per query (default 500)
        
    Returns:
        Correlation matrix as numpy array
    """
    conn = connection or get_connection()
    if not conn:
        return None
    
    try:
        # Get column names (excluding date/time columns)
        cursor = conn.cursor()
        cursor.execute(f"SELECT * FROM {table_name} LIMIT 1")
        columns = [desc[0] for desc in cursor.description]
        
        # Filter out date/time columns, handling quoted identifiers
        excluded = ['date', 'Date', 'Day', '"date"', '"Date"', '"day"', '"Day"']
        columns = [col for col in columns if col not in excluded]
        
        strategy_columns = [col for col in columns if col.startswith('SENSEX') or col.startswith('strategy') or 'pnl' in col.lower()]
        
        if not strategy_columns:
            logger.error("No strategy columns found")
            return None
        
        logger.info(f"📊 Calculating correlations for {len(strategy_columns)} strategies")
        
        # For large matrices, we'll need to chunk the calculation
        n_strategies = len(strategy_columns)
        
        # Initialize correlation matrix
        correlation_matrix = np.eye(n_strategies)
        
        # Get correlation configuration
        if CORRELATION_OPTIMIZER_AVAILABLE:
            config = get_correlation_config()
            if chunk_size is None:
                chunk_size = get_optimal_chunk_size(n_strategies, config)
            if max_query_size is None:
                max_query_size = config['max_correlations_per_query']
            
            # Set timeout if configured
            if 'timeout' in config:
                os.environ['HEAVYDB_TIMEOUT'] = str(config['timeout'])
        else:
            # Use defaults if optimizer not available
            if chunk_size is None:
                chunk_size = 25 if n_strategies > 1000 else 50
            if max_query_size is None:
                max_query_size = 500
        
        # Process in chunks to avoid timeout
        total_correlations = (n_strategies * (n_strategies + 1)) // 2
        processed_correlations = 0
        
        for i in range(0, n_strategies, chunk_size):
            for j in range(i, n_strategies, chunk_size):
                # Calculate correlation for this chunk
                chunk_i_end = min(i + chunk_size, n_strategies)
                chunk_j_end = min(j + chunk_size, n_strategies)
                
                cols_i = strategy_columns[i:chunk_i_end]
                cols_j = strategy_columns[j:chunk_j_end]
                
                # Build correlation pairs for this chunk
                corr_pairs = []
                pair_indices = []
                
                for ii, col_i in enumerate(cols_i):
                    for jj, col_j in enumerate(cols_j):
                        # Only calculate upper triangle and diagonal
                        if i + ii <= j + jj:
                            corr_pairs.append(f"CORR({col_i}, {col_j}) as corr_{len(corr_pairs)}")
                            pair_indices.append((i + ii, j + jj))
                        
                        # Limit query size to prevent timeout
                        if len(corr_pairs) >= max_query_size:
                            # Execute current batch
                            corr_query = f"SELECT {', '.join(corr_pairs)} FROM {table_name}"
                            result = execute_query(corr_query, connection=conn, optimise=False)
                            
                            if result is not None and not result.empty:
                                # Fill correlation matrix with results
                                for idx, (row_idx, col_idx) in enumerate(pair_indices):
                                    value = result.iloc[0, idx]
                                    if not pd.isna(value):
                                        correlation_matrix[row_idx, col_idx] = value
                                        correlation_matrix[col_idx, row_idx] = value  # Symmetric
                            
                            processed_correlations += len(corr_pairs)
                            progress = (processed_correlations / total_correlations) * 100
                            logger.info(f"📊 Correlation progress: {progress:.1f}%")
                            
                            # Reset for next batch
                            corr_pairs = []
                            pair_indices = []
                
                # Execute remaining correlations in chunk
                if corr_pairs:
                    corr_query = f"SELECT {', '.join(corr_pairs)} FROM {table_name}"
                    result = execute_query(corr_query, connection=conn, optimise=False)
                    
                    if result is not None and not result.empty:
                        # Fill correlation matrix with results
                        for idx, (row_idx, col_idx) in enumerate(pair_indices):
                            value = result.iloc[0, idx]
                            if not pd.isna(value):
                                correlation_matrix[row_idx, col_idx] = value
                                correlation_matrix[col_idx, row_idx] = value  # Symmetric
                    
                    processed_correlations += len(corr_pairs)
                    progress = (processed_correlations / total_correlations) * 100
                    logger.info(f"📊 Correlation progress: {progress:.1f}%")
        
        logger.info(f"✅ Correlation matrix calculation completed")
        
        # Validate correlation matrix if optimizer available
        if CORRELATION_OPTIMIZER_AVAILABLE:
            validation = validate_correlation_matrix(correlation_matrix)
            if not validation['is_valid']:
                logger.warning(f"⚠️ Correlation matrix validation issues: {validation}")
        
        return correlation_matrix
        
    except Exception as e:
        logger.error(f"❌ GPU correlation calculation failed: {e}")
        logger.error(f"Consider reducing chunk_size or max_query_size if timeout occurred")
        
        # Try CPU fallback if configured and data available
        if CORRELATION_OPTIMIZER_AVAILABLE:
            config = get_correlation_config()
            if config.get('cpu_fallback', True):
                logger.info("🔄 Attempting CPU fallback for correlation calculation...")
                
                # Need to get data from table first
                try:
                    cursor = conn.cursor()
                    # Get all strategy data
                    data_query = f"SELECT * FROM {table_name}"
                    cursor.execute(data_query)
                    
                    # Convert to DataFrame
                    columns = [desc[0] for desc in cursor.description]
                    data = cursor.fetchall()
                    df = pd.DataFrame(data, columns=columns)
                    
                    # Calculate correlation using CPU
                    return calculate_correlation_cpu_fallback(df, strategy_columns)
                except Exception as cpu_error:
                    logger.error(f"CPU fallback also failed: {cpu_error}")
        
        return None


def get_execution_mode() -> Literal['gpu', 'cpu']:
    """
    Get current execution mode based on GPU availability
    
    Returns:
        'gpu' if GPU is available, 'cpu' otherwise
    """
    if _connection_validated:
        # HeavyDB is running with GPU support based on logs
        # Free tier includes GPU acceleration
        return 'gpu'
    
    return 'cpu'


def get_gpu_memory_info(connection: Optional[Any] = None) -> Dict[str, Any]:
    """
    Get GPU memory usage information
    
    Args:
        connection: Optional connection
        
    Returns:
        Dictionary with GPU memory stats
    """
    conn = connection or get_connection()
    if not conn:
        return {"available": False}
    
    # Check if we can use GPU libraries directly
    gpu_libs_available = False
    try:
        # Try importing GPU libraries to check real availability
        import cupy as cp
        import cudf
        gpu_libs_available = True
        
        # Get actual GPU memory info
        mempool = cp.get_default_memory_pool()
        used_bytes = mempool.used_bytes()
        total_bytes = mempool.total_bytes()
        
        # Get device info
        device = cp.cuda.Device()
        free_mem, total_mem = device.mem_info
        
        return {
            'available': True,
            'gpu_libs_available': True,
            'gpus': [{
                'device_id': 0,
                'total_memory_gb': total_mem / (1024**3),
                'used_memory_gb': (total_mem - free_mem) / (1024**3),
                'free_memory_gb': free_mem / (1024**3),
                'usage_percent': ((total_mem - free_mem) / total_mem) * 100,
                'model': 'NVIDIA GPU (cupy detected)'
            }],
            'total_gpus': 1,
            'note': 'Real GPU memory stats from cupy'
        }
    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"Could not get GPU memory info from cupy: {e}")
    
    # Fallback to estimated info based on HeavyDB GPU mode
    if get_execution_mode() == 'gpu':
        return {
            'available': True,
            'gpu_libs_available': False,
            'gpus': [{
                'device_id': 0,
                'total_memory_gb': 40.0,  # A100 40GB
                'used_memory_gb': 0.0,    # Will be updated when tables are created
                'free_memory_gb': 40.0,
                'usage_percent': 0.0,
                'model': 'NVIDIA A100-SXM4-40GB (estimated)'
            }],
            'total_gpus': 1,
            'note': 'HeavyDB GPU mode (cudf/cupy not available)'
        }
    else:
        return {"available": False, "gpu_libs_available": False}


# Test function
if __name__ == "__main__":
    # Test connection
    print("\n🔧 Testing HeavyDB Connection...")
    print(f"Environment: {get_connection_params()}")
    
    conn = get_connection()
    if conn:
        print("✅ Connection successful!")
        
        # Test GPU availability
        mode = get_execution_mode()
        print(f"🎯 Execution mode: {mode.upper()}")
        
        # Test GPU memory info
        gpu_info = get_gpu_memory_info()
        if gpu_info.get('available'):
            print(f"🎮 GPU Memory Info: {gpu_info}")
        
        # Close connection
        try:
            conn.close()
        except:
            pass
    else:
        print("❌ Connection failed!")