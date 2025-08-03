"""
Arrow Memory Management for Zero-Copy GPU Transfers
Optimized for large strategy datasets
"""

import pyarrow as pa
import pyarrow.parquet as pq
import logging
from typing import Optional, List, Tuple
import gc
import psutil

# Try to import cuDF for GPU operations
try:
    import cudf
    CUDF_AVAILABLE = True
except (ImportError, RuntimeError) as e:
    CUDF_AVAILABLE = False
    logging.warning(f"cuDF not available ({str(e)}), GPU operations will be disabled")

logger = logging.getLogger(__name__)

class ArrowMemoryPool:
    """
    Custom memory pool for Arrow operations with monitoring
    """
    def __init__(self, pool_size_gb: float = 4.0):
        """
        Initialize memory pool
        
        Args:
            pool_size_gb: Size of memory pool in GB
        """
        self.pool_size_bytes = int(pool_size_gb * 1024 * 1024 * 1024)
        self.pool = pa.default_memory_pool()
        self.initial_bytes = self.pool.bytes_allocated()
        logger.info(f"Initialized Arrow memory pool with {pool_size_gb}GB limit")
    
    def get_usage(self) -> dict:
        """Get current memory usage statistics"""
        return {
            'allocated_bytes': self.pool.bytes_allocated(),
            'allocated_gb': self.pool.bytes_allocated() / (1024**3),
            'max_memory_bytes': self.pool.max_memory(),
            'max_memory_gb': self.pool.max_memory() / (1024**3)
        }
    
    def cleanup(self):
        """Force garbage collection and cleanup"""
        gc.collect()
        logger.info(f"Memory cleanup completed. Current usage: {self.get_usage()}")

def create_memory_pool(size_gb: float = 4.0) -> ArrowMemoryPool:
    """
    Create and configure Arrow memory pool
    
    Args:
        size_gb: Pool size in GB
        
    Returns:
        Configured memory pool
    """
    return ArrowMemoryPool(size_gb)

def load_parquet_to_arrow(parquet_path: str, 
                         columns: Optional[List[str]] = None,
                         row_groups: Optional[List[int]] = None,
                         memory_pool: Optional[ArrowMemoryPool] = None) -> pa.Table:
    """
    Load Parquet file into Arrow Table with memory optimization
    
    Args:
        parquet_path: Path to Parquet file
        columns: Specific columns to load (None for all)
        row_groups: Specific row groups to load (None for all)
        memory_pool: Memory pool to use
        
    Returns:
        Arrow Table
    """
    try:
        logger.info(f"Loading Parquet file: {parquet_path}")
        
        # Open Parquet file
        pf = pq.ParquetFile(parquet_path)
        
        # Log metadata
        metadata = pf.metadata
        logger.info(f"Parquet file: {metadata.num_rows} rows, {metadata.num_columns} columns, "
                   f"{metadata.num_row_groups} row groups")
        
        # Determine columns to load
        if columns is None:
            columns = pf.schema.names
        else:
            # Validate requested columns exist
            available_cols = set(pf.schema.names)
            requested_cols = set(columns)
            missing_cols = requested_cols - available_cols
            if missing_cols:
                logger.warning(f"Requested columns not found: {missing_cols}")
                columns = list(requested_cols & available_cols)
        
        # Load data
        if row_groups is not None:
            # Load specific row groups
            table = pf.read_row_groups(row_groups, columns=columns)
        else:
            # Load all data
            table = pf.read(columns=columns)
        
        logger.info(f"Loaded Arrow table: {table.num_rows} rows, {table.num_columns} columns")
        
        # Log memory usage
        if memory_pool:
            usage = memory_pool.get_usage()
            logger.info(f"Memory usage: {usage['allocated_gb']:.2f}GB")
        
        return table
        
    except Exception as e:
        logger.error(f"Error loading Parquet to Arrow: {str(e)}")
        raise

def arrow_to_cudf(arrow_table: pa.Table, 
                  chunk_size: Optional[int] = None) -> 'cudf.DataFrame':
    """
    Convert Arrow Table to cuDF DataFrame with zero-copy transfer
    
    Args:
        arrow_table: Input Arrow Table
        chunk_size: If specified, process in chunks for large datasets
        
    Returns:
        cuDF DataFrame
    """
    if not CUDF_AVAILABLE:
        raise RuntimeError("cuDF is not available. Please install RAPIDS cuDF.")
    
    try:
        logger.info(f"Converting Arrow table to cuDF: {arrow_table.num_rows} rows")
        
        if chunk_size is None or arrow_table.num_rows <= chunk_size:
            # Direct conversion for small datasets
            df = cudf.DataFrame.from_arrow(arrow_table)
            logger.info(f"Created cuDF DataFrame: {len(df)} rows, {len(df.columns)} columns")
            return df
        else:
            # Chunked processing for large datasets
            chunks = []
            num_chunks = (arrow_table.num_rows + chunk_size - 1) // chunk_size
            
            for i in range(num_chunks):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, arrow_table.num_rows)
                
                # Slice Arrow table
                chunk_table = arrow_table.slice(start_idx, end_idx - start_idx)
                
                # Convert to cuDF
                chunk_df = cudf.DataFrame.from_arrow(chunk_table)
                chunks.append(chunk_df)
                
                logger.info(f"Processed chunk {i+1}/{num_chunks}: {len(chunk_df)} rows")
            
            # Concatenate chunks
            df = cudf.concat(chunks, ignore_index=True)
            logger.info(f"Combined {len(chunks)} chunks into cuDF DataFrame: {len(df)} rows")
            
            # Clean up chunks
            del chunks
            gc.collect()
            
            return df
            
    except Exception as e:
        logger.error(f"Error converting Arrow to cuDF: {str(e)}")
        raise

def monitor_memory_usage() -> dict:
    """
    Monitor system and GPU memory usage
    
    Returns:
        Dictionary with memory statistics
    """
    stats = {
        'system': {
            'total_gb': psutil.virtual_memory().total / (1024**3),
            'available_gb': psutil.virtual_memory().available / (1024**3),
            'used_gb': psutil.virtual_memory().used / (1024**3),
            'percent': psutil.virtual_memory().percent
        }
    }
    
    # Add GPU memory stats if available
    if CUDF_AVAILABLE:
        try:
            import cupy as cp
            mempool = cp.get_default_memory_pool()
            stats['gpu'] = {
                'used_bytes': mempool.used_bytes(),
                'used_gb': mempool.used_bytes() / (1024**3),
                'total_bytes': mempool.total_bytes(),
                'total_gb': mempool.total_bytes() / (1024**3)
            }
        except Exception as e:
            logger.warning(f"Could not get GPU memory stats: {e}")
    
    return stats

def load_parquet_to_cudf(parquet_path: str,
                        columns: Optional[List[str]] = None,
                        date_range: Optional[Tuple[str, str]] = None,
                        chunk_size: Optional[int] = None) -> 'cudf.DataFrame':
    """
    Convenience function to load Parquet directly to cuDF
    
    Args:
        parquet_path: Path to Parquet file
        columns: Columns to load
        date_range: Optional date range filter (start_date, end_date)
        chunk_size: Chunk size for large datasets
        
    Returns:
        cuDF DataFrame
    """
    # Load to Arrow first
    arrow_table = load_parquet_to_arrow(parquet_path, columns=columns)
    
    # Apply date filter if specified
    if date_range and 'Date' in arrow_table.column_names:
        start_date, end_date = date_range
        date_column = arrow_table['Date']
        
        # Create filter mask
        mask = (date_column >= pa.scalar(start_date)) & (date_column <= pa.scalar(end_date))
        arrow_table = arrow_table.filter(mask)
        logger.info(f"Filtered to date range {start_date} to {end_date}: {arrow_table.num_rows} rows")
    
    # Convert to cuDF
    return arrow_to_cudf(arrow_table, chunk_size=chunk_size)