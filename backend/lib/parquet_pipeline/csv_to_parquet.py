"""
CSV to Parquet Converter with Auto Schema Detection
Supports both legacy and enhanced CSV formats
"""

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

# Set up logging
logger = logging.getLogger(__name__)

# Define mandatory and enhanced columns
MANDATORY_COLUMNS = ['Date']
ENHANCED_COLUMNS = [
    'start_time', 'end_time', 'market_regime', 
    'Regime_Confidence_%', 'Market_regime_transition_threshold',
    'capital', 'zone'
]

def detect_csv_schema(csv_path: str, sample_rows: int = 1000) -> Dict[str, str]:
    """
    Auto-detect CSV schema including strategy columns
    
    Args:
        csv_path: Path to CSV file
        sample_rows: Number of rows to sample for schema detection
        
    Returns:
        Dictionary mapping column names to detected data types
    """
    logger.info(f"Detecting schema for {csv_path}")
    
    # Read sample of CSV
    df_sample = pd.read_csv(csv_path, nrows=sample_rows)
    
    schema = {}
    strategy_columns = []
    
    for col in df_sample.columns:
        # Skip any Symbol columns that might have gotten included by mistake
        if col.lower() in ['symbol', 'symbols', 'ticker', 'tickers']:
            logger.info(f"Skipping symbol column '{col}' - not needed for optimization")
            continue
            
        # Check if it's a mandatory column
        if col in MANDATORY_COLUMNS:
            if col == 'Date':
                schema[col] = 'date'
        # Check if it's an enhanced column
        elif col in ENHANCED_COLUMNS:
            if col in ['start_time', 'end_time']:
                schema[col] = 'datetime64[ns]'
            elif col in ['market_regime', 'zone']:
                schema[col] = 'string'
            elif col in ['Regime_Confidence_%', 'Market_regime_transition_threshold', 'capital']:
                schema[col] = 'float64'
        # Otherwise, it's a strategy column
        else:
            # Verify it's numeric and not all NaN
            if pd.api.types.is_numeric_dtype(df_sample[col]) and not df_sample[col].isna().all():
                strategy_columns.append(col)
                schema[col] = 'float64'
            else:
                logger.warning(f"Non-numeric or empty column '{col}' detected, skipping")
    
    logger.info(f"Detected {len(strategy_columns)} strategy columns")
    logger.info(f"Enhanced columns found: {[col for col in ENHANCED_COLUMNS if col in df_sample.columns]}")
    
    return schema, strategy_columns

def csv_to_parquet(csv_path: str, 
                  parquet_path: str, 
                  format_config: Optional[Dict] = None,
                  compression: str = 'snappy',
                  row_group_size: int = 50000) -> bool:
    """
    Convert CSV file to Parquet format with optimized schema
    
    Args:
        csv_path: Input CSV file path
        parquet_path: Output Parquet file path
        format_config: Optional configuration for column handling
        compression: Compression algorithm (snappy, gzip, lz4, zstd)
        row_group_size: Number of rows per row group
        
    Returns:
        True if conversion successful, False otherwise
    """
    try:
        logger.info(f"Converting {csv_path} to Parquet format")
        
        # Detect schema
        schema, strategy_columns = detect_csv_schema(csv_path)
        
        # Read CSV in chunks for memory efficiency
        chunk_size = 100000
        chunks = []
        
        # Define date parser
        date_parser = lambda x: pd.to_datetime(x, format='%Y-%m-%d')
        
        # Read CSV with appropriate data types
        dtype_dict = {}
        parse_dates = []
        
        for col, dtype in schema.items():
            if dtype == 'date':
                parse_dates.append(col)
            elif dtype == 'datetime64[ns]':
                parse_dates.append(col)
            elif dtype == 'float64':
                dtype_dict[col] = np.float64
            elif dtype == 'string':
                dtype_dict[col] = str
        
        # Read CSV in chunks
        for chunk_df in pd.read_csv(csv_path, 
                                   chunksize=chunk_size,
                                   dtype=dtype_dict,
                                   parse_dates=parse_dates,
                                   date_parser=date_parser if parse_dates else None):
            
            # Apply format configuration if provided
            if format_config:
                # Handle column exclusions
                if 'exclude_columns' in format_config:
                    chunk_df = chunk_df.drop(columns=format_config['exclude_columns'], errors='ignore')
                
                # Handle column renaming
                if 'rename_columns' in format_config:
                    chunk_df = chunk_df.rename(columns=format_config['rename_columns'])
            
            chunks.append(chunk_df)
        
        # Combine chunks
        df = pd.concat(chunks, ignore_index=True)
        logger.info(f"Loaded {len(df)} rows with {len(df.columns)} columns")
        
        # Create PyArrow table with schema
        arrow_schema = []
        for col in df.columns:
            if col == 'Date' or col in ['start_time', 'end_time']:
                arrow_schema.append((col, pa.timestamp('ns')))
            elif col in ['market_regime', 'zone']:
                arrow_schema.append((col, pa.string()))
            else:
                arrow_schema.append((col, pa.float64()))
        
        table = pa.Table.from_pandas(df, schema=pa.schema(arrow_schema))
        
        # Write Parquet file with partitioning by date
        pq.write_table(
            table,
            parquet_path,
            compression=compression,
            row_group_size=row_group_size,
            use_dictionary=True,
            compression_level=9 if compression == 'gzip' else None
        )
        
        logger.info(f"Successfully wrote Parquet file to {parquet_path}")
        
        # Validate the written file
        if validate_parquet_file(parquet_path):
            logger.info("Parquet file validation successful")
            return True
        else:
            logger.error("Parquet file validation failed")
            return False
            
    except Exception as e:
        logger.error(f"Error converting CSV to Parquet: {str(e)}")
        return False

def validate_parquet_file(parquet_path: str) -> bool:
    """
    Validate Parquet file integrity and schema
    
    Args:
        parquet_path: Path to Parquet file
        
    Returns:
        True if valid, False otherwise
    """
    try:
        # Open Parquet file
        pf = pq.ParquetFile(parquet_path)
        
        # Check metadata
        metadata = pf.metadata
        logger.info(f"Parquet file has {metadata.num_rows} rows and {metadata.num_columns} columns")
        
        # Read first row group to validate
        first_row_group = pf.read_row_group(0)
        
        # Verify mandatory columns
        columns = first_row_group.column_names
        if 'Date' not in columns:
            logger.error("Missing mandatory 'Date' column")
            return False
        
        # Check for strategy columns
        strategy_cols = [col for col in columns if col not in MANDATORY_COLUMNS + ENHANCED_COLUMNS]
        if len(strategy_cols) == 0:
            logger.error("No strategy columns found")
            return False
        
        logger.info(f"Validation passed: {len(strategy_cols)} strategy columns found")
        return True
        
    except Exception as e:
        logger.error(f"Validation error: {str(e)}")
        return False

def get_parquet_metadata(parquet_path: str) -> Dict:
    """
    Get metadata information from Parquet file
    
    Args:
        parquet_path: Path to Parquet file
        
    Returns:
        Dictionary containing metadata information
    """
    try:
        pf = pq.ParquetFile(parquet_path)
        metadata = pf.metadata
        
        info = {
            'num_rows': metadata.num_rows,
            'num_columns': metadata.num_columns,
            'num_row_groups': metadata.num_row_groups,
            'format_version': metadata.format_version,
            'created_by': metadata.created_by,
            'columns': pf.schema.names,
            'file_size_mb': Path(parquet_path).stat().st_size / (1024 * 1024)
        }
        
        # Get column statistics
        col_stats = []
        for i in range(metadata.num_row_groups):
            rg = metadata.row_group(i)
            for j in range(rg.num_columns):
                col = rg.column(j)
                col_stats.append({
                    'column': pf.schema.names[j],
                    'row_group': i,
                    'compressed_size': col.total_compressed_size,
                    'uncompressed_size': col.total_uncompressed_size
                })
        
        info['column_stats'] = col_stats
        
        return info
        
    except Exception as e:
        logger.error(f"Error getting metadata: {str(e)}")
        return {}

def optimize_parquet_storage(parquet_path: str, 
                           output_path: str,
                           partitions: Optional[List[str]] = None,
                           prune_columns: Optional[List[str]] = None) -> None:
    """
    Optimize Parquet file storage with partitioning and column pruning
    
    Args:
        parquet_path: Input Parquet file
        output_path: Output directory for optimized Parquet
        partitions: List of columns to partition by
        prune_columns: List of columns to exclude
    """
    try:
        logger.info(f"Optimizing Parquet storage for {parquet_path}")
        
        # Read Parquet file
        df = pd.read_parquet(parquet_path)
        
        # Prune columns if specified
        if prune_columns:
            df = df.drop(columns=prune_columns, errors='ignore')
            logger.info(f"Pruned {len(prune_columns)} columns")
        
        # Write with partitioning
        if partitions:
            # Extract year and month from Date for partitioning
            if 'Date' in partitions:
                df['year'] = pd.to_datetime(df['Date']).dt.year
                df['month'] = pd.to_datetime(df['Date']).dt.month
                partitions = ['year', 'month']
            
            # Write partitioned dataset
            table = pa.Table.from_pandas(df)
            pq.write_to_dataset(
                table,
                root_path=output_path,
                partition_cols=partitions,
                compression='snappy',
                use_legacy_dataset=False
            )
            logger.info(f"Wrote partitioned dataset to {output_path}")
        else:
            # Write single optimized file
            df.to_parquet(
                output_path,
                engine='pyarrow',
                compression='snappy',
                index=False
            )
            logger.info(f"Wrote optimized file to {output_path}")
            
    except Exception as e:
        logger.error(f"Error optimizing storage: {str(e)}")
        raise