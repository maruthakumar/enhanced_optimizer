"""
Dynamic Table Creation and Schema Management

This module handles automatic table creation with schema detection,
data type mapping, and optimization for wide tables.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import re
import configparser
from datetime import datetime
from dataclasses import dataclass, field

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class ColumnSchema:
    """Schema information for a single column"""
    name: str
    original_name: str
    data_type: str
    sql_type: str
    is_index: bool = False
    is_primary_key: bool = False
    is_nullable: bool = True
    has_special_handling: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TableSchema:
    """Complete schema for a table"""
    table_name: str
    columns: List[ColumnSchema]
    indexes: List[str]
    partitions: List[str]
    metadata: Dict[str, Any]
    
    @property
    def column_count(self) -> int:
        return len(self.columns)
    
    @property
    def strategy_column_count(self) -> int:
        """Count of strategy columns (excluding metadata columns)"""
        return sum(1 for col in self.columns 
                  if col.metadata.get('is_strategy', False))


class DynamicTableManager:
    """
    Manages dynamic table creation with automatic schema detection
    and optimization for wide tables
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the Dynamic Table Manager
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.logger = logging.getLogger(__name__)
        
        # Data type mapping rules
        self.type_mapping = self._initialize_type_mapping()
        
        # Special column patterns
        self.special_columns = {
            'date': re.compile(r'date|day|time', re.I),
            'zone': re.compile(r'zone|region|area', re.I),
            'index': re.compile(r'index|idx|id', re.I),
            'strategy': re.compile(r'strategy[\s_-]?\d+', re.I)
        }
        
        # Column name sanitization rules
        self.sanitize_rules = [
            (r'\s+', '_'),           # Replace spaces with underscores
            (r'[^\w]', '_'),         # Replace non-alphanumeric with underscores
            (r'_+', '_'),            # Collapse multiple underscores
            (r'^_|_$', ''),          # Remove leading/trailing underscores
            (r'^(\d)', r'col_\1'),   # Prefix columns starting with numbers
        ]
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file"""
        config = {
            'schema': {
                'auto_detect_types': True,
                'prefer_double_precision': True,
                'create_indexes': True,
                'add_metadata_columns': True,
                'max_column_name_length': 63,
                'default_string_length': 255
            },
            'optimization': {
                'wide_table_threshold': 1000,
                'use_columnar_storage': True,
                'partition_by_date': False,
                'compression': 'auto',
                'fragment_size': 75000000  # 75M rows per fragment
            },
            'validation': {
                'check_data_integrity': True,
                'verify_row_counts': True,
                'sample_validation_rows': 1000,
                'log_schema_info': True
            }
        }
        
        if config_path and os.path.exists(config_path):
            parser = configparser.ConfigParser()
            parser.read(config_path)
            
            # Update config from file
            for section in ['schema', 'optimization', 'validation']:
                if section in parser:
                    config[section].update(dict(parser[section]))
        
        return config
    
    def _initialize_type_mapping(self) -> Dict[str, str]:
        """Initialize data type mapping rules"""
        return {
            # Pandas dtype -> SQL type mapping
            'int8': 'SMALLINT',
            'int16': 'SMALLINT',
            'int32': 'INTEGER',
            'int64': 'BIGINT',
            'uint8': 'SMALLINT',
            'uint16': 'INTEGER',
            'uint32': 'BIGINT',
            'uint64': 'BIGINT',
            'float16': 'FLOAT',
            'float32': 'FLOAT',
            'float64': 'DOUBLE',
            'bool': 'BOOLEAN',
            'datetime64[ns]': 'TIMESTAMP',
            'timedelta64[ns]': 'BIGINT',
            'object': 'TEXT',
            'string': 'TEXT',
            'category': 'TEXT'
        }
    
    def detect_schema_from_csv(self, filepath: str, 
                              sample_size: int = 10000) -> TableSchema:
        """
        Detect schema from CSV file
        
        Args:
            filepath: Path to CSV file
            sample_size: Number of rows to sample for type detection
            
        Returns:
            TableSchema object with detected schema
        """
        self.logger.info(f"Detecting schema from {filepath}")
        
        # Extract table name from file
        base_name = os.path.splitext(os.path.basename(filepath))[0]
        table_name = self._sanitize_table_name(base_name)
        
        # Read sample for schema detection
        df_sample = pd.read_csv(filepath, nrows=sample_size)
        self.logger.info(f"Analyzing {len(df_sample.columns)} columns from {len(df_sample)} sample rows")
        
        # Detect column schemas
        columns = []
        indexes = []
        strategy_count = 0
        
        for col_name in df_sample.columns:
            col_schema = self._analyze_column(col_name, df_sample[col_name])
            columns.append(col_schema)
            
            # Track strategy columns
            if col_schema.metadata.get('is_strategy', False):
                strategy_count += 1
            
            # Identify index candidates
            if col_schema.is_index:
                indexes.append(col_schema.name)
        
        self.logger.info(f"Detected {strategy_count} strategy columns")
        
        # Add metadata columns if configured
        if self.config['schema']['add_metadata_columns']:
            columns.extend(self._create_metadata_columns())
        
        # Create indexes based on configuration
        if self.config['schema']['create_indexes']:
            indexes.extend(self._determine_indexes(columns))
        
        # Determine partitioning strategy
        partitions = self._determine_partitions(columns)
        
        # Build schema metadata
        metadata = {
            'source_file': filepath,
            'detected_at': datetime.now().isoformat(),
            'column_count': len(columns),
            'strategy_column_count': strategy_count,
            'is_wide_table': len(columns) > self.config['optimization']['wide_table_threshold'],
            'sample_size': len(df_sample)
        }
        
        schema = TableSchema(
            table_name=table_name,
            columns=columns,
            indexes=list(set(indexes)),  # Remove duplicates
            partitions=partitions,
            metadata=metadata
        )
        
        # Log schema information if configured
        if self.config['validation']['log_schema_info']:
            self._log_schema_info(schema)
        
        return schema
    
    def _analyze_column(self, col_name: str, series: pd.Series) -> ColumnSchema:
        """
        Analyze a single column to determine its schema
        
        Args:
            col_name: Original column name
            series: Pandas series with column data
            
        Returns:
            ColumnSchema object
        """
        # Sanitize column name
        sanitized_name = self._sanitize_column_name(col_name)
        
        # Detect data type
        dtype_str = str(series.dtype)
        
        # Try to infer better type if object
        if dtype_str == 'object':
            # Try numeric conversion
            try:
                numeric_series = pd.to_numeric(series, errors='coerce')
                if numeric_series.notna().sum() > len(series) * 0.9:  # 90% numeric
                    if numeric_series.dtype == 'float64':
                        dtype_str = 'float64'
                    else:
                        dtype_str = 'int64'
            except:
                pass
            
            # Try datetime conversion
            if dtype_str == 'object':
                try:
                    datetime_series = pd.to_datetime(series, errors='coerce', format='mixed')
                    if datetime_series.notna().sum() > len(series) * 0.9:  # 90% dates
                        dtype_str = 'datetime64[ns]'
                except:
                    pass
        
        # Map to SQL type
        sql_type = self._map_to_sql_type(dtype_str, series)
        
        # Check for special column types
        is_index = False
        is_strategy = False
        has_special = False
        
        for special_type, pattern in self.special_columns.items():
            if pattern.search(col_name):
                has_special = True
                if special_type == 'index':
                    is_index = True
                elif special_type == 'strategy':
                    is_strategy = True
                    sql_type = 'DOUBLE' if self.config['schema']['prefer_double_precision'] else 'FLOAT'
                break
        
        # Check nullability
        is_nullable = series.isna().any()
        
        # Build metadata
        metadata = {
            'original_dtype': dtype_str,
            'null_count': int(series.isna().sum()),
            'unique_count': len(series.unique()),
            'is_strategy': is_strategy
        }
        
        return ColumnSchema(
            name=sanitized_name,
            original_name=col_name,
            data_type=dtype_str,
            sql_type=sql_type,
            is_index=is_index,
            is_nullable=is_nullable,
            has_special_handling=has_special,
            metadata=metadata
        )
    
    def _map_to_sql_type(self, dtype_str: str, series: pd.Series) -> str:
        """Map pandas dtype to SQL type"""
        # Check mapping table first
        if dtype_str in self.type_mapping:
            return self.type_mapping[dtype_str]
        
        # Handle special cases
        if 'int' in dtype_str:
            # Check range for optimal integer type
            if series.notna().any():
                min_val = series.min()
                max_val = series.max()
                if -32768 <= min_val and max_val <= 32767:
                    return 'SMALLINT'
                elif -2147483648 <= min_val and max_val <= 2147483647:
                    return 'INTEGER'
            return 'BIGINT'
        
        elif 'float' in dtype_str:
            return 'DOUBLE' if self.config['schema']['prefer_double_precision'] else 'FLOAT'
        
        elif dtype_str == 'object':
            # Check if it's all strings
            max_len = series.astype(str).str.len().max()
            if max_len <= self.config['schema']['default_string_length']:
                return f"VARCHAR({self.config['schema']['default_string_length']})"
            else:
                return 'TEXT'
        
        # Default fallback
        return 'TEXT'
    
    def _sanitize_column_name(self, name: str) -> str:
        """Sanitize column name for SQL compatibility"""
        sanitized = name
        
        # Apply sanitization rules
        for pattern, replacement in self.sanitize_rules:
            sanitized = re.sub(pattern, replacement, sanitized)
        
        # Ensure not empty
        if not sanitized:
            sanitized = 'column'
        
        # Truncate if too long
        max_len = self.config['schema']['max_column_name_length']
        if len(sanitized) > max_len:
            sanitized = sanitized[:max_len]
        
        # Ensure uniqueness will be handled by caller if needed
        return sanitized.lower()
    
    def _sanitize_table_name(self, name: str) -> str:
        """Sanitize table name for SQL compatibility"""
        # Similar to column name but with table-specific rules
        sanitized = self._sanitize_column_name(name)
        
        # Prefix with 'tbl_' if starts with number
        if sanitized and sanitized[0].isdigit():
            sanitized = f"tbl_{sanitized}"
        
        return sanitized
    
    def _create_metadata_columns(self) -> List[ColumnSchema]:
        """Create metadata columns for tracking"""
        return [
            ColumnSchema(
                name='load_timestamp',
                original_name='load_timestamp',
                data_type='timestamp',
                sql_type='TIMESTAMP',
                is_nullable=False,
                metadata={'default': 'CURRENT_TIMESTAMP'}
            ),
            ColumnSchema(
                name='row_hash',
                original_name='row_hash',
                data_type='string',
                sql_type='VARCHAR(64)',
                is_nullable=True,
                metadata={'description': 'MD5 hash of row data for integrity checking'}
            )
        ]
    
    def _determine_indexes(self, columns: List[ColumnSchema]) -> List[str]:
        """Determine which columns should be indexed"""
        indexes = []
        
        # Always index primary keys and explicit index columns
        for col in columns:
            if col.is_primary_key or col.is_index:
                indexes.append(col.name)
        
        # Index date columns for time-series queries
        for col in columns:
            if col.sql_type in ['TIMESTAMP', 'DATE']:
                indexes.append(col.name)
        
        # Index low-cardinality columns that might be used for filtering
        for col in columns:
            if col.metadata.get('unique_count', 0) < 100 and col.name not in indexes:
                if not col.metadata.get('is_strategy', False):
                    indexes.append(col.name)
        
        return indexes
    
    def _determine_partitions(self, columns: List[ColumnSchema]) -> List[str]:
        """Determine partitioning strategy"""
        partitions = []
        
        if self.config['optimization']['partition_by_date']:
            # Find date columns
            for col in columns:
                if col.sql_type in ['TIMESTAMP', 'DATE']:
                    partitions.append(col.name)
                    break  # Usually partition by one date column
        
        return partitions
    
    def _log_schema_info(self, schema: TableSchema) -> None:
        """Log detailed schema information"""
        self.logger.info(f"Schema Detection Complete for table '{schema.table_name}':")
        self.logger.info(f"  Total columns: {schema.column_count}")
        self.logger.info(f"  Strategy columns: {schema.strategy_column_count}")
        self.logger.info(f"  Indexes: {', '.join(schema.indexes)}")
        
        if schema.metadata.get('is_wide_table'):
            self.logger.warning(f"  ⚠️  Wide table detected ({schema.column_count} columns)")
            self.logger.info("  Optimization will be applied for wide table handling")
        
        # Log column type distribution
        type_counts = {}
        for col in schema.columns:
            sql_type = col.sql_type.split('(')[0]  # Remove size specs
            type_counts[sql_type] = type_counts.get(sql_type, 0) + 1
        
        self.logger.info("  Column type distribution:")
        for sql_type, count in sorted(type_counts.items()):
            self.logger.info(f"    {sql_type}: {count}")
    
    def generate_create_table_sql(self, schema: TableSchema, 
                                 if_not_exists: bool = True) -> str:
        """
        Generate CREATE TABLE SQL statement
        
        Args:
            schema: TableSchema object
            if_not_exists: Add IF NOT EXISTS clause
            
        Returns:
            SQL CREATE TABLE statement
        """
        sql_parts = []
        
        # Table declaration
        if if_not_exists:
            sql_parts.append(f"CREATE TABLE IF NOT EXISTS {schema.table_name} (")
        else:
            sql_parts.append(f"CREATE TABLE {schema.table_name} (")
        
        # Column definitions
        col_defs = []
        for col in schema.columns:
            col_def = f"  {col.name} {col.sql_type}"
            
            if not col.is_nullable:
                col_def += " NOT NULL"
            
            if 'default' in col.metadata:
                col_def += f" DEFAULT {col.metadata['default']}"
            
            col_defs.append(col_def)
        
        sql_parts.append(",\n".join(col_defs))
        sql_parts.append(")")
        
        # Add table options for wide tables
        if schema.metadata.get('is_wide_table'):
            options = []
            
            if self.config['optimization']['use_columnar_storage']:
                options.append("COLUMNAR")
            
            if self.config['optimization']['fragment_size']:
                options.append(f"FRAGMENT_SIZE={self.config['optimization']['fragment_size']}")
            
            if self.config['optimization']['compression'] != 'none':
                options.append(f"COMPRESSION='{self.config['optimization']['compression']}'")
            
            if options:
                sql_parts.append(f"WITH ({', '.join(options)})")
        
        return "\n".join(sql_parts)
    
    def generate_index_sql(self, schema: TableSchema) -> List[str]:
        """Generate CREATE INDEX SQL statements"""
        index_statements = []
        
        for idx_col in schema.indexes:
            idx_name = f"idx_{schema.table_name}_{idx_col}"
            sql = f"CREATE INDEX IF NOT EXISTS {idx_name} ON {schema.table_name} ({idx_col})"
            index_statements.append(sql)
        
        return index_statements
    
    def validate_schema_implementation(self, schema: TableSchema, 
                                     actual_df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate that the implemented table matches the schema
        
        Args:
            schema: Expected schema
            actual_df: DataFrame with actual data
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check column count (excluding metadata columns)
        non_metadata_columns = [c for c in schema.columns if c.name not in ['load_timestamp', 'row_hash']]
        if len(actual_df.columns) != len(non_metadata_columns):
            issues.append(f"Column count mismatch: expected {len(non_metadata_columns)}, got {len(actual_df.columns)}")
        
        # Check column names
        expected_names = {col.original_name for col in non_metadata_columns}
        actual_names = set(actual_df.columns)
        
        missing = expected_names - actual_names
        if missing:
            issues.append(f"Missing columns: {missing}")
        
        extra = actual_names - expected_names
        if extra:
            issues.append(f"Extra columns: {extra}")
        
        # Sample data type validation
        if self.config['validation']['check_data_integrity']:
            sample_size = min(
                self.config['validation']['sample_validation_rows'],
                len(actual_df)
            )
            
            for col in schema.columns:
                if col.original_name in actual_df.columns:
                    series = actual_df[col.original_name].head(sample_size)
                    
                    # Check nullability
                    if not col.is_nullable and series.isna().any():
                        issues.append(f"Column {col.original_name} has nulls but is NOT NULL")
                    
                    # Check data type compatibility
                    if col.data_type.startswith('float') or col.data_type.startswith('int'):
                        try:
                            pd.to_numeric(series, errors='coerce')
                            non_numeric = series.notna() & pd.to_numeric(series, errors='coerce').isna()
                            if non_numeric.any():
                                issues.append(f"Column {col.original_name} has non-numeric values")
                        except:
                            issues.append(f"Column {col.original_name} type validation failed")
        
        is_valid = len(issues) == 0
        return is_valid, issues
    
    def optimize_for_wide_table(self, schema: TableSchema) -> TableSchema:
        """
        Apply optimizations for wide tables (many columns)
        
        Args:
            schema: Original schema
            
        Returns:
            Optimized schema
        """
        if schema.column_count <= self.config['optimization']['wide_table_threshold']:
            return schema  # No optimization needed
        
        self.logger.info(f"Applying wide table optimizations for {schema.column_count} columns")
        
        # Clone schema for modifications
        optimized = TableSchema(
            table_name=schema.table_name,
            columns=schema.columns.copy(),
            indexes=[],  # Minimize indexes for wide tables
            partitions=schema.partitions.copy(),
            metadata=schema.metadata.copy()
        )
        
        # Optimization 1: Reduce indexes to essential only
        essential_indexes = []
        for col in optimized.columns:
            if col.is_primary_key or col.name in ['date', 'zone', 'index']:
                essential_indexes.append(col.name)
        optimized.indexes = essential_indexes[:3]  # Limit to 3 indexes max
        
        # Optimization 2: Use appropriate data types for strategy columns
        for col in optimized.columns:
            if col.metadata.get('is_strategy', False):
                # Use FLOAT instead of DOUBLE for strategy columns to save space
                if not self.config['schema']['prefer_double_precision']:
                    col.sql_type = 'FLOAT'
        
        # Optimization 3: Add wide table metadata
        optimized.metadata['optimizations_applied'] = [
            'reduced_indexes',
            'optimized_data_types',
            'columnar_storage'
        ]
        
        return optimized