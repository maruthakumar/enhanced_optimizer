"""
HeavyDB Schema Optimization Module

Implements advanced optimization strategies for HeavyDB table schemas,
specifically designed for large-scale financial data with 25,544+ strategy columns.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import hashlib
import configparser

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dal.dynamic_table_manager import TableSchema, ColumnSchema, DynamicTableManager


class HeavyDBSchemaOptimizer:
    """
    Advanced schema optimization for HeavyDB with focus on:
    1. Columnar storage optimization for 25,544+ strategy columns
    2. Date-based partitioning for 82 trading days
    3. Optimal data encoding for Date, Strategy names, and P&L values
    4. Strategic indexing for analytical queries
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize HeavyDB Schema Optimizer
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_optimization_config(config_path)
        self.logger = logging.getLogger(__name__)
        
        # Production data specifications from story
        self.production_specs = {
            'file_size_mb': 39.2,
            'trading_days': 82,
            'strategies': 25544,
            'data_points': 2094608,
            'date_range': ('2024-01-04', '2024-07-26')
        }
        
        # Performance benchmarks from story
        self.performance_targets = {
            'load_time_seconds': 30,
            'memory_usage_gb': 2,
            'correlation_query_seconds': 5,
            'optimization_seconds': 300
        }
        
    def _load_optimization_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load optimization-specific configuration"""
        config = {
            'columnar_storage': {
                'fragment_size': 75000000,  # 75M rows per fragment
                'max_chunk_size': 32000000,  # 32M for GPU memory efficiency
                'compression': 'LZ4',  # Fast compression for real-time queries
                'dictionary_encoding': True,  # For strategy names
                'run_length_encoding': True,  # For date columns
            },
            'partitioning': {
                'strategy': 'date_based',
                'partition_size_days': 7,  # Weekly partitions for 82 days
                'partition_pruning': True,
                'parallel_scan': True,
            },
            'indexing': {
                'temporal_index': True,
                'strategy_hash_index': True,
                'correlation_bloom_filter': True,
                'max_indexes_per_table': 5,
            },
            'data_encoding': {
                'date_type': 'DATE',  # Not TIMESTAMP for daily data
                'strategy_precision': 'DOUBLE',  # Full precision for P&L
                'strategy_scale': 6,  # 6 decimal places
                'null_optimization': True,
            },
            'memory_optimization': {
                'gpu_memory_limit_gb': 8,
                'cpu_memory_limit_gb': 16,
                'batch_size_strategies': 1000,  # Process 1000 strategies at a time
                'lazy_loading': True,
            }
        }
        
        if config_path and os.path.exists(config_path):
            parser = configparser.ConfigParser()
            parser.read(config_path)
            
            # Update with file-based config
            for section_name in config.keys():
                if section_name in parser:
                    config[section_name].update(dict(parser[section_name]))
        
        return config
    
    def optimize_production_schema(self, schema: TableSchema) -> TableSchema:
        """
        Apply production-specific optimizations for 25,544 strategy dataset
        
        Args:
            schema: Base table schema
            
        Returns:
            Optimized schema for production workload
        """
        self.logger.info("Applying production-specific optimizations for large strategy dataset")
        
        # Clone schema for modifications
        optimized = TableSchema(
            table_name=schema.table_name,
            columns=schema.columns.copy(),
            indexes=[],
            partitions=[],
            metadata=schema.metadata.copy()
        )
        
        # 1. Optimize columnar storage for 25,544 strategies
        optimized = self._optimize_columnar_storage(optimized)
        
        # 2. Implement date-based partitioning for 82 trading days
        optimized = self._implement_date_partitioning(optimized)
        
        # 3. Optimize data encoding
        optimized = self._optimize_data_encoding(optimized)
        
        # 4. Strategic indexing for analytical queries
        optimized = self._implement_strategic_indexing(optimized)
        
        # 5. Add production metadata
        optimized.metadata.update({
            'optimization_version': '1.0',
            'optimized_for_production': True,
            'target_strategies': self.production_specs['strategies'],
            'target_trading_days': self.production_specs['trading_days'],
            'optimization_timestamp': datetime.now().isoformat()
        })
        
        self._log_optimization_summary(optimized)
        
        return optimized
    
    def _optimize_columnar_storage(self, schema: TableSchema) -> TableSchema:
        """
        Optimize columnar storage for 25,544+ strategy columns
        
        Args:
            schema: Input schema
            
        Returns:
            Schema with columnar storage optimizations
        """
        self.logger.info("Optimizing columnar storage for large strategy dataset")
        
        # Calculate optimal fragment size based on strategy count
        strategy_count = schema.strategy_column_count
        if strategy_count > 20000:
            # For very wide tables, use smaller fragments to fit in GPU memory
            fragment_size = self.config['columnar_storage']['max_chunk_size']
        else:
            fragment_size = self.config['columnar_storage']['fragment_size']
        
        # Group strategy columns for better cache locality
        strategy_columns = [col for col in schema.columns if col.metadata.get('is_strategy', False)]
        non_strategy_columns = [col for col in schema.columns if not col.metadata.get('is_strategy', False)]
        
        # Reorder columns: metadata first, then strategies grouped by similarity
        reordered_columns = non_strategy_columns + self._group_similar_strategies(strategy_columns)
        schema.columns = reordered_columns
        
        # Add columnar storage metadata
        schema.metadata.update({
            'columnar_optimizations': {
                'fragment_size': fragment_size,
                'compression': self.config['columnar_storage']['compression'],
                'dictionary_encoding': self.config['columnar_storage']['dictionary_encoding'],
                'strategy_grouping': 'similarity_based',
                'column_reordering': True
            }
        })
        
        return schema
    
    def _group_similar_strategies(self, strategy_columns: List[ColumnSchema]) -> List[ColumnSchema]:
        """
        Group similar strategy columns together for better cache locality
        
        Args:
            strategy_columns: List of strategy columns
            
        Returns:
            Reordered list with similar strategies grouped
        """
        # Simple grouping by strategy name patterns (SL/TP values)
        grouped = {
            'conservative': [],  # Low SL/TP values
            'moderate': [],      # Medium SL/TP values  
            'aggressive': []     # High SL/TP values
        }
        
        for col in strategy_columns:
            strategy_name = col.original_name.lower()
            
            # Extract SL/TP values from strategy name if possible
            if 'sl' in strategy_name and 'tp' in strategy_name:
                # Try to categorize by risk level
                if any(val in strategy_name for val in ['0.5', '1.0', '1.5']):
                    grouped['conservative'].append(col)
                elif any(val in strategy_name for val in ['2.0', '2.5', '3.0']):
                    grouped['moderate'].append(col)
                else:
                    grouped['aggressive'].append(col)
            else:
                grouped['moderate'].append(col)  # Default group
        
        # Return flattened list
        return grouped['conservative'] + grouped['moderate'] + grouped['aggressive']
    
    def _implement_date_partitioning(self, schema: TableSchema) -> TableSchema:
        """
        Implement date-based partitioning for 82 trading days
        
        Args:
            schema: Input schema
            
        Returns:
            Schema with date partitioning
        """
        self.logger.info("Implementing date-based partitioning for 82 trading days")
        
        # Find date column
        date_column = None
        for col in schema.columns:
            if col.sql_type in ['DATE', 'TIMESTAMP'] or 'date' in col.name.lower():
                date_column = col.name
                break
        
        if not date_column:
            self.logger.warning("No date column found for partitioning")
            return schema
        
        # Calculate partition strategy for 82 days
        partition_size_days = self.config['partitioning']['partition_size_days']
        total_partitions = (self.production_specs['trading_days'] + partition_size_days - 1) // partition_size_days
        
        schema.partitions = [f"PARTITION BY RANGE({date_column})"]
        
        # Add partition metadata
        schema.metadata.update({
            'partitioning': {
                'strategy': 'date_range',
                'column': date_column,
                'partition_size_days': partition_size_days,
                'total_partitions': total_partitions,
                'pruning_enabled': self.config['partitioning']['partition_pruning'],
                'parallel_scan': self.config['partitioning']['parallel_scan']
            }
        })
        
        return schema
    
    def _optimize_data_encoding(self, schema: TableSchema) -> TableSchema:
        """
        Optimize data encoding for Date, Strategy names, and P&L values
        
        Args:
            schema: Input schema
            
        Returns:
            Schema with optimized data encoding
        """
        self.logger.info("Optimizing data encoding for production data types")
        
        for col in schema.columns:
            original_type = col.sql_type
            
            # Optimize date columns
            if col.sql_type in ['TIMESTAMP', 'DATETIME'] and 'date' in col.name.lower():
                col.sql_type = self.config['data_encoding']['date_type']
                col.metadata['encoding_optimization'] = 'date_optimized'
                
                # Add run-length encoding for date columns (many repeated values)
                if self.config['columnar_storage']['run_length_encoding']:
                    col.metadata['compression'] = 'RLE'
            
            # Optimize strategy name columns
            elif col.sql_type in ['TEXT', 'VARCHAR'] and not col.metadata.get('is_strategy', False):
                # Use dictionary encoding for strategy names
                if self.config['columnar_storage']['dictionary_encoding']:
                    col.sql_type = 'TEXT ENCODING DICT(32)'
                    col.metadata['encoding_optimization'] = 'dictionary_encoded'
            
            # Optimize P&L value columns (strategy data)
            elif col.metadata.get('is_strategy', False):
                precision = self.config['data_encoding']['strategy_precision']
                scale = self.config['data_encoding']['strategy_scale']
                
                if precision == 'DOUBLE':
                    col.sql_type = 'DOUBLE'
                else:
                    col.sql_type = f'DECIMAL(15,{scale})'
                
                col.metadata['encoding_optimization'] = 'precision_optimized'
                col.metadata['decimal_scale'] = scale
            
            # Log encoding changes
            if original_type != col.sql_type:
                self.logger.debug(f"Optimized encoding for {col.name}: {original_type} -> {col.sql_type}")
        
        return schema
    
    def _implement_strategic_indexing(self, schema: TableSchema) -> TableSchema:
        """
        Implement strategic indexing for analytical queries
        
        Args:
            schema: Input schema
            
        Returns:
            Schema with optimized indexes
        """
        self.logger.info("Implementing strategic indexing for analytical performance")
        
        indexes = []
        
        # 1. Temporal index for time-series queries
        if self.config['indexing']['temporal_index']:
            date_cols = [col.name for col in schema.columns 
                        if col.sql_type in ['DATE', 'TIMESTAMP'] or 'date' in col.name.lower()]
            if date_cols:
                indexes.extend(date_cols[:1])  # Index primary date column
        
        # 2. Hash index for strategy lookups
        if self.config['indexing']['strategy_hash_index']:
            # Create composite hash index for frequently queried strategy groups
            strategy_cols = [col.name for col in schema.columns 
                           if col.metadata.get('is_strategy', False)]
            if len(strategy_cols) > 100:  # Only for large strategy sets
                # Create index on first few strategies as sample
                indexes.append(f"HASH({','.join(strategy_cols[:10])})")
        
        # 3. Correlation bloom filter for correlation matrix queries
        if self.config['indexing']['correlation_bloom_filter']:
            # Bloom filter for existence checks in correlation calculations
            indexes.append("BLOOM_FILTER")
        
        # Limit total indexes
        max_indexes = self.config['indexing']['max_indexes_per_table']
        schema.indexes = indexes[:max_indexes]
        
        # Add indexing metadata
        schema.metadata.update({
            'indexing_strategy': {
                'temporal_index': len([idx for idx in indexes if 'date' in idx.lower()]),
                'strategy_indexes': len([idx for idx in indexes if 'HASH' in idx]),
                'bloom_filters': len([idx for idx in indexes if 'BLOOM' in idx]),
                'total_indexes': len(schema.indexes)
            }
        })
        
        return schema
    
    def generate_optimized_create_sql(self, schema: TableSchema) -> str:
        """
        Generate optimized CREATE TABLE SQL with HeavyDB-specific features
        
        Args:
            schema: Optimized schema
            
        Returns:
            SQL CREATE TABLE statement with optimizations
        """
        sql_parts = []
        
        # Table declaration
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
        
        # Add HeavyDB-specific optimizations
        with_options = []
        
        # Columnar storage options
        if 'columnar_optimizations' in schema.metadata:
            opts = schema.metadata['columnar_optimizations']
            with_options.append(f"FRAGMENT_SIZE={opts['fragment_size']}")
            
            if opts.get('compression'):
                with_options.append(f"COMPRESSION='{opts['compression']}'")
        
        # Partitioning
        if schema.partitions:
            partition_def = schema.partitions[0]
            sql_parts.append(partition_def)
        
        # Memory optimization for GPU
        if schema.strategy_column_count > 10000:
            with_options.append("GPU_SHARED_MEMORY=TRUE")
            with_options.append("MAX_CHUNKS_PER_FRAGMENT=32")
        
        if with_options:
            sql_parts.append(f"WITH ({', '.join(with_options)})")
        
        return "\n".join(sql_parts)
    
    def generate_optimization_indexes(self, schema: TableSchema) -> List[str]:
        """
        Generate index creation SQL statements for optimization
        
        Args:
            schema: Optimized schema
            
        Returns:
            List of CREATE INDEX SQL statements
        """
        index_statements = []
        
        for i, index_def in enumerate(schema.indexes):
            if index_def == "BLOOM_FILTER":
                # Skip bloom filter - handled in table creation
                continue
            elif index_def.startswith("HASH("):
                # Create hash index
                idx_name = f"idx_{schema.table_name}_hash_{i}"
                columns = index_def[5:-1]  # Extract columns from HASH(...)
                sql = f"CREATE INDEX {idx_name} ON {schema.table_name} ({columns}) USING HASH"
                index_statements.append(sql)
            else:
                # Standard index
                idx_name = f"idx_{schema.table_name}_{index_def}"
                sql = f"CREATE INDEX {idx_name} ON {schema.table_name} ({index_def})"
                index_statements.append(sql)
        
        return index_statements
    
    def validate_optimization_performance(self, schema: TableSchema, 
                                        actual_performance: Dict[str, float]) -> Tuple[bool, List[str]]:
        """
        Validate that optimization meets performance targets
        
        Args:
            schema: Optimized schema
            actual_performance: Measured performance metrics
            
        Returns:
            Tuple of (meets_targets, list_of_issues)
        """
        issues = []
        
        # Check load time
        if actual_performance.get('load_time_seconds', 0) > self.performance_targets['load_time_seconds']:
            issues.append(f"Load time {actual_performance['load_time_seconds']:.2f}s exceeds target {self.performance_targets['load_time_seconds']}s")
        
        # Check memory usage
        if actual_performance.get('memory_usage_gb', 0) > self.performance_targets['memory_usage_gb']:
            issues.append(f"Memory usage {actual_performance['memory_usage_gb']:.2f}GB exceeds target {self.performance_targets['memory_usage_gb']}GB")
        
        # Check correlation query performance
        if actual_performance.get('correlation_query_seconds', 0) > self.performance_targets['correlation_query_seconds']:
            issues.append(f"Correlation query time {actual_performance['correlation_query_seconds']:.2f}s exceeds target {self.performance_targets['correlation_query_seconds']}s")
        
        # Check optimization performance
        if actual_performance.get('optimization_seconds', 0) > self.performance_targets['optimization_seconds']:
            issues.append(f"Optimization time {actual_performance['optimization_seconds']:.2f}s exceeds target {self.performance_targets['optimization_seconds']}s")
        
        meets_targets = len(issues) == 0
        return meets_targets, issues
    
    def _log_optimization_summary(self, schema: TableSchema) -> None:
        """Log summary of applied optimizations"""
        self.logger.info("🎯 HeavyDB Schema Optimization Complete")
        self.logger.info(f"   Table: {schema.table_name}")
        self.logger.info(f"   Total columns: {schema.column_count}")
        self.logger.info(f"   Strategy columns: {schema.strategy_column_count}")
        self.logger.info(f"   Partitions: {len(schema.partitions)}")
        self.logger.info(f"   Indexes: {len(schema.indexes)}")
        
        if 'columnar_optimizations' in schema.metadata:
            opts = schema.metadata['columnar_optimizations']
            self.logger.info(f"   Fragment size: {opts['fragment_size']:,}")
            self.logger.info(f"   Compression: {opts['compression']}")
        
        if 'partitioning' in schema.metadata:
            part = schema.metadata['partitioning']
            self.logger.info(f"   Partition strategy: {part['strategy']}")
            self.logger.info(f"   Partitions: {part['total_partitions']}")
        
        self.logger.info("🚀 Schema optimized for production workload")


class ProductionBenchmarkValidator:
    """
    Validates HeavyDB performance against production benchmarks
    """
    
    def __init__(self, schema_optimizer: HeavyDBSchemaOptimizer):
        """
        Initialize benchmark validator
        
        Args:
            schema_optimizer: Schema optimizer instance
        """
        self.optimizer = schema_optimizer
        self.logger = logging.getLogger(__name__)
    
    def run_production_benchmark(self, table_name: str, 
                                connection_manager) -> Dict[str, float]:
        """
        Run production benchmark tests
        
        Args:
            table_name: Name of optimized table
            connection_manager: HeavyDB connection manager
            
        Returns:
            Performance metrics dictionary
        """
        self.logger.info("🔥 Running production benchmark validation")
        
        metrics = {}
        
        # 1. Test correlation matrix query performance
        metrics['correlation_query_seconds'] = self._benchmark_correlation_query(
            table_name, connection_manager
        )
        
        # 2. Test memory usage during operations
        metrics['memory_usage_gb'] = self._benchmark_memory_usage(
            table_name, connection_manager
        )
        
        # 3. Test strategy subset queries
        metrics['strategy_query_seconds'] = self._benchmark_strategy_queries(
            table_name, connection_manager
        )
        
        # 4. Test temporal queries (date range)
        metrics['temporal_query_seconds'] = self._benchmark_temporal_queries(
            table_name, connection_manager
        )
        
        self.logger.info("📊 Benchmark Results:")
        for metric, value in metrics.items():
            self.logger.info(f"   {metric}: {value:.3f}")
        
        return metrics
    
    def _benchmark_correlation_query(self, table_name: str, 
                                   connection_manager) -> float:
        """Benchmark correlation matrix calculation"""
        import time
        
        # Test correlation calculation for top 1000 strategies
        start_time = time.time()
        
        query = f"""
        SELECT CORR(strategy_1, strategy_2) as correlation
        FROM {table_name}
        LIMIT 1000
        """
        
        try:
            result = connection_manager.execute_gpu_query(query)
            correlation_time = time.time() - start_time
            
            return correlation_time
        except Exception as e:
            self.logger.error(f"Correlation benchmark failed: {e}")
            return 999.0  # High value indicates failure
    
    def _benchmark_memory_usage(self, table_name: str, 
                              connection_manager) -> float:
        """Benchmark memory usage during operations"""
        import psutil
        
        process = psutil.Process()
        memory_before = process.memory_info().rss / (1024**3)  # GB
        
        # Perform memory-intensive operation
        query = f"SELECT COUNT(*) FROM {table_name}"
        
        try:
            connection_manager.execute_gpu_query(query)
            memory_after = process.memory_info().rss / (1024**3)  # GB
            
            return memory_after - memory_before
        except Exception as e:
            self.logger.error(f"Memory benchmark failed: {e}")
            return 999.0  # High value indicates failure
    
    def _benchmark_strategy_queries(self, table_name: str, 
                                  connection_manager) -> float:
        """Benchmark strategy subset queries"""
        import time
        
        start_time = time.time()
        
        # Query a subset of strategies
        query = f"""
        SELECT strategy_1, strategy_100, strategy_1000
        FROM {table_name}
        LIMIT 10000
        """
        
        try:
            result = connection_manager.execute_gpu_query(query)
            query_time = time.time() - start_time
            
            return query_time
        except Exception as e:
            self.logger.error(f"Strategy query benchmark failed: {e}")
            return 999.0
    
    def _benchmark_temporal_queries(self, table_name: str, 
                                  connection_manager) -> float:
        """Benchmark date-range queries"""
        import time
        
        start_time = time.time()
        
        # Query data for a specific date range
        query = f"""
        SELECT date, COUNT(*) as record_count
        FROM {table_name}
        WHERE date >= '2024-01-04' AND date <= '2024-01-31'
        GROUP BY date
        ORDER BY date
        """
        
        try:
            result = connection_manager.execute_gpu_query(query)
            query_time = time.time() - start_time
            
            return query_time
        except Exception as e:
            self.logger.error(f"Temporal query benchmark failed: {e}")
            return 999.0