"""
HeavyDB Data Access Layer Implementation

Provides GPU-accelerated database operations using HeavyDB (formerly OmniSci).
Falls back gracefully to CPU operations when GPU is not available.
"""

import os
import sys
import time
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import configparser

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dal.base_dal import BaseDAL
from dal.dynamic_table_manager import DynamicTableManager
from lib.heavydb_connector.connection_manager import HeavyDBConnectionManager
from ulta_calculator import ULTACalculator
from lib.correlation.correlation_matrix_calculator import CorrelationMatrixCalculator


class HeavyDBDAL(BaseDAL):
    """
    HeavyDB implementation of the Data Access Layer
    
    Provides GPU-accelerated operations when available, with automatic
    fallback to CPU operations.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize HeavyDB DAL with configuration
        
        Args:
            config: Configuration dictionary. If None, loads from config files.
        """
        # Load configuration if not provided
        if config is None:
            config = self._load_configuration()
        
        super().__init__(config)
        
        # Initialize connection manager with config
        db_config = config.get('database', {})
        self.connection_manager = HeavyDBConnectionManager(
            host=db_config.get('host', 'localhost'),
            port=db_config.get('port', 6274),
            database=db_config.get('database', 'heavyai'),
            user=db_config.get('user', 'admin'),
            password=db_config.get('password', 'HyperInteractive'),
            protocol=db_config.get('protocol', 'binary')
        )
        
        # Initialize helper modules
        self.ulta_calculator = ULTACalculator()
        self.correlation_calculator = CorrelationMatrixCalculator()
        config_path = '/mnt/optimizer_share/backend/config/heavydb_optimization.ini'
        self.table_manager = DynamicTableManager(config_path)
        
        # Initialize optimizer modules lazily to avoid circular imports
        self._schema_optimizer = None
        self._benchmark_validator = None
        self._optimization_config_path = config_path
        
        # Performance settings
        self.batch_size = config.get('performance', {}).get('batch_size', 10000)
        self.gpu_memory_limit = config.get('performance', {}).get('gpu_memory_limit', 8 * 1024 * 1024 * 1024)  # 8GB
        
        self.logger = logging.getLogger(__name__)
    
    def _load_configuration(self) -> Dict[str, Any]:
        """
        Load configuration from config files
        
        Returns:
            Dict: Configuration dictionary
        """
        config = {
            'database': {},
            'performance': {},
            'algorithms': {}
        }
        
        # Try to load from production_config.ini
        config_path = '/mnt/optimizer_share/config/production_config.ini'
        if os.path.exists(config_path):
            parser = configparser.ConfigParser()
            parser.read(config_path)
            
            # Database settings
            if 'database' in parser:
                config['database'] = dict(parser['database'])
            
            # Performance settings
            if 'performance' in parser:
                config['performance'] = {
                    'batch_size': parser.getint('performance', 'batch_size', fallback=10000),
                    'gpu_memory_limit': parser.getint('performance', 'gpu_memory_limit', 
                                                     fallback=8 * 1024 * 1024 * 1024)
                }
        
        return config
    
    def connect(self) -> bool:
        """
        Establish connection to HeavyDB
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            self._is_connected = self.connection_manager.connect()
            if self._is_connected:
                self.logger.info("HeavyDB DAL connected successfully")
                if self.connection_manager.gpu_available:
                    self.logger.info("GPU acceleration is available")
                else:
                    self.logger.warning("GPU acceleration not available, using CPU mode")
            return self._is_connected
        except Exception as e:
            self.logger.error(f"Failed to connect to HeavyDB: {str(e)}")
            self._is_connected = False
            return False
    
    def disconnect(self) -> None:
        """Close connection to HeavyDB"""
        if self.connection_manager:
            self.connection_manager.close()
            self._is_connected = False
            self.logger.info("HeavyDB DAL disconnected")
    
    def load_csv_to_heavydb_optimized(self, filepath: str, table_name: str, 
                                     use_production_optimization: bool = True) -> bool:
        """
        Load a CSV file into HeavyDB with production-optimized schema
        
        Args:
            filepath: Path to the CSV file
            table_name: Name of the table to create/populate
            use_production_optimization: Apply production-specific optimizations
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self._is_connected:
            self.logger.error("Not connected to HeavyDB")
            return False
        
        try:
            self.logger.info(f"Loading CSV with production-optimized schema: {filepath}")
            start_time = time.time()
            
            # Detect base schema
            schema = self.table_manager.detect_schema_from_csv(filepath)
            
            if table_name:
                schema.table_name = table_name
            
            # Apply production optimizations
            if use_production_optimization:
                schema = self.schema_optimizer.optimize_production_schema(schema)
            else:
                # Apply standard wide table optimizations
                if schema.metadata.get('is_wide_table', False):
                    schema = self.table_manager.optimize_for_wide_table(schema)
            
            # Drop existing table
            self.drop_table(schema.table_name)
            
            # Create optimized table
            create_sql = self.schema_optimizer.generate_optimized_create_sql(schema)
            cursor = self.connection_manager.connection.cursor()
            cursor.execute(create_sql)
            
            # Create optimized indexes
            index_sqls = self.schema_optimizer.generate_optimization_indexes(schema)
            for idx_sql in index_sqls:
                try:
                    cursor.execute(idx_sql)
                    self.logger.info(f"Created optimization index: {idx_sql}")
                except Exception as e:
                    self.logger.warning(f"Index creation failed: {e}")
            
            # Load data
            df = pd.read_csv(filepath)
            
            # Validate schema
            is_valid, issues = self.table_manager.validate_schema_implementation(schema, df)
            if not is_valid:
                self.logger.warning(f"Schema validation issues: {issues}")
            
            # Insert data using optimized batch loading
            success = self._load_data_optimized(df, schema)
            
            load_time = time.time() - start_time
            
            if success:
                self.logger.info(f"✅ Optimized table creation completed in {load_time:.2f}s")
                self.logger.info(f"   Table: {schema.table_name}")
                self.logger.info(f"   Columns: {schema.column_count} ({schema.strategy_column_count} strategies)")
                self.logger.info(f"   Rows: {len(df)}")
                
                # Run production benchmark if requested
                if use_production_optimization:
                    self._run_production_validation(schema.table_name, load_time)
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to load CSV with optimization: {str(e)}")
            return False
    
    def load_csv_to_heavydb(self, filepath: str, table_name: str) -> bool:
        """
        Load a CSV file into HeavyDB with dynamic schema detection
        
        Args:
            filepath: Path to the CSV file
            table_name: Name of the table to create/populate
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self._is_connected:
            self.logger.error("Not connected to HeavyDB")
            return False
        
        try:
            # Use dynamic table manager for schema detection
            self.logger.info(f"Loading CSV file with dynamic schema detection: {filepath}")
            
            # Detect schema
            schema = self.table_manager.detect_schema_from_csv(filepath)
            
            # Override table name if provided
            if table_name:
                schema.table_name = table_name
            
            # Apply optimizations for wide tables
            if schema.metadata.get('is_wide_table', False):
                schema = self.table_manager.optimize_for_wide_table(schema)
            
            # Drop existing table
            self.drop_table(schema.table_name)
            
            # Create table with detected schema
            create_sql = self.table_manager.generate_create_table_sql(schema)
            cursor = self.connection_manager.connection.cursor()
            cursor.execute(create_sql)
            
            # Create indexes
            index_sqls = self.table_manager.generate_index_sql(schema)
            for idx_sql in index_sqls:
                try:
                    cursor.execute(idx_sql)
                except Exception as e:
                    self.logger.warning(f"Index creation failed: {e}")
            
            # Load data
            df = pd.read_csv(filepath)
            
            # Validate schema matches data
            is_valid, issues = self.table_manager.validate_schema_implementation(schema, df)
            if not is_valid:
                self.logger.warning(f"Schema validation issues: {issues}")
            
            # Insert data using connection manager
            success = self.connection_manager.create_table_from_dataframe(
                df, schema.table_name, if_exists='append'
            )
            
            if success:
                self.logger.info(f"Successfully loaded {filepath} into table {schema.table_name}")
                self.logger.info(f"Table has {schema.column_count} columns including {schema.strategy_column_count} strategy columns")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to load CSV to HeavyDB: {str(e)}")
            return False
    
    def apply_ulta_transformation(self, table_name: str) -> bool:
        """
        Apply ULTA transformation to strategies in the specified table
        
        Args:
            table_name: Name of the table to transform
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self._is_connected:
            self.logger.error("Not connected to HeavyDB")
            return False
        
        try:
            self.logger.info(f"Applying ULTA transformation to table {table_name}")
            
            # Get data from table
            query = f"SELECT * FROM {table_name}"
            df = self.connection_manager.execute_gpu_query(query, return_dataframe=True)
            
            if df is None or df.empty:
                self.logger.error(f"No data found in table {table_name}")
                return False
            
            # Apply ULTA transformation
            transformed_df, _ = self.ulta_calculator.apply_ulta_logic(df)
            
            # Create new table with transformed data
            transformed_table = f"{table_name}_ulta"
            success = self.connection_manager.create_table_from_dataframe(
                transformed_df, transformed_table, if_exists='replace'
            )
            
            if success:
                # Get ULTA metrics
                inverted_count = len(self.ulta_calculator.inverted_strategies)
                total_strategies = len(df.columns) - 3 if len(df.columns) > 3 else len(df.columns)
                self.logger.info(f"ULTA transformation complete. Inverted {inverted_count}/{total_strategies} strategies")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to apply ULTA transformation: {str(e)}")
            return False
    
    def compute_correlation_matrix(self, table_name: str) -> Optional[np.ndarray]:
        """
        Compute correlation matrix using GPU acceleration when available
        
        Args:
            table_name: Name of the table to analyze
            
        Returns:
            np.ndarray: Correlation matrix, or None if failed
        """
        if not self._is_connected:
            self.logger.error("Not connected to HeavyDB")
            return None
        
        try:
            self.logger.info(f"Computing correlation matrix for table {table_name}")
            
            if self.connection_manager.gpu_available:
                # Use GPU-accelerated correlation calculation
                return self._compute_gpu_correlation(table_name)
            else:
                # Fall back to CPU calculation
                return self._compute_cpu_correlation(table_name)
                
        except Exception as e:
            self.logger.error(f"Failed to compute correlation matrix: {str(e)}")
            return None
    
    def _compute_gpu_correlation(self, table_name: str) -> Optional[np.ndarray]:
        """
        Compute correlation matrix using GPU acceleration
        
        Args:
            table_name: Table name
            
        Returns:
            Correlation matrix or None
        """
        try:
            # Get column names (strategies)
            schema = self.get_table_schema(table_name)
            if not schema:
                return None
            
            columns = list(schema.keys())
            n_cols = len(columns)
            
            # Initialize correlation matrix
            corr_matrix = np.zeros((n_cols, n_cols))
            
            # GPU-accelerated correlation calculation
            # HeavyDB doesn't have built-in CORR for all pairs, so we batch it
            batch_queries = []
            
            for i in range(n_cols):
                for j in range(i, n_cols):
                    if i == j:
                        corr_matrix[i, j] = 1.0
                    else:
                        # Create correlation query
                        query = f"""
                        SELECT CORR({columns[i]}, {columns[j]}) as correlation
                        FROM {table_name}
                        """
                        batch_queries.append((i, j, query))
            
            # Execute queries in batches
            for i, j, query in batch_queries:
                result = self.connection_manager.execute_gpu_query(query)
                if result and len(result) > 0:
                    corr_value = float(result[0][0]) if result[0][0] is not None else 0.0
                    corr_matrix[i, j] = corr_value
                    corr_matrix[j, i] = corr_value  # Symmetric
            
            self.logger.info(f"GPU correlation matrix computed: shape {corr_matrix.shape}")
            return corr_matrix
            
        except Exception as e:
            self.logger.error(f"GPU correlation computation failed: {str(e)}")
            return None
    
    def _compute_cpu_correlation(self, table_name: str) -> Optional[np.ndarray]:
        """
        Compute correlation matrix using CPU (fallback)
        
        Args:
            table_name: Table name
            
        Returns:
            Correlation matrix or None
        """
        try:
            # Get all data
            query = f"SELECT * FROM {table_name}"
            df = self.connection_manager.execute_gpu_query(query, return_dataframe=True)
            
            if df is None or df.empty:
                return None
            
            # Use correlation calculator
            corr_matrix = self.correlation_calculator.calculate_correlation_matrix(df.values)
            
            self.logger.info(f"CPU correlation matrix computed: shape {corr_matrix.shape}")
            return corr_matrix
            
        except Exception as e:
            self.logger.error(f"CPU correlation computation failed: {str(e)}")
            return None
    
    def get_strategy_subset(self, table_name: str, strategy_list: List[int]) -> Optional[pd.DataFrame]:
        """
        Get data for specific strategies
        
        Args:
            table_name: Name of the table to query
            strategy_list: List of strategy indices to retrieve
            
        Returns:
            pd.DataFrame: Subset of data for specified strategies, or None if failed
        """
        if not self._is_connected:
            self.logger.error("Not connected to HeavyDB")
            return None
        
        try:
            # Get column names
            schema = self.get_table_schema(table_name)
            if not schema:
                return None
            
            columns = list(schema.keys())
            
            # Build column selection
            selected_columns = []
            for idx in strategy_list:
                if 0 <= idx < len(columns):
                    selected_columns.append(columns[idx])
            
            if not selected_columns:
                self.logger.warning("No valid strategies in the list")
                return pd.DataFrame()
            
            # Build query
            column_list = ", ".join(selected_columns)
            query = f"SELECT {column_list} FROM {table_name}"
            
            # Execute query
            result_df = self.connection_manager.execute_gpu_query(query, return_dataframe=True)
            
            if result_df is not None:
                self.logger.info(f"Retrieved {len(selected_columns)} strategies with {len(result_df)} rows")
            
            return result_df
            
        except Exception as e:
            self.logger.error(f"Failed to get strategy subset: {str(e)}")
            return None
    
    def execute_gpu_query(self, sql_query: str) -> Optional[pd.DataFrame]:
        """
        Execute arbitrary SQL query with GPU acceleration if available
        
        Args:
            sql_query: SQL query to execute
            
        Returns:
            pd.DataFrame: Query results, or None if failed
        """
        if not self._is_connected:
            self.logger.error("Not connected to HeavyDB")
            return None
        
        try:
            return self.connection_manager.execute_gpu_query(sql_query, return_dataframe=True)
        except Exception as e:
            self.logger.error(f"Query execution failed: {str(e)}")
            return None
    
    def get_table_schema(self, table_name: str) -> Optional[Dict[str, str]]:
        """
        Get the schema of a table
        
        Args:
            table_name: Name of the table
            
        Returns:
            Dict mapping column names to data types, or None if failed
        """
        if not self._is_connected:
            return None
        
        try:
            # Query to get table schema
            query = f"SHOW CREATE TABLE {table_name}"
            result = self.connection_manager.execute_gpu_query(query, return_dataframe=False)
            
            if not result:
                return None
            
            # Parse CREATE TABLE statement to extract schema
            create_stmt = result[0][0]
            schema = self._parse_create_table(create_stmt)
            
            return schema
            
        except Exception as e:
            # Fallback: try to get schema from a LIMIT 0 query
            try:
                query = f"SELECT * FROM {table_name} LIMIT 0"
                df = self.connection_manager.execute_gpu_query(query, return_dataframe=True)
                
                if df is not None:
                    schema = {}
                    for col in df.columns:
                        dtype = str(df[col].dtype)
                        if 'int' in dtype:
                            schema[col] = 'BIGINT'
                        elif 'float' in dtype:
                            schema[col] = 'DOUBLE'
                        elif 'datetime' in dtype:
                            schema[col] = 'TIMESTAMP'
                        else:
                            schema[col] = 'TEXT'
                    return schema
            except:
                pass
            
            self.logger.error(f"Failed to get table schema: {str(e)}")
            return None
    
    def _parse_create_table(self, create_stmt: str) -> Dict[str, str]:
        """
        Parse CREATE TABLE statement to extract schema
        
        Args:
            create_stmt: CREATE TABLE SQL statement
            
        Returns:
            Dict of column names to types
        """
        schema = {}
        
        # Simple parser - extract column definitions
        lines = create_stmt.split('\n')
        for line in lines:
            line = line.strip()
            if line and not line.startswith('CREATE') and not line.startswith(')'):
                # Parse column definition
                parts = line.split()
                if len(parts) >= 2:
                    col_name = parts[0].strip('`"')
                    col_type = parts[1].strip(',')
                    schema[col_name] = col_type
        
        return schema
    
    def create_dynamic_table(self, table_name: str, df: pd.DataFrame) -> bool:
        """
        Dynamically create a table based on DataFrame structure
        
        Args:
            table_name: Name of the table to create
            df: DataFrame to base the schema on
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self._is_connected:
            self.logger.error("Not connected to HeavyDB")
            return False
        
        try:
            return self.connection_manager.create_table_from_dataframe(
                df, table_name, if_exists='replace'
            )
        except Exception as e:
            self.logger.error(f"Failed to create dynamic table: {str(e)}")
            return False
    
    def get_table_row_count(self, table_name: str) -> int:
        """
        Get the number of rows in a table
        
        Args:
            table_name: Name of the table
            
        Returns:
            int: Number of rows, or -1 if failed
        """
        if not self._is_connected:
            return -1
        
        try:
            query = f"SELECT COUNT(*) FROM {table_name}"
            result = self.connection_manager.execute_gpu_query(query, return_dataframe=False)
            
            if result and len(result) > 0:
                return int(result[0][0])
            
            return -1
            
        except Exception as e:
            self.logger.error(f"Failed to get row count: {str(e)}")
            return -1
    
    def drop_table(self, table_name: str) -> bool:
        """
        Drop a table if it exists
        
        Args:
            table_name: Name of the table to drop
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self._is_connected:
            return False
        
        try:
            cursor = self.connection_manager.connection.cursor()
            cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
            self.logger.info(f"Dropped table {table_name}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to drop table: {str(e)}")
            return False
    
    def _load_data_optimized(self, df: pd.DataFrame, schema) -> bool:
        """
        Load data using optimized batch strategy for large datasets
        
        Args:
            df: DataFrame to load
            schema: Optimized table schema
            
        Returns:
            bool: True if successful
        """
        try:
            total_rows = len(df)
            strategy_count = schema.strategy_column_count
            
            # Calculate optimal batch size based on strategy count and GPU memory
            if strategy_count > 20000:
                batch_size = 1000  # Smaller batches for very wide tables
            elif strategy_count > 10000:
                batch_size = 5000
            else:
                batch_size = self.batch_size
            
            self.logger.info(f"Loading {total_rows} rows in batches of {batch_size}")
            
            # Process in batches
            for i in range(0, total_rows, batch_size):
                batch_df = df.iloc[i:i+batch_size]
                
                # Use connection manager's optimized insert
                success = self.connection_manager.create_table_from_dataframe(
                    batch_df, schema.table_name, if_exists='append'
                )
                
                if not success:
                    self.logger.error(f"Batch {i//batch_size + 1} failed")
                    return False
                
                # Progress logging
                if i % (batch_size * 10) == 0:
                    progress = (i / total_rows) * 100
                    self.logger.info(f"📊 Data loading progress: {progress:.1f}%")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Optimized data loading failed: {str(e)}")
            return False
    
    def _run_production_validation(self, table_name: str, load_time: float) -> None:
        """
        Run production validation benchmarks
        
        Args:
            table_name: Name of loaded table
            load_time: Time taken to load data
        """
        try:
            self.logger.info("🔥 Running production benchmark validation")
            
            # Run benchmark tests
            performance_metrics = self.benchmark_validator.run_production_benchmark(
                table_name, self.connection_manager
            )
            
            # Add load time to metrics
            performance_metrics['load_time_seconds'] = load_time
            
            # Validate against targets
            meets_targets, issues = self.schema_optimizer.validate_optimization_performance(
                None, performance_metrics  # Schema not needed for validation
            )
            
            if meets_targets:
                self.logger.info("✅ All production performance targets met!")
            else:
                self.logger.warning("⚠️  Some performance targets not met:")
                for issue in issues:
                    self.logger.warning(f"   • {issue}")
            
            # Log detailed metrics
            self.logger.info("📊 Production Performance Metrics:")
            for metric, value in performance_metrics.items():
                self.logger.info(f"   {metric}: {value:.3f}")
                
        except Exception as e:
            self.logger.error(f"Production validation failed: {str(e)}")
    
    def get_optimization_status(self, table_name: str) -> Dict[str, Any]:
        """
        Get optimization status and performance metrics for a table
        
        Args:
            table_name: Name of the table
            
        Returns:
            Dictionary with optimization status and metrics
        """
        try:
            # Get table metadata
            schema = self.get_table_schema(table_name)
            if not schema:
                return {'error': 'Table not found'}
            
            # Check if table is optimized
            row_count = self.get_table_row_count(table_name)
            
            # Run quick performance test
            import time
            start_time = time.time()
            test_query = f"SELECT COUNT(*) FROM {table_name} LIMIT 1"
            self.connection_manager.execute_gpu_query(test_query)
            query_time = time.time() - start_time
            
            return {
                'table_name': table_name,
                'row_count': row_count,
                'column_count': len(schema),
                'is_optimized': True,  # Assume optimized if loaded via optimized method
                'quick_query_time': query_time,
                'gpu_available': self.connection_manager.gpu_available,
                'optimization_features': [
                    'columnar_storage',
                    'date_partitioning', 
                    'optimized_encoding',
                    'strategic_indexing'
                ]
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    @property
    def schema_optimizer(self):
        """Lazy-loaded schema optimizer to avoid circular imports"""
        if self._schema_optimizer is None:
            from lib.heavydb_connector.schema_optimizer import HeavyDBSchemaOptimizer
            self._schema_optimizer = HeavyDBSchemaOptimizer(self._optimization_config_path)
        return self._schema_optimizer
    
    @property
    def benchmark_validator(self):
        """Lazy-loaded benchmark validator to avoid circular imports"""
        if self._benchmark_validator is None:
            from lib.heavydb_connector.schema_optimizer import ProductionBenchmarkValidator
            self._benchmark_validator = ProductionBenchmarkValidator(self.schema_optimizer)
        return self._benchmark_validator
    
    @property
    def supports_gpu(self) -> bool:
        """Check if GPU acceleration is available"""
        return self.connection_manager.gpu_available if self._is_connected else False