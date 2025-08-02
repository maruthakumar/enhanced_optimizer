"""
CSV Data Access Layer Implementation

Provides a CSV-based fallback implementation of the DAL interface.
This allows the system to operate without HeavyDB, using pandas/numpy
for all data operations.
"""

import os
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import configparser

# Add parent directory to path for imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dal.base_dal import BaseDAL
from dal.dynamic_table_manager import DynamicTableManager
from ulta_calculator import ULTACalculator
from lib.correlation.correlation_matrix_calculator import CorrelationMatrixCalculator


class CSVDAL(BaseDAL):
    """
    CSV-based implementation of the Data Access Layer
    
    Uses pandas DataFrames as in-memory "tables" to provide the same
    interface as HeavyDB but without database dependencies.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize CSV DAL with configuration
        
        Args:
            config: Configuration dictionary. If None, loads from config files.
        """
        # Load configuration if not provided
        if config is None:
            config = self._load_configuration()
        
        super().__init__(config)
        
        # In-memory storage for "tables"
        self.tables: Dict[str, pd.DataFrame] = {}
        
        # Initialize helper modules
        self.ulta_calculator = ULTACalculator()
        self.correlation_calculator = CorrelationMatrixCalculator()
        # DynamicTableManager expects a config path, not a dict
        # Use None to let it use its default config loading
        self.table_manager = DynamicTableManager(None)
        
        # Performance settings
        self.batch_size = config.get('performance', {}).get('batch_size', 10000)
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("CSV DAL initialized (no database required)")
    
    def _load_configuration(self) -> Dict[str, Any]:
        """
        Load configuration from config files
        
        Returns:
            Dict: Configuration dictionary
        """
        config = {
            'performance': {},
            'algorithms': {}
        }
        
        # Try to load from production_config.ini
        config_path = '/mnt/optimizer_share/config/production_config.ini'
        if os.path.exists(config_path):
            parser = configparser.ConfigParser()
            parser.read(config_path)
            
            # Performance settings
            if 'performance' in parser:
                config['performance'] = {
                    'batch_size': parser.getint('performance', 'batch_size', fallback=10000)
                }
        
        return config
    
    def connect(self) -> bool:
        """
        Establish connection (always succeeds for CSV DAL)
        
        Returns:
            bool: Always True
        """
        self._is_connected = True
        self.logger.info("CSV DAL connected (no database connection required)")
        return True
    
    def disconnect(self) -> None:
        """Disconnect (clear in-memory tables)"""
        self.tables.clear()
        self._is_connected = False
        self.logger.info("CSV DAL disconnected")
    
    def load_csv_to_heavydb(self, filepath: str, table_name: str) -> bool:
        """
        Load a CSV file into in-memory "table"
        
        Args:
            filepath: Path to the CSV file
            table_name: Name of the table to create/populate
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.logger.info(f"Loading CSV file with schema detection: {filepath}")
            
            # Detect schema using dynamic table manager
            schema = self.table_manager.detect_schema_from_csv(filepath)
            
            # Override table name if provided
            if table_name:
                schema.table_name = table_name
            
            # Read CSV file
            df = pd.read_csv(filepath)
            
            # Validate schema
            is_valid, issues = self.table_manager.validate_schema_implementation(schema, df)
            if not is_valid:
                self.logger.warning(f"Schema validation issues: {issues}")
            
            # Store in memory
            self.tables[schema.table_name] = df
            
            # Store schema metadata
            self.tables[f"_schema_{schema.table_name}"] = pd.DataFrame([{
                'column_count': schema.column_count,
                'strategy_count': schema.strategy_column_count,
                'is_wide_table': schema.metadata.get('is_wide_table', False),
                'indexes': ','.join(schema.indexes),
                'source_file': filepath
            }])
            
            self.logger.info(f"Loaded {len(df)} rows and {schema.column_count} columns into table '{schema.table_name}'")
            if schema.metadata.get('is_wide_table'):
                self.logger.info(f"Wide table detected with {schema.strategy_column_count} strategy columns")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load CSV: {str(e)}")
            return False
    
    def apply_ulta_transformation(self, table_name: str) -> bool:
        """
        Apply ULTA transformation to strategies in the specified table
        
        Args:
            table_name: Name of the table to transform
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if table_name not in self.tables:
                self.logger.error(f"Table '{table_name}' not found")
                return False
            
            self.logger.info(f"Applying ULTA transformation to table {table_name}")
            
            # Get data
            df = self.tables[table_name].copy()
            
            # Apply ULTA transformation
            transformed_df, _ = self.ulta_calculator.apply_ulta_logic(df)
            
            # Store transformed data
            transformed_table = f"{table_name}_ulta"
            self.tables[transformed_table] = transformed_df
            
            # Get ULTA metrics
            inverted_count = len(self.ulta_calculator.inverted_strategies)
            
            total_strategies = len(df.columns) - 3 if len(df.columns) > 3 else len(df.columns)
            self.logger.info(f"ULTA transformation complete. Inverted {inverted_count}/{total_strategies} strategies")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to apply ULTA transformation: {str(e)}")
            return False
    
    def compute_correlation_matrix(self, table_name: str) -> Optional[np.ndarray]:
        """
        Compute correlation matrix for the specified table
        
        Args:
            table_name: Name of the table to analyze
            
        Returns:
            np.ndarray: Correlation matrix, or None if failed
        """
        try:
            if table_name not in self.tables:
                self.logger.error(f"Table '{table_name}' not found")
                return None
            
            self.logger.info(f"Computing correlation matrix for table {table_name}")
            
            # Get data
            df = self.tables[table_name]
            
            # Use correlation calculator
            corr_matrix = self.correlation_calculator.calculate_correlation_matrix(df.values)
            
            self.logger.info(f"Correlation matrix computed: shape {corr_matrix.shape}")
            
            return corr_matrix
            
        except Exception as e:
            self.logger.error(f"Failed to compute correlation matrix: {str(e)}")
            return None
    
    def get_strategy_subset(self, table_name: str, strategy_list: List[int]) -> Optional[pd.DataFrame]:
        """
        Return a subset of the data for a given list of strategies
        
        Args:
            table_name: Name of the table to query
            strategy_list: List of strategy indices to retrieve
            
        Returns:
            pd.DataFrame: Subset of data for specified strategies, or None if failed
        """
        try:
            if table_name not in self.tables:
                self.logger.error(f"Table '{table_name}' not found")
                return None
            
            df = self.tables[table_name]
            columns = df.columns.tolist()
            
            # Select columns based on strategy indices
            selected_columns = []
            for idx in strategy_list:
                if 0 <= idx < len(columns):
                    selected_columns.append(columns[idx])
            
            if not selected_columns:
                self.logger.warning("No valid strategies in the list")
                return pd.DataFrame()
            
            # Return subset
            subset_df = df[selected_columns].copy()
            self.logger.info(f"Retrieved {len(selected_columns)} strategies with {len(subset_df)} rows")
            
            return subset_df
            
        except Exception as e:
            self.logger.error(f"Failed to get strategy subset: {str(e)}")
            return None
    
    def execute_gpu_query(self, sql_query: str) -> Optional[pd.DataFrame]:
        """
        Execute SQL-like query on in-memory data (limited support)
        
        Note: This is a simplified implementation that only supports basic SELECT queries.
        For full SQL support, consider using SQLite or DuckDB.
        
        Args:
            sql_query: SQL query to execute
            
        Returns:
            pd.DataFrame: Query results, or None if failed
        """
        try:
            self.logger.warning("CSV DAL has limited SQL support. Complex queries may fail.")
            
            # Very basic SQL parsing (only handles simple SELECT)
            sql_upper = sql_query.upper().strip()
            
            if sql_upper.startswith("SELECT"):
                # Extract table name (very simple parser)
                from_idx = sql_upper.find("FROM")
                if from_idx == -1:
                    self.logger.error("No FROM clause found in query")
                    return None
                
                # Get table name
                remaining = sql_query[from_idx + 4:].strip()
                table_name = remaining.split()[0].strip('";')
                
                if table_name not in self.tables:
                    self.logger.error(f"Table '{table_name}' not found")
                    return None
                
                # For now, return entire table (ignoring WHERE, GROUP BY, etc.)
                # A real implementation would need a proper SQL parser
                return self.tables[table_name].copy()
            
            else:
                self.logger.error(f"Unsupported SQL operation: {sql_upper[:20]}...")
                return None
                
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
        try:
            if table_name not in self.tables:
                self.logger.error(f"Table '{table_name}' not found")
                return None
            
            df = self.tables[table_name]
            schema = {}
            
            for col in df.columns:
                dtype = str(df[col].dtype)
                if 'int' in dtype:
                    schema[col] = 'INTEGER'
                elif 'float' in dtype:
                    schema[col] = 'DOUBLE'
                elif 'datetime' in dtype:
                    schema[col] = 'TIMESTAMP'
                elif 'bool' in dtype:
                    schema[col] = 'BOOLEAN'
                else:
                    schema[col] = 'TEXT'
            
            return schema
            
        except Exception as e:
            self.logger.error(f"Failed to get table schema: {str(e)}")
            return None
    
    def create_dynamic_table(self, table_name: str, df: pd.DataFrame) -> bool:
        """
        Create a table from DataFrame
        
        Args:
            table_name: Name of the table to create
            df: DataFrame to store
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.tables[table_name] = df.copy()
            self.logger.info(f"Created table '{table_name}' with {len(df)} rows and {len(df.columns)} columns")
            return True
        except Exception as e:
            self.logger.error(f"Failed to create table: {str(e)}")
            return False
    
    def get_table_row_count(self, table_name: str) -> int:
        """
        Get the number of rows in a table
        
        Args:
            table_name: Name of the table
            
        Returns:
            int: Number of rows, or -1 if failed
        """
        try:
            if table_name not in self.tables:
                self.logger.error(f"Table '{table_name}' not found")
                return -1
            
            return len(self.tables[table_name])
            
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
        try:
            if table_name in self.tables:
                del self.tables[table_name]
                self.logger.info(f"Dropped table '{table_name}'")
            else:
                self.logger.warning(f"Table '{table_name}' does not exist")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to drop table: {str(e)}")
            return False
    
    @property
    def supports_gpu(self) -> bool:
        """CSV DAL does not support GPU acceleration"""
        return False
    
    def list_tables(self) -> List[str]:
        """
        List all available tables
        
        Returns:
            List of table names
        """
        return list(self.tables.keys())
    
    def save_table_to_csv(self, table_name: str, filepath: str) -> bool:
        """
        Save a table to CSV file
        
        Args:
            table_name: Name of the table to save
            filepath: Output file path
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if table_name not in self.tables:
                self.logger.error(f"Table '{table_name}' not found")
                return False
            
            self.tables[table_name].to_csv(filepath, index=False)
            self.logger.info(f"Saved table '{table_name}' to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save table to CSV: {str(e)}")
            return False