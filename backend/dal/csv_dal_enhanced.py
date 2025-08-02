"""
Enhanced CSV Data Access Layer Implementation

Uses the new enterprise CSV loader for efficient data loading with all
required features including streaming, validation, and progress tracking.
"""

import os
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
import configparser

# Add parent directory to path for imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dal.base_dal import BaseDAL
from dal.dynamic_table_manager import DynamicTableManager
from ulta_calculator import ULTACalculator
from lib.correlation.correlation_matrix_calculator import CorrelationMatrixCalculator
from lib.csv_loader import EnterpriseCSVLoader, BatchInsertHandler, LoadMetrics


class EnhancedCSVDAL(BaseDAL):
    """
    Enhanced CSV-based implementation of the Data Access Layer
    
    Uses enterprise CSV loader for efficient data operations with full
    feature support including streaming, batch inserts, and progress tracking.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Enhanced CSV DAL with configuration
        
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
        self.table_manager = DynamicTableManager(None)
        
        # Initialize enterprise CSV loader
        self.csv_loader = EnterpriseCSVLoader(
            chunk_size=config.get('csv_loader', {}).get('chunk_size', 10000),
            batch_size=config.get('csv_loader', {}).get('batch_size', 1000),
            enable_gpu_monitoring=config.get('csv_loader', {}).get('enable_gpu_monitoring', True),
            checkpoint_dir=config.get('csv_loader', {}).get('checkpoint_dir', '/tmp/csv_loader_checkpoints'),
            audit_log_path=config.get('csv_loader', {}).get('audit_log_path', '/mnt/optimizer_share/logs/csv_loader_audit.json')
        )
        
        # Initialize batch insert handler
        self.batch_handler = BatchInsertHandler(
            connection=None,  # No real DB connection for CSV DAL
            batch_size=config.get('csv_loader', {}).get('batch_size', 1000),
            use_gpu_transfer=config.get('csv_loader', {}).get('use_gpu_transfer', True)
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("Enhanced CSV DAL initialized with enterprise features")
    
    def _load_configuration(self) -> Dict[str, Any]:
        """
        Load configuration from config files
        
        Returns:
            Dict: Configuration dictionary
        """
        config = {
            'performance': {},
            'algorithms': {},
            'csv_loader': {
                'chunk_size': 10000,
                'batch_size': 1000,
                'enable_gpu_monitoring': True,
                'use_gpu_transfer': True,
                'checkpoint_dir': '/tmp/csv_loader_checkpoints',
                'audit_log_path': '/mnt/optimizer_share/logs/csv_loader_audit.json'
            }
        }
        
        # Try to load from production_config.ini
        config_path = '/mnt/optimizer_share/config/production_config.ini'
        if os.path.exists(config_path):
            parser = configparser.ConfigParser()
            parser.read(config_path)
            
            # CSV loader settings
            if 'CSV_LOADER' in parser:
                config['csv_loader'].update({
                    'chunk_size': parser.getint('CSV_LOADER', 'chunk_size', fallback=10000),
                    'batch_size': parser.getint('CSV_LOADER', 'batch_size', fallback=1000),
                    'enable_gpu_monitoring': parser.getboolean('CSV_LOADER', 'enable_gpu_monitoring', fallback=True)
                })
        
        return config
    
    def connect(self) -> bool:
        """
        Establish connection (always succeeds for CSV DAL)
        
        Returns:
            bool: Always True
        """
        self._is_connected = True
        self.logger.info("Enhanced CSV DAL connected")
        return True
    
    def disconnect(self) -> None:
        """Disconnect (clear in-memory tables)"""
        self.tables.clear()
        self._is_connected = False
        self.logger.info("Enhanced CSV DAL disconnected")
    
    def load_csv_to_heavydb(self, filepath: str, table_name: str,
                           progress_callback: Optional[Callable] = None) -> bool:
        """
        Load a CSV file using enterprise loader with all features
        
        Args:
            filepath: Path to the CSV file
            table_name: Name of the table to create/populate
            progress_callback: Optional callback for progress updates
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.logger.info(f"Loading CSV file with enterprise loader: {filepath}")
            
            # First validate the file
            validation_result = self.csv_loader.validate_csv_file(filepath)
            
            if not validation_result.is_valid:
                self.logger.error(f"CSV validation failed: {validation_result.errors}")
                return False
            
            if validation_result.warnings:
                self.logger.warning(f"CSV validation warnings: {validation_result.warnings}")
            
            # Prepare data handler for batch processing
            accumulated_chunks = []
            
            def chunk_handler(tbl_name: str, chunk: pd.DataFrame):
                """Handler to accumulate chunks"""
                accumulated_chunks.append(chunk)
            
            # Load CSV with streaming
            metrics = self.csv_loader.load_csv_streaming(
                file_path=filepath,
                table_name=table_name,
                data_handler=chunk_handler,
                resume_from_checkpoint=True,
                validate_data=True,
                fill_missing_values=True,
                missing_fill_value=0.0,
                progress_callback=progress_callback
            )
            
            # Combine all chunks
            if accumulated_chunks:
                df = pd.concat(accumulated_chunks, ignore_index=True)
                
                # Store in memory
                self.tables[table_name] = df
                
                # Log metrics
                self.logger.info(f"Load completed: {metrics.total_rows} rows in {metrics.load_time_seconds:.2f}s")
                self.logger.info(f"Throughput: {metrics.throughput_mbps:.2f} MB/s")
                self.logger.info(f"Peak memory: {metrics.peak_memory_mb:.1f} MB")
                
                if metrics.gpu_memory_used_mb > 0:
                    self.logger.info(f"GPU memory used: {metrics.gpu_memory_used_mb:.1f} MB")
                
                return True
            else:
                self.logger.error("No data loaded from CSV file")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to load CSV with enterprise loader: {str(e)}")
            return False
    
    def batch_insert_dataframe(self, table_name: str, df: pd.DataFrame,
                              progress_callback: Optional[Callable] = None) -> bool:
        """
        Insert DataFrame using batch handler
        
        Args:
            table_name: Target table name
            df: DataFrame to insert
            progress_callback: Optional progress callback
            
        Returns:
            bool: True if successful
        """
        try:
            # Use batch handler for efficient insertion
            metrics = self.batch_handler.insert_dataframe_in_batches(
                table_name=table_name,
                df=df,
                progress_callback=progress_callback
            )
            
            # Store in memory (for CSV DAL)
            if table_name in self.tables:
                # Append to existing table
                self.tables[table_name] = pd.concat([self.tables[table_name], df], 
                                                   ignore_index=True)
            else:
                # Create new table
                self.tables[table_name] = df.copy()
            
            self.logger.info(f"Batch insert completed: {metrics.total_rows_inserted} rows "
                           f"in {metrics.total_time_seconds:.2f}s "
                           f"({metrics.rows_per_second:.1f} rows/s)")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Batch insert failed: {e}")
            return False
    
    def get_load_metrics(self, limit: int = 10) -> List[LoadMetrics]:
        """
        Get recent load metrics from audit trail
        
        Args:
            limit: Number of recent entries to retrieve
            
        Returns:
            List of LoadMetrics objects
        """
        audit_entries = self.csv_loader.get_audit_trail(limit)
        
        metrics_list = []
        for entry in audit_entries:
            if 'metrics' in entry:
                metrics_dict = entry['metrics']
                metrics = LoadMetrics(**metrics_dict)
                metrics_list.append(metrics)
        
        return metrics_list
    
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
        Execute SQL-like query on in-memory data
        
        Enhanced version uses pandas query capabilities
        
        Args:
            sql_query: SQL query to execute
            
        Returns:
            pd.DataFrame: Query results, or None if failed
        """
        try:
            # For enhanced version, we could use pandasql or duckdb
            # For now, still use basic implementation
            return super().execute_gpu_query(sql_query)
            
        except Exception as e:
            self.logger.error(f"Query execution failed: {str(e)}")
            return None
    
    @property
    def supports_gpu(self) -> bool:
        """Enhanced CSV DAL supports GPU monitoring but not GPU queries"""
        return False
    
    def get_table_info(self, table_name: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a table
        
        Args:
            table_name: Name of the table
            
        Returns:
            Dictionary with table information
        """
        if table_name not in self.tables:
            return None
            
        df = self.tables[table_name]
        
        info = {
            'name': table_name,
            'rows': len(df),
            'columns': len(df.columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
            'dtypes': df.dtypes.to_dict(),
            'null_counts': df.isnull().sum().to_dict(),
            'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
            'datetime_columns': df.select_dtypes(include=['datetime']).columns.tolist()
        }
        
        return info