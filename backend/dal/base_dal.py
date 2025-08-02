"""
Base Data Access Layer (DAL) abstract class

Defines the interface that all DAL implementations must follow.
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple


class BaseDAL(ABC):
    """
    Abstract base class for Data Access Layer implementations
    
    This class defines the interface that all DAL implementations must follow,
    whether they're using HeavyDB, CSV files, or any other data source.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the DAL with configuration
        
        Args:
            config: Configuration dictionary containing connection parameters,
                   memory limits, batch sizes, etc.
        """
        self.config = config
        self._is_connected = False
    
    @abstractmethod
    def connect(self) -> bool:
        """
        Establish connection to the data source
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        """
        Close connection to the data source
        """
        pass
    
    @abstractmethod
    def load_csv_to_heavydb(self, filepath: str, table_name: str) -> bool:
        """
        Load a CSV file into a specified table
        
        Args:
            filepath: Path to the CSV file
            table_name: Name of the table to create/populate
            
        Returns:
            bool: True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def apply_ulta_transformation(self, table_name: str) -> bool:
        """
        Execute ULTA (Ultra Low Trading Activity) transformation on the specified table
        
        Args:
            table_name: Name of the table to transform
            
        Returns:
            bool: True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def compute_correlation_matrix(self, table_name: str) -> Optional[np.ndarray]:
        """
        Compute and return the correlation matrix for the specified table
        
        Args:
            table_name: Name of the table to analyze
            
        Returns:
            np.ndarray: Correlation matrix, or None if failed
        """
        pass
    
    @abstractmethod
    def get_strategy_subset(self, table_name: str, strategy_list: List[int]) -> Optional[pd.DataFrame]:
        """
        Return a subset of the data for a given list of strategies
        
        Args:
            table_name: Name of the table to query
            strategy_list: List of strategy indices to retrieve
            
        Returns:
            pd.DataFrame: Subset of data for specified strategies, or None if failed
        """
        pass
    
    @abstractmethod
    def execute_gpu_query(self, sql_query: str) -> Optional[pd.DataFrame]:
        """
        Execute an arbitrary SQL query with GPU acceleration if available
        
        Args:
            sql_query: SQL query to execute
            
        Returns:
            pd.DataFrame: Query results, or None if failed
        """
        pass
    
    @abstractmethod
    def get_table_schema(self, table_name: str) -> Optional[Dict[str, str]]:
        """
        Get the schema of a table (column names and types)
        
        Args:
            table_name: Name of the table
            
        Returns:
            Dict mapping column names to data types, or None if failed
        """
        pass
    
    @abstractmethod
    def create_dynamic_table(self, table_name: str, df: pd.DataFrame) -> bool:
        """
        Dynamically create a table based on DataFrame structure
        
        Args:
            table_name: Name of the table to create
            df: DataFrame to base the schema on
            
        Returns:
            bool: True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def get_table_row_count(self, table_name: str) -> int:
        """
        Get the number of rows in a table
        
        Args:
            table_name: Name of the table
            
        Returns:
            int: Number of rows, or -1 if failed
        """
        pass
    
    @abstractmethod
    def drop_table(self, table_name: str) -> bool:
        """
        Drop a table if it exists
        
        Args:
            table_name: Name of the table to drop
            
        Returns:
            bool: True if successful, False otherwise
        """
        pass
    
    @property
    def is_connected(self) -> bool:
        """Check if DAL is connected to data source"""
        return self._is_connected
    
    @property
    @abstractmethod
    def supports_gpu(self) -> bool:
        """Check if this DAL implementation supports GPU acceleration"""
        pass
    
    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.disconnect()