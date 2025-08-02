"""
Zone Optimizer DAL Integration

Extends ZoneOptimizer with database operations using the Data Access Layer.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from zone_optimizer import ZoneOptimizer, ZoneResult, CombinedZoneResult
from dal import get_dal
from dal.base_dal import BaseDAL


class ZoneOptimizerDAL(ZoneOptimizer):
    """
    Zone Optimizer with DAL integration for database operations
    
    Extends the base ZoneOptimizer with methods for:
    - Loading zone data from database
    - Filtering data by zone using SQL
    - Storing optimization results in database
    - GPU-accelerated operations when available
    """
    
    def __init__(self, dal: Optional[BaseDAL] = None, 
                 config_path: Optional[str] = None,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize Zone Optimizer with DAL
        
        Args:
            dal: Data Access Layer instance (creates one if None)
            config_path: Path to configuration file
            logger: Logger instance
        """
        super().__init__(config_path, logger)
        
        # Initialize DAL
        self.dal = dal or get_dal()
        if not self.dal.is_connected:
            self.dal.connect()
    
    def load_zone_data_from_table(self, table_name: str, 
                                 zone_column: str = "Zone") -> Dict[str, pd.DataFrame]:
        """
        Load zone-specific data from database table
        
        Args:
            table_name: Name of the table containing data
            zone_column: Name of the column containing zone information
            
        Returns:
            Dictionary mapping zone names to DataFrames
        """
        self.logger.info(f"Loading zone data from table '{table_name}'")
        
        zones = {}
        
        # Get unique zones using SQL
        if self.dal.supports_gpu:
            # Use SQL query for HeavyDB
            zone_query = f"SELECT DISTINCT {zone_column} FROM {table_name}"
            zone_df = self.dal.execute_gpu_query(zone_query)
            
            if zone_df is not None and not zone_df.empty:
                unique_zones = zone_df.iloc[:, 0].tolist()
            else:
                self.logger.error("Failed to retrieve zones from database")
                return zones
        else:
            # For CSV DAL, load full data and extract zones
            full_query = f"SELECT * FROM {table_name}"
            full_df = self.dal.execute_gpu_query(full_query)
            
            if full_df is None or full_df.empty:
                self.logger.error("Failed to load data from table")
                return zones
                
            unique_zones = full_df[zone_column].unique()
        
        # Load data for each configured zone
        for zone in unique_zones:
            zone_id = str(zone).lower().replace(" ", "").strip()
            
            if zone_id in self.config.zones:
                if self.dal.supports_gpu:
                    # Use SQL filtering for efficiency
                    zone_query = f"""
                    SELECT * FROM {table_name} 
                    WHERE {zone_column} = '{zone}'
                    """
                    zone_data = self.dal.execute_gpu_query(zone_query)
                else:
                    # Filter in memory for CSV DAL
                    zone_data = full_df[full_df[zone_column] == zone].copy()
                
                if zone_data is not None and not zone_data.empty:
                    zones[zone_id] = zone_data
                    self.logger.info(f"Loaded {len(zone_data)} rows for zone '{zone_id}'")
                else:
                    self.logger.warning(f"No data found for zone '{zone_id}'")
            else:
                self.logger.debug(f"Skipping unconfigured zone '{zone}'")
        
        return zones
    
    def optimize_zones_from_table(self, table_name: str,
                                 portfolio_size_per_zone: Optional[int] = None,
                                 algorithm: str = "genetic",
                                 save_results: bool = True,
                                 progress_callback: Optional[callable] = None,
                                 **kwargs) -> Dict[str, ZoneResult]:
        """
        Optimize zones directly from database table
        
        Args:
            table_name: Table containing the data
            portfolio_size_per_zone: Portfolio size for each zone
            algorithm: Optimization algorithm
            save_results: Whether to save results to database
            progress_callback: Progress callback function
            **kwargs: Additional algorithm parameters
            
        Returns:
            Zone optimization results
        """
        # Load zone data
        zone_data = self.load_zone_data_from_table(table_name)
        
        if not zone_data:
            raise ValueError(f"No valid zones found in table {table_name}")
        
        # Run optimization
        results = {}
        total_zones = len(zone_data)
        
        for idx, (zone_name, zone_df) in enumerate(zone_data.items()):
            if progress_callback:
                progress = (idx / total_zones) * 100
                progress_callback(progress, f"Optimizing zone: {zone_name}")
            
            try:
                # Check if we need to apply ULTA first
                if self.config.apply_ulta:
                    # Create temporary table for ULTA results
                    ulta_table = f"{table_name}_zone_{zone_name}_ulta"
                    
                    if self.dal.create_dynamic_table(ulta_table, zone_df):
                        if self.dal.apply_ulta_transformation(ulta_table):
                            # Load ULTA-transformed data
                            ulta_query = f"SELECT * FROM {ulta_table}"
                            zone_df = self.dal.execute_gpu_query(ulta_query)
                            
                            # Clean up temporary table
                            self.dal.drop_table(ulta_table)
                
                # Run optimization
                portfolio_size = portfolio_size_per_zone or self.config.min_portfolio_size
                result = self.optimize_single_zone(
                    zone_name, zone_df, portfolio_size, algorithm, **kwargs
                )
                results[zone_name] = result
                
                # Save results if requested
                if save_results:
                    self._save_zone_result_to_db(zone_name, result, table_name)
                    
            except Exception as e:
                self.logger.error(f"Failed to optimize zone {zone_name}: {str(e)}")
        
        if progress_callback:
            progress_callback(100, "Zone optimization complete")
        
        return results
    
    def _save_zone_result_to_db(self, zone_name: str, result: ZoneResult, 
                               source_table: str) -> bool:
        """
        Save zone optimization result to database
        
        Args:
            zone_name: Name of the zone
            result: Optimization result
            source_table: Source table name
            
        Returns:
            Success status
        """
        try:
            # Create results DataFrame
            result_data = pd.DataFrame([{
                'zone_name': zone_name,
                'fitness_score': result.fitness_score,
                'portfolio_size': len(result.portfolio_indices),
                'algorithm': result.algorithm_used,
                'optimization_time': result.optimization_time,
                'portfolio_indices': ','.join(map(str, result.portfolio_indices)),
                'portfolio_columns': ','.join(result.portfolio_columns),
                'source_table': source_table,
                'timestamp': pd.Timestamp.now()
            }])
            
            # Save to results table
            result_table = f"zone_optimization_results_{source_table}"
            success = self.dal.create_dynamic_table(result_table, result_data)
            
            if success:
                self.logger.info(f"Saved zone {zone_name} results to {result_table}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to save zone results: {str(e)}")
            return False
    
    def create_zone_filtered_view(self, source_table: str, zone_name: str,
                                 view_name: Optional[str] = None) -> bool:
        """
        Create a database view filtered by zone
        
        Args:
            source_table: Source table name
            zone_name: Zone to filter by
            view_name: Name for the view (auto-generated if None)
            
        Returns:
            Success status
        """
        if not self.dal.supports_gpu:
            self.logger.warning("View creation not supported in CSV DAL")
            return False
        
        try:
            view_name = view_name or f"{source_table}_zone_{zone_name}"
            
            create_view_sql = f"""
            CREATE VIEW {view_name} AS
            SELECT * FROM {source_table}
            WHERE Zone = '{zone_name}'
            """
            
            result = self.dal.execute_gpu_query(create_view_sql)
            
            if result is not None:
                self.logger.info(f"Created zone view: {view_name}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to create zone view: {str(e)}")
            return False
    
    def get_zone_statistics(self, table_name: str) -> pd.DataFrame:
        """
        Get statistics about zones in a table
        
        Args:
            table_name: Table to analyze
            
        Returns:
            DataFrame with zone statistics
        """
        try:
            if self.dal.supports_gpu:
                # Use SQL aggregation
                stats_query = f"""
                SELECT 
                    Zone,
                    COUNT(*) as row_count,
                    COUNT(DISTINCT Date) as unique_dates,
                    MIN(Date) as min_date,
                    MAX(Date) as max_date
                FROM {table_name}
                GROUP BY Zone
                ORDER BY Zone
                """
                return self.dal.execute_gpu_query(stats_query)
            else:
                # Load data and compute stats
                df = self.dal.execute_gpu_query(f"SELECT * FROM {table_name}")
                
                if df is not None and not df.empty:
                    stats = df.groupby('Zone').agg({
                        'Date': ['count', 'nunique', 'min', 'max']
                    }).reset_index()
                    
                    stats.columns = ['Zone', 'row_count', 'unique_dates', 'min_date', 'max_date']
                    return stats
                
            return pd.DataFrame()
            
        except Exception as e:
            self.logger.error(f"Failed to get zone statistics: {str(e)}")
            return pd.DataFrame()
    
    def combine_and_save_results(self, zone_results: Dict[str, ZoneResult],
                               output_table: str,
                               method: Optional[str] = None) -> CombinedZoneResult:
        """
        Combine zone results and save to database
        
        Args:
            zone_results: Individual zone results
            output_table: Table name for combined results
            method: Combination method
            
        Returns:
            Combined results
        """
        # Combine results
        combined = self.combine_zone_results(zone_results, method)
        
        # Create combined portfolio DataFrame
        portfolio_data = []
        
        for zone_name, result in zone_results.items():
            for idx, col in zip(result.portfolio_indices, result.portfolio_columns):
                portfolio_data.append({
                    'zone': zone_name,
                    'strategy_index': idx,
                    'strategy_column': col,
                    'zone_fitness': result.fitness_score,
                    'zone_weight': combined.zone_contributions.get(zone_name, 0),
                    'in_combined_portfolio': idx in combined.combined_portfolio
                })
        
        portfolio_df = pd.DataFrame(portfolio_data)
        
        # Save to database
        if self.dal.create_dynamic_table(output_table, portfolio_df):
            self.logger.info(f"Saved combined results to {output_table}")
        
        # Save summary
        summary_data = pd.DataFrame([{
            'combination_method': combined.combination_method,
            'combined_fitness': combined.combined_fitness,
            'combined_portfolio_size': len(combined.combined_portfolio),
            'num_zones': len(zone_results),
            'timestamp': pd.Timestamp.now()
        }])
        
        summary_table = f"{output_table}_summary"
        if self.dal.create_dynamic_table(summary_table, summary_data):
            self.logger.info(f"Saved summary to {summary_table}")
        
        return combined