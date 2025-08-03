"""
ULTA (Ultra Low Trading Algorithm) Logic Calculator Module

This module extracts and reimplements the ULTA strategy inversion logic
from the legacy Heavy Optimizer Platform. It provides both in-memory
(pandas/numpy) and database (HeavyDB) implementations.

The ULTA logic inverts poorly performing strategies to potentially improve
their performance by reversing trading signals.
"""

import logging
from typing import Dict, Tuple, List, Optional, Any
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
import json
import time

# Try to import cuDF for GPU operations
try:
    import cudf
    import cupy as cp
    CUDF_AVAILABLE = True
except (ImportError, RuntimeError) as e:
    CUDF_AVAILABLE = False
    cudf = None
    cp = None


@dataclass
class ULTAStrategyMetrics:
    """Metrics for a strategy before and after ULTA inversion."""
    strategy_name: str
    original_roi: float
    inverted_roi: float
    original_drawdown: float
    inverted_drawdown: float
    original_ratio: float
    inverted_ratio: float
    improvement_percentage: float
    was_inverted: bool


class ULTACalculator:
    """
    ULTA (Ultra Low Trading Algorithm) Calculator
    
    Implements the core ULTA logic for strategy inversion based on ROI/Drawdown ratio.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None, config_path: Optional[str] = None):
        """
        Initialize ULTA Calculator with optional logger and configuration.
        
        Args:
            logger: Optional logger instance
            config_path: Optional path to configuration file
        """
        self.logger = logger or logging.getLogger(__name__)
        self.inverted_strategies: Dict[str, ULTAStrategyMetrics] = {}
        
        # Load configuration
        if config_path:
            from config.config_manager import get_config_manager
            self.config_manager = get_config_manager(config_path)
            self.ulta_config = self.config_manager.get_ulta_config()
        else:
            # Default configuration
            self.ulta_config = {
                'enabled': True,
                'roi_threshold': 0.0,
                'inversion_method': 'negative_daily_returns',
                'min_negative_days': 10,
                'negative_day_percentage': 0.6
            }
        
        self.logger.info(f"Initialized ULTA Calculator with config: {self.ulta_config}")
        
    def calculate_roi(self, returns: np.ndarray) -> float:
        """
        Calculate Return on Investment (ROI) from daily returns.
        
        Args:
            returns: Array of daily returns
            
        Returns:
            ROI as a float (e.g., 0.1 for 10% return)
        """
        # Legacy logic: ROI = sum of daily returns
        return float(np.sum(returns))
    
    def calculate_drawdown(self, returns: np.ndarray) -> float:
        """
        Calculate maximum drawdown from daily returns.
        
        Args:
            returns: Array of daily returns
            
        Returns:
            Maximum drawdown as a negative float
        """
        # Legacy logic: drawdown = minimum value in returns
        return float(np.min(returns))
    
    def calculate_ratio(self, roi: float, drawdown: float) -> float:
        """
        Calculate ROI/Drawdown ratio.
        
        Args:
            roi: Return on Investment
            drawdown: Maximum drawdown (negative value)
            
        Returns:
            ROI/Drawdown ratio (higher is better)
        """
        loss = abs(drawdown)
        return roi / loss if loss != 0 else float('inf')
    
    def invert_strategy(self, returns: np.ndarray) -> np.ndarray:
        """
        Invert a strategy by multiplying all returns by -1.
        
        Args:
            returns: Array of daily returns
            
        Returns:
            Inverted returns array
        """
        return -returns
    
    def should_invert_strategy(self, returns: np.ndarray) -> Tuple[bool, ULTAStrategyMetrics]:
        """
        Determine if a strategy should be inverted based on ULTA logic.
        
        Args:
            returns: Array of daily returns
            
        Returns:
            Tuple of (should_invert, metrics)
        """
        # Check if ULTA is enabled
        if not self.ulta_config['enabled']:
            return False, None
        
        # Calculate original metrics
        original_roi = self.calculate_roi(returns)
        original_drawdown = self.calculate_drawdown(returns)
        original_ratio = self.calculate_ratio(original_roi, original_drawdown)
        
        # Only consider inversion if ratio is below threshold
        roi_threshold = self.ulta_config['roi_threshold']
        if original_roi >= roi_threshold:
            return False, None
        
        # Check minimum negative days requirement
        negative_days = np.sum(returns < 0)
        min_negative_days = self.ulta_config['min_negative_days']
        negative_percentage = self.ulta_config['negative_day_percentage']
        
        if negative_days < min_negative_days:
            return False, None
        
        if len(returns) > 0 and (negative_days / len(returns)) < negative_percentage:
            return False, None
        
        # Calculate inverted metrics
        inverted_returns = self.invert_strategy(returns)
        inverted_roi = self.calculate_roi(inverted_returns)
        inverted_drawdown = self.calculate_drawdown(inverted_returns)
        inverted_ratio = self.calculate_ratio(inverted_roi, inverted_drawdown)
        
        # Decision: invert only if inverted ratio is better
        should_invert = inverted_ratio > original_ratio
        
        # Calculate improvement percentage
        if original_ratio != 0:
            improvement = ((inverted_ratio - original_ratio) / abs(original_ratio)) * 100
        else:
            improvement = float('inf') if inverted_ratio > 0 else 0
        
        metrics = ULTAStrategyMetrics(
            strategy_name="",  # Will be set by caller
            original_roi=original_roi,
            inverted_roi=inverted_roi,
            original_drawdown=original_drawdown,
            inverted_drawdown=inverted_drawdown,
            original_ratio=original_ratio,
            inverted_ratio=inverted_ratio,
            improvement_percentage=improvement,
            was_inverted=should_invert
        )
        
        return should_invert, metrics
    
    def apply_ulta_logic(self, data: pd.DataFrame, 
                        start_column: int = 3) -> Tuple[pd.DataFrame, Dict[str, ULTAStrategyMetrics]]:
        """
        Apply ULTA logic to a DataFrame of strategy returns.
        
        This method preserves the exact logic from the legacy implementation.
        
        Args:
            data: DataFrame with strategies as columns (columns 0-2 are metadata)
            start_column: Index of first strategy column (default: 3)
            
        Returns:
            Tuple of (processed_data, inverted_strategies_metrics)
        """
        self.inverted_strategies.clear()
        updated_data = data.copy()
        new_columns = {}
        columns_to_drop = []
        
        # Process each strategy column
        strategy_columns = data.columns[start_column:]
        
        for col in strategy_columns:
            try:
                # Convert to numeric array, handling non-numeric values
                arr = pd.to_numeric(data[col], errors='coerce').fillna(0).values
                
                # Check if strategy should be inverted
                should_invert, metrics = self.should_invert_strategy(arr)
                
                if should_invert and metrics:
                    # Create inverted column name
                    inv_col = f"{col}_inv"
                    
                    # Store inverted returns
                    new_columns[inv_col] = self.invert_strategy(arr)
                    
                    # Mark original column for removal
                    columns_to_drop.append(col)
                    
                    # Store metrics
                    metrics.strategy_name = col
                    self.inverted_strategies[col] = metrics
                    
                    self.logger.debug(
                        f"Inverted strategy {col}: ratio improved from "
                        f"{metrics.original_ratio:.2f} to {metrics.inverted_ratio:.2f}"
                    )
                    
            except Exception as e:
                self.logger.error(f"Error applying ULTA logic to column {col}: {e}")
                continue
        
        # Apply changes to DataFrame
        if columns_to_drop:
            updated_data.drop(columns=columns_to_drop, inplace=True)
            
        if new_columns:
            new_cols_df = pd.DataFrame(new_columns, index=updated_data.index)
            updated_data = pd.concat([updated_data, new_cols_df], axis=1)
        
        return updated_data, self.inverted_strategies
    
    def generate_inversion_report(self, output_format: str = "markdown", output_path: Optional[str] = None) -> str:
        """
        Generate a report of inverted strategies.
        
        Args:
            output_format: Output format ('markdown', 'json', or 'excel')
            output_path: Path to save the report (required for excel format)
            
        Returns:
            Formatted report string or file path for excel
        """
        if output_format == "json":
            return self._generate_json_report()
        elif output_format == "excel":
            if not output_path:
                raise ValueError("output_path is required for Excel format")
            return self._generate_excel_report(output_path)
        else:
            return self._generate_markdown_report()
    
    def _generate_markdown_report(self) -> str:
        """Generate markdown-formatted inversion report."""
        report_lines = [
            "# ULTA Inversion Report",
            "",
            f"Total strategies analyzed: {len(self.inverted_strategies)}",
            f"Strategies inverted: {sum(1 for m in self.inverted_strategies.values() if m.was_inverted)}",
            "",
            "## Inverted Strategies",
            "",
            "The following strategies had their returns inverted based on ULTA logic:",
            ""
        ]
        
        for strat, metrics in sorted(self.inverted_strategies.items()):
            if metrics.was_inverted:
                report_lines.extend([
                    f"### {strat}",
                    f"- **Original ROI:** {metrics.original_roi:.2f}",
                    f"- **Inverted ROI:** {metrics.inverted_roi:.2f}",
                    f"- **ROI Improvement:** {abs(metrics.inverted_roi - metrics.original_roi):.2f}",
                    f"- **Original Drawdown:** {metrics.original_drawdown:.2f}",
                    f"- **Inverted Drawdown:** {metrics.inverted_drawdown:.2f}",
                    f"- **Original Ratio:** {metrics.original_ratio:.2f}",
                    f"- **Inverted Ratio:** {metrics.inverted_ratio:.2f}",
                    f"- **Ratio Improvement:** {metrics.improvement_percentage:.1f}%",
                    ""
                ])
        
        return "\n".join(report_lines)
    
    def _generate_json_report(self) -> str:
        """Generate JSON-formatted inversion report."""
        report_data = {
            "summary": {
                "total_strategies_analyzed": len(self.inverted_strategies),
                "strategies_inverted": sum(1 for m in self.inverted_strategies.values() if m.was_inverted),
                "average_improvement": self._calculate_average_improvement()
            },
            "inverted_strategies": {
                name: asdict(metrics) 
                for name, metrics in self.inverted_strategies.items() 
                if metrics.was_inverted
            }
        }
        return json.dumps(report_data, indent=2)
    
    def _generate_excel_report(self, output_path: str) -> str:
        """Generate Excel-formatted comprehensive inversion report."""
        try:
            import pandas as pd
            
            # Create summary data
            total_analyzed = len(self.inverted_strategies)
            total_inverted = sum(1 for m in self.inverted_strategies.values() if m.was_inverted)
            avg_improvement = self._calculate_average_improvement()
            
            summary_data = {
                'Metric': [
                    'Total Strategies Analyzed',
                    'Strategies Inverted', 
                    'Inversion Rate (%)',
                    'Average Improvement (%)'
                ],
                'Value': [
                    total_analyzed,
                    total_inverted,
                    (total_inverted / max(1, total_analyzed)) * 100,
                    avg_improvement
                ]
            }
            summary_df = pd.DataFrame(summary_data)
            
            # Create detailed inversion data
            inverted_data = []
            for name, metrics in self.inverted_strategies.items():
                if metrics.was_inverted:
                    inverted_data.append({
                        'Strategy Name': name,
                        'Original ROI': metrics.original_roi,
                        'Inverted ROI': metrics.inverted_roi,
                        'ROI Improvement': abs(metrics.inverted_roi - metrics.original_roi),
                        'Original Drawdown': metrics.original_drawdown,
                        'Inverted Drawdown': metrics.inverted_drawdown,
                        'Original Ratio': metrics.original_ratio,
                        'Inverted Ratio': metrics.inverted_ratio,
                        'Improvement %': metrics.improvement_percentage
                    })
            
            if inverted_data:
                detailed_df = pd.DataFrame(inverted_data)
                # Sort by improvement percentage descending
                detailed_df = detailed_df.sort_values('Improvement %', ascending=False)
            else:
                detailed_df = pd.DataFrame({'Message': ['No strategies were inverted']})
            
            # Write to Excel with multiple sheets
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                detailed_df.to_excel(writer, sheet_name='Inverted Strategies', index=False)
                
                # Auto-adjust column widths
                for sheet_name in ['Summary', 'Inverted Strategies']:
                    worksheet = writer.sheets[sheet_name]
                    for column in worksheet.columns:
                        max_length = 0
                        column_letter = column[0].column_letter
                        for cell in column:
                            try:
                                if len(str(cell.value)) > max_length:
                                    max_length = len(str(cell.value))
                            except:
                                pass
                        adjusted_width = min(max_length + 2, 50)
                        worksheet.column_dimensions[column_letter].width = adjusted_width
            
            self.logger.info(f"Excel ULTA report saved to: {output_path}")
            return output_path
            
        except ImportError:
            self.logger.error("openpyxl not available for Excel export")
            raise ValueError("openpyxl package required for Excel export")
        except Exception as e:
            self.logger.error(f"Failed to generate Excel report: {e}")
            raise
    
    def _calculate_average_improvement(self) -> float:
        """Calculate average improvement percentage for inverted strategies."""
        improvements = [
            m.improvement_percentage 
            for m in self.inverted_strategies.values() 
            if m.was_inverted and m.improvement_percentage != float('inf')
        ]
        return float(np.mean(improvements)) if improvements else 0.0
    
    def get_inverted_strategy_names(self) -> List[str]:
        """Get list of strategy names that were inverted."""
        return [
            name for name, metrics in self.inverted_strategies.items() 
            if metrics.was_inverted
        ]
    
    def get_inversion_metrics(self, strategy_name: str) -> Optional[ULTAStrategyMetrics]:
        """Get metrics for a specific strategy."""
        return self.inverted_strategies.get(strategy_name)


class HeavyDBULTACalculator(ULTACalculator):
    """
    HeavyDB implementation of ULTA Calculator.
    
    This class extends ULTACalculator to work with HeavyDB for GPU-accelerated
    processing of large datasets.
    """
    
    def __init__(self, db_connection: Any, logger: Optional[logging.Logger] = None):
        """
        Initialize HeavyDB ULTA Calculator.
        
        Args:
            db_connection: HeavyDB connection object
            logger: Optional logger instance
        """
        super().__init__(logger)
        self.db_connection = db_connection
        
    def create_ulta_metadata_table(self, table_name: str = "ulta_inversions"):
        """
        Create metadata table to track ULTA inversions.
        
        Args:
            table_name: Name for the metadata table
        """
        query = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            strategy_name TEXT ENCODING DICT(32),
            original_roi DOUBLE,
            inverted_roi DOUBLE,
            original_drawdown DOUBLE,
            inverted_drawdown DOUBLE,
            original_ratio DOUBLE,
            inverted_ratio DOUBLE,
            improvement_percentage DOUBLE,
            was_inverted BOOLEAN
        )
        """
        self.db_connection.execute(query)
        self.logger.info(f"Created ULTA metadata table: {table_name}")
    
    def apply_ulta_to_table(self, 
                           input_table: str,
                           output_table: str,
                           metadata_table: str = "ulta_inversions",
                           batch_size: int = 100) -> Dict[str, ULTAStrategyMetrics]:
        """
        Apply ULTA logic to a HeavyDB table using GPU acceleration.
        
        Args:
            input_table: Name of input table with strategy data
            output_table: Name of output table for processed data
            metadata_table: Name of metadata table for tracking inversions
            batch_size: Number of strategies to process in each batch
            
        Returns:
            Dictionary of inversion metrics
        """
        # Get list of strategy columns
        strategy_columns = self._get_strategy_columns(input_table)
        
        # Create output table structure
        self._create_output_table(input_table, output_table)
        
        # Clear metadata table
        self.db_connection.execute(f"TRUNCATE TABLE {metadata_table}")
        
        # Process strategies in batches
        for i in range(0, len(strategy_columns), batch_size):
            batch_columns = strategy_columns[i:i + batch_size]
            self._process_strategy_batch(
                input_table, output_table, metadata_table, batch_columns
            )
        
        # Load inversion metrics from metadata table
        self._load_inversion_metrics(metadata_table)
        
        return self.inverted_strategies
    
    def _get_strategy_columns(self, table_name: str) -> List[str]:
        """Get list of strategy columns from table."""
        # This would use HeavyDB metadata queries to get column names
        # For now, returning placeholder
        query = f"""
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_name = '{table_name}'
        AND column_name LIKE 'strategy_%'
        """
        # Execute query and return results
        # Placeholder implementation
        return []
    
    def _create_output_table(self, input_table: str, output_table: str):
        """Create output table with same structure as input."""
        query = f"CREATE TABLE {output_table} AS SELECT * FROM {input_table} LIMIT 0"
        self.db_connection.execute(query)
    
    def _process_strategy_batch(self, 
                               input_table: str,
                               output_table: str,
                               metadata_table: str,
                               strategy_columns: List[str]):
        """Process a batch of strategies using GPU acceleration."""
        # This would contain the HeavyDB-specific implementation
        # using UPDATE queries with CASE statements as specified
        pass
    
    def _load_inversion_metrics(self, metadata_table: str):
        """Load inversion metrics from metadata table."""
        query = f"SELECT * FROM {metadata_table} WHERE was_inverted = true"
        # Execute query and populate self.inverted_strategies
        pass


class cuDFULTACalculator(ULTACalculator):
    """
    cuDF implementation of ULTA Calculator for GPU-accelerated processing.
    
    This class extends ULTACalculator to work with cuDF DataFrames for 
    high-performance processing of large strategy datasets.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None, config_path: Optional[str] = None, 
                 chunk_size: int = 10000):
        """
        Initialize cuDF ULTA Calculator.
        
        Args:
            logger: Optional logger instance
            config_path: Optional path to configuration file
            chunk_size: Number of strategies to process in each chunk for memory management
        """
        super().__init__(logger, config_path)
        
        if not CUDF_AVAILABLE:
            self.logger.info("💻 ULTA Calculator using CPU mode (cuDF not available)")
            self.use_gpu = False
        else:
            self.use_gpu = True
            self.logger.info("cuDF available, GPU acceleration enabled")
        
        self.chunk_size = chunk_size
        self.processing_stats = {
            'total_strategies': 0,
            'strategies_inverted': 0,
            'processing_time': 0.0,
            'gpu_memory_used': 0.0
        }
        
    def calculate_roi_cudf(self, returns: 'cudf.Series') -> float:
        """
        Calculate ROI using cuDF operations (GPU-accelerated).
        
        Args:
            returns: cuDF Series of daily returns
            
        Returns:
            ROI as a float
        """
        if not self.use_gpu:
            return self.calculate_roi(returns.to_pandas().values)
        
        # GPU-accelerated sum operation
        return float(returns.sum())
    
    def calculate_drawdown_cudf(self, returns: 'cudf.Series') -> float:
        """
        Calculate maximum drawdown using cuDF operations (GPU-accelerated).
        
        Args:
            returns: cuDF Series of daily returns
            
        Returns:
            Maximum drawdown as a negative float
        """
        if not self.use_gpu:
            return self.calculate_drawdown(returns.to_pandas().values)
        
        # GPU-accelerated min operation
        return float(returns.min())
    
    def calculate_ratio_cudf(self, roi: float, drawdown: float) -> float:
        """
        Calculate ROI/Drawdown ratio using GPU operations.
        
        Args:
            roi: Return on Investment
            drawdown: Maximum drawdown (negative value)
            
        Returns:
            ROI/Drawdown ratio
        """
        # Use cupy for GPU calculation if available
        if self.use_gpu and cp:
            roi_gpu = cp.asarray([roi])
            drawdown_gpu = cp.asarray([abs(drawdown)])
            
            # Avoid division by zero
            ratio_gpu = cp.where(drawdown_gpu != 0, roi_gpu / drawdown_gpu, cp.inf)
            return float(ratio_gpu[0])
        else:
            return self.calculate_ratio(roi, drawdown)
    
    def invert_strategy_cudf(self, returns: 'cudf.Series') -> 'cudf.Series':
        """
        Invert a strategy using cuDF operations (GPU-accelerated).
        
        Args:
            returns: cuDF Series of daily returns
            
        Returns:
            Inverted returns as cuDF Series
        """
        if not self.use_gpu:
            return cudf.Series(-returns.to_pandas().values)
        
        # GPU-accelerated inversion
        return -returns
    
    def should_invert_strategy_cudf(self, returns: 'cudf.Series', strategy_name: str = "") -> Tuple[bool, ULTAStrategyMetrics]:
        """
        Determine if a strategy should be inverted using cuDF operations.
        
        Args:
            returns: cuDF Series of daily returns
            strategy_name: Name of the strategy
            
        Returns:
            Tuple of (should_invert, metrics)
        """
        # Check if ULTA is enabled
        if not self.ulta_config['enabled']:
            return False, None
        
        # Calculate original metrics using GPU
        original_roi = self.calculate_roi_cudf(returns)
        original_drawdown = self.calculate_drawdown_cudf(returns)
        original_ratio = self.calculate_ratio_cudf(original_roi, original_drawdown)
        
        # ROI threshold check
        roi_threshold = self.ulta_config['roi_threshold']
        if original_roi >= roi_threshold:
            return False, None
        
        # Negative days calculation using cuDF
        if self.use_gpu:
            negative_days = int((returns < 0).sum())
            total_days = len(returns)
        else:
            # Fallback to pandas/numpy
            returns_np = returns.to_pandas().values
            negative_days = int(np.sum(returns_np < 0))
            total_days = len(returns_np)
        
        # Check thresholds
        min_negative_days = self.ulta_config['min_negative_days']
        negative_percentage = self.ulta_config['negative_day_percentage']
        
        if negative_days < min_negative_days:
            return False, None
        
        if total_days > 0 and (negative_days / total_days) < negative_percentage:
            return False, None
        
        # Calculate inverted metrics
        inverted_returns = self.invert_strategy_cudf(returns)
        inverted_roi = self.calculate_roi_cudf(inverted_returns)
        inverted_drawdown = self.calculate_drawdown_cudf(inverted_returns)
        inverted_ratio = self.calculate_ratio_cudf(inverted_roi, inverted_drawdown)
        
        # Decision: invert only if inverted ratio is better
        should_invert = inverted_ratio > original_ratio
        
        # Calculate improvement percentage
        if original_ratio != 0:
            improvement = ((inverted_ratio - original_ratio) / abs(original_ratio)) * 100
        else:
            improvement = float('inf') if inverted_ratio > 0 else 0
        
        metrics = ULTAStrategyMetrics(
            strategy_name=strategy_name,
            original_roi=original_roi,
            inverted_roi=inverted_roi,
            original_drawdown=original_drawdown,
            inverted_drawdown=inverted_drawdown,
            original_ratio=original_ratio,
            inverted_ratio=inverted_ratio,
            improvement_percentage=improvement,
            was_inverted=should_invert
        )
        
        return should_invert, metrics
    
    def apply_ulta_logic_cudf(self, data: 'cudf.DataFrame', 
                             start_column: int = 3) -> Tuple['cudf.DataFrame', Dict[str, ULTAStrategyMetrics]]:
        """
        Apply ULTA logic to a cuDF DataFrame using GPU acceleration.
        
        Args:
            data: cuDF DataFrame with strategies as columns
            start_column: Index of first strategy column
            
        Returns:
            Tuple of (processed_data, inverted_strategies_metrics)
        """
        start_time = time.time()
        self.inverted_strategies.clear()
        
        # Create a copy of the data
        if self.use_gpu:
            updated_data = data.copy()
        else:
            # Convert to pandas for processing
            pandas_data = data.to_pandas()
            return self._apply_ulta_logic_pandas_fallback(pandas_data, start_column)
        
        # Get strategy columns
        strategy_columns = data.columns[start_column:]
        self.processing_stats['total_strategies'] = len(strategy_columns)
        
        columns_to_drop = []
        new_columns = {}
        strategies_processed = 0
        
        # Process strategies in chunks for memory management
        for i in range(0, len(strategy_columns), self.chunk_size):
            chunk_columns = strategy_columns[i:i + self.chunk_size]
            
            for col in chunk_columns:
                try:
                    # Get strategy returns as cuDF Series
                    returns = data[col].fillna(0)
                    
                    # Check if strategy should be inverted
                    should_invert, metrics = self.should_invert_strategy_cudf(returns, str(col))
                    
                    if should_invert and metrics:
                        # Create inverted column name
                        inv_col = f"{col}_inv"
                        
                        # Store inverted returns
                        new_columns[inv_col] = self.invert_strategy_cudf(returns)
                        
                        # Mark original column for removal
                        columns_to_drop.append(col)
                        
                        # Store metrics
                        self.inverted_strategies[str(col)] = metrics
                        
                        self.logger.debug(
                            f"Inverted strategy {col}: ratio improved from "
                            f"{metrics.original_ratio:.2f} to {metrics.inverted_ratio:.2f}"
                        )
                        
                    strategies_processed += 1
                    
                    # Log progress for large datasets
                    if strategies_processed % 1000 == 0:
                        self.logger.info(f"Processed {strategies_processed}/{len(strategy_columns)} strategies")
                        
                except Exception as e:
                    self.logger.error(f"Error applying ULTA logic to column {col}: {e}")
                    continue
            
            # Memory management: trigger garbage collection between chunks
            if self.use_gpu and cp:
                cp.get_default_memory_pool().free_all_blocks()
        
        # Apply changes to DataFrame
        if columns_to_drop:
            updated_data = updated_data.drop(columns_to_drop)
            
        if new_columns:
            # Add new inverted columns
            for col_name, col_data in new_columns.items():
                updated_data[col_name] = col_data
        
        # Update processing stats
        self.processing_stats['strategies_inverted'] = len(self.inverted_strategies)
        self.processing_stats['processing_time'] = time.time() - start_time
        
        # Get GPU memory usage if available
        if self.use_gpu and cp:
            memory_pool = cp.get_default_memory_pool()
            self.processing_stats['gpu_memory_used'] = memory_pool.used_bytes() / (1024**3)  # GB
        
        self.logger.info(
            f"ULTA processing completed: {self.processing_stats['strategies_inverted']} "
            f"strategies inverted out of {self.processing_stats['total_strategies']} "
            f"in {self.processing_stats['processing_time']:.2f} seconds"
        )
        
        return updated_data, self.inverted_strategies
    
    def _apply_ulta_logic_pandas_fallback(self, data: pd.DataFrame, 
                                         start_column: int) -> Tuple['cudf.DataFrame', Dict[str, ULTAStrategyMetrics]]:
        """
        Fallback to pandas implementation when cuDF is not available.
        
        Args:
            data: Pandas DataFrame
            start_column: Index of first strategy column
            
        Returns:
            Tuple of (cuDF DataFrame, metrics)
        """
        self.logger.info("Using pandas fallback for ULTA processing")
        
        # Use parent class implementation
        processed_data, metrics = super().apply_ulta_logic(data, start_column)
        
        # Convert back to cuDF if available, otherwise keep as pandas
        if CUDF_AVAILABLE:
            try:
                cudf_data = cudf.from_pandas(processed_data)
                return cudf_data, metrics
            except Exception as e:
                self.logger.warning(f"Failed to convert back to cuDF: {e}")
                return processed_data, metrics
        else:
            return processed_data, metrics
    
    def apply_ulta_logic_batch(self, data_batches: List['cudf.DataFrame'], 
                              start_column: int = 3) -> Tuple[List['cudf.DataFrame'], Dict[str, ULTAStrategyMetrics]]:
        """
        Apply ULTA logic to multiple batches of data efficiently.
        
        Args:
            data_batches: List of cuDF DataFrames to process
            start_column: Index of first strategy column
            
        Returns:
            Tuple of (processed_batches, combined_metrics)
        """
        all_metrics = {}
        processed_batches = []
        
        for i, batch in enumerate(data_batches):
            self.logger.info(f"Processing batch {i+1}/{len(data_batches)}")
            
            processed_batch, batch_metrics = self.apply_ulta_logic_cudf(batch, start_column)
            processed_batches.append(processed_batch)
            all_metrics.update(batch_metrics)
        
        # Update combined inverted strategies
        self.inverted_strategies = all_metrics
        
        return processed_batches, all_metrics
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """
        Get processing statistics including performance metrics.
        
        Returns:
            Dictionary of processing statistics
        """
        stats = self.processing_stats.copy()
        stats['inversion_rate'] = (
            stats['strategies_inverted'] / max(1, stats['total_strategies']) * 100
        )
        stats['gpu_enabled'] = self.use_gpu
        stats['average_processing_time_per_strategy'] = (
            stats['processing_time'] / max(1, stats['total_strategies'])
        )
        
        return stats
    
    def generate_performance_report(self) -> str:
        """
        Generate a performance report for cuDF ULTA processing.
        
        Returns:
            Formatted performance report string
        """
        stats = self.get_processing_stats()
        
        report_lines = [
            "# cuDF ULTA Performance Report",
            "",
            f"**GPU Acceleration**: {'Enabled' if stats['gpu_enabled'] else 'Disabled'}",
            f"**Total Strategies Processed**: {stats['total_strategies']:,}",
            f"**Strategies Inverted**: {stats['strategies_inverted']:,}",
            f"**Inversion Rate**: {stats['inversion_rate']:.2f}%",
            f"**Total Processing Time**: {stats['processing_time']:.2f} seconds",
            f"**Avg Time per Strategy**: {stats['average_processing_time_per_strategy']*1000:.3f} ms",
            "",
            "## Performance Metrics",
            f"- **Throughput**: {stats['total_strategies']/max(0.001, stats['processing_time']):.0f} strategies/second",
        ]
        
        if stats['gpu_enabled']:
            report_lines.extend([
                f"- **GPU Memory Used**: {stats['gpu_memory_used']:.2f} GB",
                f"- **Memory per Strategy**: {stats['gpu_memory_used']*1024/max(1, stats['total_strategies']):.2f} MB",
            ])
        
        return "\n".join(report_lines)


# Convenience function for backward compatibility
def apply_ulta_logic(data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    """
    Apply ULTA logic to a DataFrame (backward compatibility wrapper).
    
    Args:
        data: DataFrame with strategy returns
        
    Returns:
        Tuple of (processed_data, inverted_strategies_dict)
    """
    calculator = ULTACalculator()
    processed_data, metrics = calculator.apply_ulta_logic(data)
    
    # Convert metrics to legacy format
    legacy_format = {}
    for name, metric in metrics.items():
        if metric.was_inverted:
            legacy_format[name] = {
                "original_roi": metric.original_roi,
                "inverted_roi": metric.inverted_roi,
                "original_drawdown": metric.original_drawdown,
                "inverted_drawdown": metric.inverted_drawdown,
                "original_ratio": metric.original_ratio,
                "inverted_ratio": metric.inverted_ratio
            }
    
    return processed_data, legacy_format