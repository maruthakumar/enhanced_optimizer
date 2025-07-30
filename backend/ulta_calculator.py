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
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize ULTA Calculator with optional logger."""
        self.logger = logger or logging.getLogger(__name__)
        self.inverted_strategies: Dict[str, ULTAStrategyMetrics] = {}
        
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
        # Calculate original metrics
        original_roi = self.calculate_roi(returns)
        original_drawdown = self.calculate_drawdown(returns)
        original_ratio = self.calculate_ratio(original_roi, original_drawdown)
        
        # Only consider inversion if ratio is negative
        if original_ratio >= 0:
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
    
    def generate_inversion_report(self, output_format: str = "markdown") -> str:
        """
        Generate a report of inverted strategies.
        
        Args:
            output_format: Output format ('markdown' or 'json')
            
        Returns:
            Formatted report string
        """
        if output_format == "json":
            return self._generate_json_report()
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