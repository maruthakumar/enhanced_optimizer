#!/usr/bin/env python3
"""
Zone-Based Optimizer for HeavyDB Integration
Handles intraday zone-based portfolio optimization with consolidated data
"""

import os
import sys
import time
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lib.heavydb_connector import get_connection, execute_query, load_strategy_data
from lib.correlation.correlation_matrix_calculator import CorrelationMatrixCalculator
from ulta_calculator import ULTACalculator
from dal import get_dal

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


@dataclass
class IntradayZoneConfiguration:
    """Configuration for intraday zone-based optimization"""
    enabled: bool = True
    zones: List[str] = field(default_factory=lambda: ["Zone 1", "Zone 2", "Zone 3", "Zone 4"])
    zone_weights: Dict[str, float] = field(default_factory=dict)
    market_open_time: str = "09:15:00"
    market_close_time: str = "15:30:00"
    zone_method: str = "equal"  # equal, freeform, custom
    num_zones: int = 4
    apply_ulta: bool = True
    min_portfolio_size: int = 10
    max_portfolio_size: int = 50
    
    def __post_init__(self):
        """Initialize zone weights if not provided"""
        if not self.zone_weights:
            # Equal weights for all zones by default
            weight = 1.0 / len(self.zones)
            self.zone_weights = {zone: weight for zone in self.zones}


@dataclass
class ZoneOptimizationResult:
    """Result of zone-based optimization"""
    portfolio_indices: List[int]
    portfolio_columns: List[str]
    fitness_score: float
    zone_performance: Dict[str, float]  # Performance by zone
    weighted_performance: float
    metrics: Dict[str, Any]
    optimization_time: float
    algorithm_used: str
    zone_matrix_shape: Tuple[int, int, int]  # (dates, zones, strategies)
    metadata: Dict[str, Any] = field(default_factory=dict)


class IntradayZoneOptimizer:
    """
    Optimizer for intraday zone-based portfolio selection
    Expects consolidated data with Zone column from Strategy_consolidator.py
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        self.config = self._load_configuration(config_path)
        
        # Initialize components
        self.ulta_calculator = ULTACalculator() if self.config.apply_ulta else None
        self.correlation_calculator = CorrelationMatrixCalculator()
        self.dal = get_dal()
        
        # Cache for performance
        self._zone_matrix_cache = None
        self._correlation_cache = {}
        
    def _load_configuration(self, config_path: Optional[str]) -> IntradayZoneConfiguration:
        """Load configuration from file or use defaults"""
        config = IntradayZoneConfiguration()
        
        if config_path and os.path.exists(config_path):
            import configparser
            cfg = configparser.ConfigParser()
            cfg.read(config_path)
            
            if 'ZONE_OPTIMIZATION' in cfg:
                zone_cfg = cfg['ZONE_OPTIMIZATION']
                config.enabled = zone_cfg.getboolean('enabled', True)
                config.zone_method = zone_cfg.get('zone_method', 'equal')
                config.num_zones = zone_cfg.getint('num_zones', 4)
                config.apply_ulta = zone_cfg.getboolean('apply_ulta', True)
                
                # Parse zone weights
                if 'zone_weights' in zone_cfg:
                    weights = [float(w.strip()) for w in zone_cfg['zone_weights'].split(',')]
                    if len(weights) == config.num_zones:
                        config.zone_weights = {f"Zone {i+1}": w for i, w in enumerate(weights)}
        
        self.logger.info(f"Zone configuration: {config.num_zones} zones, weights: {config.zone_weights}")
        return config
    
    def build_zone_matrix(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """
        Build 3D zone matrix from consolidated dataframe
        Returns: (zone_matrix, strategy_columns)
        where zone_matrix shape is (dates, zones, strategies)
        """
        # Validate Zone column exists
        if 'Zone' not in df.columns:
            raise ValueError("DataFrame must contain 'Zone' column. Use Strategy_consolidator.py to create zones.")
        
        # Clean zone data
        df = df.copy()
        df["Zone"] = df["Zone"].astype(str)
        df = df[~df["Zone"].isin({"Outside Market", "nan", ""})]
        df.fillna(0, inplace=True)
        
        # Identify columns
        group_cols = ["Date", "Zone", "Day"]
        if "DAY" in df.columns:
            group_cols = ["Date", "Zone", "DAY"]
        
        df.sort_values(group_cols, inplace=True)
        strategy_cols = [col for col in df.columns if col not in group_cols]
        
        # Get unique zones and dates
        all_zones = sorted(df["Zone"].unique())
        all_dates = sorted(df["Date"].unique())
        
        self.logger.info(f"Building zone matrix: {len(all_dates)} dates × {len(all_zones)} zones × {len(strategy_cols)} strategies")
        
        # Apply ULTA if configured
        if self.config.apply_ulta and self.ulta_calculator:
            self.logger.info("Applying ULTA logic to zone data")
            df, ulta_metrics = self.ulta_calculator.apply_ulta_logic(df)
            inverted_count = len([m for m in ulta_metrics.values() if m.was_inverted])
            self.logger.info(f"ULTA applied: {inverted_count} strategies inverted")
            
            # Update strategy columns after ULTA (new columns may have been added)
            strategy_cols = [col for col in df.columns if col not in group_cols]
        
        # Build 3D matrix
        zone_matrix_list = []
        
        for date in all_dates:
            date_data = df[df["Date"] == date]
            zone_data_array = np.zeros((len(all_zones), len(strategy_cols)), dtype=float)
            
            for idx, zone in enumerate(all_zones):
                zone_group = date_data[date_data["Zone"] == zone]
                if not zone_group.empty:
                    # Sum values for this zone (handles multiple entries per zone)
                    zone_values = zone_group[strategy_cols].sum().values.astype(float)
                    zone_data_array[idx, :] = zone_values
            
            zone_matrix_list.append(zone_data_array)
        
        # Stack into 3D array: (dates, zones, strategies)
        zone_matrix = np.stack(zone_matrix_list, axis=0)
        
        self.logger.info(f"Zone matrix built: shape {zone_matrix.shape}")
        self._zone_matrix_cache = zone_matrix
        
        return zone_matrix, strategy_cols
    
    def evaluate_zone_fitness(self, individual: List[int], zone_matrix: np.ndarray, 
                            metric: str = "ratio", corr_matrix: Optional[np.ndarray] = None,
                            zone_weights: Optional[np.ndarray] = None,
                            drawdown_threshold: float = 0) -> float:
        """
        Evaluate fitness for zone-based optimization
        
        Args:
            individual: List of strategy indices
            zone_matrix: 3D array (dates, zones, strategies)
            metric: Optimization metric ('ratio', 'roi', 'sharpe')
            corr_matrix: Correlation matrix for diversification
            zone_weights: Weights for each zone
            drawdown_threshold: Maximum allowed drawdown
        """
        # Extract returns for selected strategies
        selected_returns = zone_matrix[:, :, individual]  # (dates, zones, portfolio_size)
        
        # Average returns across strategies for each date/zone
        avg_returns = np.mean(selected_returns, axis=2)  # (dates, zones)
        
        # Apply zone weights
        num_zones = avg_returns.shape[1]
        if zone_weights is None or len(zone_weights) != num_zones:
            zone_weights = np.ones(num_zones) / num_zones
        else:
            zone_weights = zone_weights / np.sum(zone_weights)
        
        # Weighted returns across zones for each date
        weighted_returns = np.dot(avg_returns, zone_weights)  # (dates,)
        
        # Calculate fitness metrics
        roi = np.sum(weighted_returns)
        
        # Calculate drawdown
        equity_curve = np.cumsum(weighted_returns)
        running_max = np.maximum.accumulate(equity_curve)
        drawdown = running_max - equity_curve
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0
        
        # Apply drawdown threshold
        if drawdown_threshold > 0 and max_drawdown > drawdown_threshold:
            return -np.inf
        
        # Calculate win rate
        win_rate = np.sum(weighted_returns > 0) / len(weighted_returns) if len(weighted_returns) > 0 else 0
        
        # Calculate fitness based on metric
        if metric == "ratio":
            # ROI/Drawdown ratio
            fitness = roi / (max_drawdown + 1e-6)
        elif metric == "roi":
            fitness = roi
        elif metric == "sharpe":
            # Sharpe-like ratio
            returns_std = np.std(weighted_returns)
            fitness = roi / (returns_std + 1e-6) if returns_std > 0 else roi
        else:
            fitness = roi / (max_drawdown + 1e-6)
        
        # Apply correlation penalty if provided
        if corr_matrix is not None and len(individual) > 1:
            # Calculate average correlation
            selected_corr = corr_matrix[np.ix_(individual, individual)]
            np.fill_diagonal(selected_corr, 0)
            avg_correlation = np.mean(np.abs(selected_corr))
            
            # Penalize high correlation
            correlation_penalty = 1.0 - (avg_correlation * 0.1)  # 10% penalty per unit correlation
            fitness *= correlation_penalty
        
        return fitness
    
    def optimize_zone_portfolio(self, zone_matrix: np.ndarray, strategy_columns: List[str],
                              portfolio_size: int, algorithm: str = "genetic",
                              **kwargs) -> ZoneOptimizationResult:
        """
        Optimize portfolio using zone-based approach
        
        Args:
            zone_matrix: 3D array (dates, zones, strategies)
            strategy_columns: List of strategy names
            portfolio_size: Target portfolio size
            algorithm: Algorithm to use
            **kwargs: Algorithm-specific parameters
        """
        start_time = time.time()
        
        num_dates, num_zones, num_strategies = zone_matrix.shape
        self.logger.info(f"Optimizing zone portfolio: {num_strategies} strategies, {portfolio_size} target size")
        
        # Validate portfolio size
        if portfolio_size > num_strategies:
            self.logger.warning(f"Reducing portfolio size from {portfolio_size} to {num_strategies}")
            portfolio_size = num_strategies
        
        # Calculate correlation matrix for entire dataset
        # Flatten zone matrix to (dates*zones, strategies) for correlation
        flattened_data = zone_matrix.reshape(-1, num_strategies)
        
        # Remove any rows that are all zeros
        non_zero_mask = np.any(flattened_data != 0, axis=1)
        flattened_data = flattened_data[non_zero_mask]
        
        if len(flattened_data) > 0:
            corr_matrix = self.correlation_calculator.calculate_correlation_matrix(flattened_data.T)
        else:
            corr_matrix = np.eye(num_strategies)
        
        # Get zone weights
        zone_weights = np.array([self.config.zone_weights.get(f"Zone {i+1}", 1.0) for i in range(num_zones)])
        zone_weights = zone_weights / np.sum(zone_weights)
        
        # Import and run algorithm
        try:
            from algorithms import get_algorithm
            optimizer = get_algorithm(algorithm)
            
            if optimizer is None:
                raise ValueError(f"Unknown algorithm: {algorithm}")
            
            # Create fitness function wrapper
            def fitness_wrapper(individual):
                return self.evaluate_zone_fitness(
                    individual, zone_matrix, 
                    metric=kwargs.get('metric', 'ratio'),
                    corr_matrix=corr_matrix,
                    zone_weights=zone_weights,
                    drawdown_threshold=kwargs.get('drawdown_threshold', 0)
                )
            
            # Run optimization
            best_indices, best_fitness, metrics = optimizer.optimize(
                flattened_data.T,  # Pass transposed data (strategies, observations)
                portfolio_size,
                fitness_function=fitness_wrapper,
                **kwargs
            )
            
        except ImportError:
            # Fallback to simple random selection
            self.logger.warning("Using fallback random selection")
            best_indices = np.random.choice(num_strategies, portfolio_size, replace=False)
            best_fitness = self.evaluate_zone_fitness(
                best_indices, zone_matrix, corr_matrix=corr_matrix, zone_weights=zone_weights
            )
            metrics = {'method': 'random'}
        
        # Calculate zone-wise performance
        selected_returns = zone_matrix[:, :, best_indices]
        avg_returns = np.mean(selected_returns, axis=2)
        zone_performance = {}
        
        for i in range(num_zones):
            zone_returns = avg_returns[:, i]
            zone_roi = np.sum(zone_returns)
            zone_performance[f"Zone {i+1}"] = zone_roi
        
        # Calculate weighted performance
        weighted_returns = np.dot(avg_returns, zone_weights)
        weighted_performance = np.sum(weighted_returns)
        
        # Create result
        result = ZoneOptimizationResult(
            portfolio_indices=best_indices.tolist() if isinstance(best_indices, np.ndarray) else best_indices,
            portfolio_columns=[strategy_columns[i] for i in best_indices],
            fitness_score=float(best_fitness),
            zone_performance=zone_performance,
            weighted_performance=float(weighted_performance),
            metrics=metrics,
            optimization_time=time.time() - start_time,
            algorithm_used=algorithm,
            zone_matrix_shape=zone_matrix.shape,
            metadata={
                'num_zones': num_zones,
                'zone_weights': zone_weights.tolist(),
                'correlation_matrix_used': corr_matrix is not None,
                'ulta_applied': self.config.apply_ulta
            }
        )
        
        self.logger.info(f"Zone optimization complete: fitness={best_fitness:.4f}, time={result.optimization_time:.2f}s")
        return result
    
    def process_consolidated_file(self, file_path: str, portfolio_size: int,
                                algorithm: str = "genetic", **kwargs) -> ZoneOptimizationResult:
        """
        Process a consolidated CSV file with Zone column
        
        Args:
            file_path: Path to consolidated CSV from Strategy_consolidator.py
            portfolio_size: Target portfolio size
            algorithm: Optimization algorithm
            **kwargs: Additional parameters
        """
        self.logger.info(f"Processing consolidated file: {file_path}")
        
        # Load data
        df = pd.read_csv(file_path)
        
        # Validate Zone column
        if 'Zone' not in df.columns:
            raise ValueError("File must contain 'Zone' column. Use Strategy_consolidator.py first.")
        
        # Build zone matrix
        zone_matrix, strategy_columns = self.build_zone_matrix(df)
        
        # Run optimization
        result = self.optimize_zone_portfolio(
            zone_matrix, strategy_columns, portfolio_size, algorithm, **kwargs
        )
        
        return result
    
    def load_to_heavydb(self, zone_matrix: np.ndarray, strategy_columns: List[str],
                       table_name: str = "zone_optimization_data") -> bool:
        """
        Load zone matrix data to HeavyDB for GPU acceleration
        """
        try:
            # Reshape zone matrix to 2D for HeavyDB
            # (dates, zones, strategies) -> (dates*zones, strategies)
            num_dates, num_zones, num_strategies = zone_matrix.shape
            
            # Create flattened dataframe
            flattened_data = zone_matrix.reshape(-1, num_strategies)
            
            # Create zone and date indices
            date_indices = np.repeat(range(num_dates), num_zones)
            zone_indices = np.tile(range(num_zones), num_dates)
            
            # Build dataframe
            df = pd.DataFrame(flattened_data, columns=strategy_columns)
            df['date_idx'] = date_indices
            df['zone_idx'] = zone_indices
            
            # Load to HeavyDB
            success = load_strategy_data(df, table_name)
            
            if success:
                self.logger.info(f"Zone data loaded to HeavyDB table: {table_name}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to load zone data to HeavyDB: {e}")
            return False


def main():
    """Test zone optimizer with sample data"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Zone-Based Portfolio Optimizer")
    parser.add_argument("--input", "-i", required=True, help="Consolidated CSV file with Zone column")
    parser.add_argument("--portfolio-size", "-p", type=int, default=30, help="Portfolio size")
    parser.add_argument("--algorithm", "-a", default="genetic", help="Algorithm to use")
    parser.add_argument("--config", "-c", help="Configuration file")
    
    args = parser.parse_args()
    
    # Create optimizer
    optimizer = IntradayZoneOptimizer(config_path=args.config)
    
    # Process file
    result = optimizer.process_consolidated_file(
        args.input,
        args.portfolio_size,
        args.algorithm
    )
    
    # Display results
    print("\n=== Zone Optimization Results ===")
    print(f"Algorithm: {result.algorithm_used}")
    print(f"Fitness Score: {result.fitness_score:.4f}")
    print(f"Portfolio Size: {len(result.portfolio_indices)}")
    print(f"Optimization Time: {result.optimization_time:.2f}s")
    
    print("\nZone Performance:")
    for zone, performance in result.zone_performance.items():
        print(f"  {zone}: {performance:.4f}")
    
    print(f"\nWeighted Performance: {result.weighted_performance:.4f}")
    
    print("\nSelected Strategies:")
    for i, strategy in enumerate(result.portfolio_columns[:10]):
        print(f"  {i+1}. {strategy}")
    if len(result.portfolio_columns) > 10:
        print(f"  ... and {len(result.portfolio_columns) - 10} more")


if __name__ == "__main__":
    main()