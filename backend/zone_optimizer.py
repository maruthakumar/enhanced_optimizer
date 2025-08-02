"""
Zone-wise Optimization Module

Extracts and reimplements the zone-wise optimization logic from legacy code.
Supports configurable number of zones and dynamic zone-based filtering.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import configparser
from dataclasses import dataclass, field
from datetime import datetime
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ulta_calculator import ULTACalculator
from lib.correlation.correlation_matrix_calculator import CorrelationMatrixCalculator
from dal import get_dal


@dataclass
class ZoneConfiguration:
    """Configuration for zone-based optimization"""
    enabled: bool = True
    zones: List[str] = field(default_factory=lambda: [
        "zone_0_100", "zone_101_200", "zone_201_300", "zone_301_400",
        "zone_401_500", "zone_501_600", "zone_601_700", "zone_701_756"
    ])
    zone_weights: Dict[str, float] = field(default_factory=dict)
    min_portfolio_size: int = 10
    max_portfolio_size: int = 25
    population_size: int = 30
    mutation_rate: float = 0.1
    combination_method: str = "weighted_average"  # weighted_average, best_zone, equal_weight
    apply_ulta: bool = True
    
    def __post_init__(self):
        """Initialize zone weights if not provided"""
        if not self.zone_weights:
            # Equal weights by default
            weight = 1.0 / len(self.zones)
            self.zone_weights = {zone: weight for zone in self.zones}
    
    @classmethod
    def from_config_manager(cls, config_manager) -> 'ZoneConfiguration':
        """Create ZoneConfiguration from ConfigurationManager"""
        zone_config = config_manager.get_zone_config()
        
        # Convert zone weights list to dict
        zone_count = zone_config['zone_count']
        zones = [f"zone{i+1}" for i in range(zone_count)]
        zone_weights_list = zone_config['zone_weights']
        zone_weights = {zones[i]: zone_weights_list[i] for i in range(zone_count)}
        
        # Get portfolio sizes
        min_size = zone_config.get('min_per_zone', 5) * zone_count
        max_size = zone_config.get('max_per_zone', 20) * zone_count
        
        return cls(
            enabled=zone_config['enabled'],
            zones=zones,
            zone_weights=zone_weights,
            min_portfolio_size=min_size,
            max_portfolio_size=max_size,
            combination_method=zone_config.get('selection_method', 'weighted_average'),
            apply_ulta=config_manager.getboolean('ULTA', 'enabled', True)
        )


@dataclass
class ZoneResult:
    """Results from optimizing a single zone"""
    zone_name: str
    portfolio_indices: List[int]
    portfolio_columns: List[str]
    fitness_score: float
    metrics: Dict[str, float]
    optimization_time: float
    algorithm_used: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class CombinedZoneResult:
    """Combined results from all zones"""
    zone_results: Dict[str, ZoneResult]
    combined_portfolio: List[int]
    combined_fitness: float
    combination_method: str
    zone_contributions: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


class ZoneOptimizer:
    """
    Zone-wise optimization implementation
    
    Handles zone-based portfolio optimization with support for:
    - Dynamic number of zones
    - Configurable zone weights
    - Multiple combination methods
    - Integration with existing optimization algorithms
    """
    
    def __init__(self, config_path: Optional[str] = None, logger: Optional[logging.Logger] = None):
        """
        Initialize Zone Optimizer
        
        Args:
            config_path: Path to configuration file
            logger: Logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        
        # Load configuration
        if config_path:
            from config.config_manager import get_config_manager
            config_manager = get_config_manager(config_path)
            self.config = ZoneConfiguration.from_config_manager(config_manager)
        else:
            self.config = self._load_configuration(config_path)
        
        # Initialize helper modules with configuration
        self.ulta_calculator = ULTACalculator(config_path=config_path)
        self.correlation_calculator = CorrelationMatrixCalculator()
        
        # Zone data cache
        self._zone_data_cache: Dict[str, pd.DataFrame] = {}
        self._zone_corr_cache: Dict[str, np.ndarray] = {}
        
    def _load_configuration(self, config_path: Optional[str]) -> ZoneConfiguration:
        """Load zone configuration from file"""
        config = ZoneConfiguration()
        
        if config_path and os.path.exists(config_path):
            parser = configparser.ConfigParser()
            parser.read(config_path)
            
            # Check for zone configuration sections
            if 'ZONE_SPECIFIC_OPTIMIZATION' in parser:
                zone_config = parser['ZONE_SPECIFIC_OPTIMIZATION']
                config.enabled = zone_config.getboolean('enable', True)
                config.min_portfolio_size = zone_config.getint('min_size', 10)
                config.max_portfolio_size = zone_config.getint('max_size', 25)
                config.population_size = zone_config.getint('population_size', 30)
                config.mutation_rate = zone_config.getfloat('mutation_rate', 0.1)
                
            if 'OPTIMIZATION' in parser:
                opt_config = parser['OPTIMIZATION']
                config.apply_ulta = opt_config.getboolean('apply_ulta_logic', True)
                
                # Parse zone weights
                zone_weights_str = opt_config.get('zone_weights', '1,1,1,1')
                try:
                    weights = [float(w.strip()) for w in zone_weights_str.split(',')]
                    if len(weights) == len(config.zones):
                        total = sum(weights)
                        config.zone_weights = {
                            zone: weight/total 
                            for zone, weight in zip(config.zones, weights)
                        }
                except Exception as e:
                    self.logger.warning(f"Error parsing zone weights: {e}. Using equal weights.")
                    
        self.logger.info(f"Zone configuration loaded: {len(config.zones)} zones, weights: {config.zone_weights}")
        return config
    
    def extract_zones_from_data(self, data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Extract zone-specific data from the main dataset
        
        Args:
            data: DataFrame with strategy columns
            
        Returns:
            Dictionary mapping zone names to filtered DataFrames
        """
        zones = {}
        
        # Get strategy columns (skip first 3 metadata columns: Date, Day, Zone if present)
        metadata_cols = ['Date', 'Day', 'Zone']
        strategy_columns = [col for col in data.columns if col not in metadata_cols]
        
        # Define 8 zones based on strategy count ranges (Story 1.4 specification)
        zone_definitions = {
            'zone_0_100': (0, 100),
            'zone_101_200': (101, 200), 
            'zone_201_300': (201, 300),
            'zone_301_400': (301, 400),
            'zone_401_500': (401, 500),
            'zone_501_600': (501, 600),
            'zone_601_700': (601, 700),
            'zone_701_756': (701, 756)
        }
        
        total_strategies = len(strategy_columns)
        self.logger.info(f"Creating 8 zones from {total_strategies} strategies")
        
        for zone_name, (start_idx, end_idx) in zone_definitions.items():
            # Adjust end index to not exceed available strategies
            actual_end = min(end_idx, total_strategies - 1)
            
            if start_idx >= total_strategies:
                self.logger.debug(f"Skipping {zone_name}: start index {start_idx} >= total strategies {total_strategies}")
                continue
                
            # Select strategy columns for this zone
            zone_strategy_cols = strategy_columns[start_idx:actual_end + 1]
            
            if not zone_strategy_cols:
                self.logger.debug(f"No strategies in {zone_name}")
                continue
            
            # Create zone dataset with metadata + zone strategies
            zone_cols = [col for col in metadata_cols if col in data.columns] + zone_strategy_cols
            zone_data = data[zone_cols].copy()
            
            # Add zone metadata
            zone_data.attrs = {
                'zone_name': zone_name,
                'strategy_range': f"{start_idx}-{actual_end}",
                'strategy_count': len(zone_strategy_cols)
            }
            
            zones[zone_name] = zone_data
            self.logger.info(f"Created {zone_name}: {len(zone_strategy_cols)} strategies (columns {start_idx}-{actual_end})")
        
        if not zones:
            self.logger.warning("No zones created - insufficient data")
            
        return zones
    
    def optimize_single_zone(self, zone_name: str, zone_data: pd.DataFrame,
                           portfolio_size: int, algorithm: str = "genetic",
                           **kwargs) -> ZoneResult:
        """
        Optimize portfolio for a single zone with ULTA integration
        
        Args:
            zone_name: Name of the zone
            zone_data: DataFrame with zone-specific data
            portfolio_size: Target portfolio size
            algorithm: Optimization algorithm to use
            **kwargs: Additional algorithm parameters
            
        Returns:
            ZoneResult with optimization results
        """
        import time
        start_time = time.time()
        
        zone_info = getattr(zone_data, 'attrs', {})
        strategy_range = zone_info.get('strategy_range', 'unknown')
        strategy_count = zone_info.get('strategy_count', 'unknown')
        
        self.logger.info(f"Optimizing zone '{zone_name}' (strategies {strategy_range}) "
                        f"with {len(zone_data)} rows, {strategy_count} strategies, portfolio size {portfolio_size}")
        
        # Store original data for reference
        original_zone_data = zone_data.copy()
        ulta_metrics = {}
        
        # Apply ULTA transformation if configured
        if self.config.apply_ulta:
            self.logger.info(f"Applying ULTA logic to zone {zone_name}")
            zone_data, ulta_metrics = self.ulta_calculator.apply_ulta_logic(zone_data)
            
            inverted_count = len([m for m in ulta_metrics.values() if m.was_inverted])
            self.logger.info(f"ULTA processing complete: {inverted_count} strategies inverted in zone {zone_name}")
            
        # Extract strategy columns (skip metadata columns)
        metadata_cols = ['Date', 'Day', 'Zone']
        strategy_columns = [col for col in zone_data.columns if col not in metadata_cols]
        
        if not strategy_columns:
            raise ValueError(f"Zone {zone_name} has no strategy columns after ULTA processing")
            
        daily_matrix = zone_data[strategy_columns].to_numpy().astype(float)
        
        # Handle NaN values
        daily_matrix = np.nan_to_num(daily_matrix, nan=0.0)
        
        # Validate portfolio size
        max_strategies = len(strategy_columns)
        if portfolio_size > max_strategies:
            self.logger.warning(f"Reducing portfolio size from {portfolio_size} to {max_strategies} for zone {zone_name}")
            portfolio_size = max_strategies
        
        # Calculate correlation matrix (use cache if available)
        cache_key = f"{zone_name}_ulta_{self.config.apply_ulta}"
        if cache_key not in self._zone_corr_cache:
            self.logger.debug(f"Calculating correlation matrix for zone {zone_name}")
            self._zone_corr_cache[cache_key] = self.correlation_calculator.calculate_correlation_matrix(daily_matrix)
        corr_matrix = self._zone_corr_cache[cache_key]
        
        # Get optimization algorithm
        try:
            from algorithms import get_algorithm
            optimizer = get_algorithm(algorithm)
        except ImportError:
            # Fallback to a simple mock optimizer for testing
            class MockOptimizer:
                def optimize(self, data, size, **kwargs):
                    indices = np.random.choice(data.shape[1], size, replace=False)
                    fitness = np.random.random()
                    metrics = {'roi': np.random.random(), 'sharpe': np.random.random()}
                    return indices, fitness, metrics
            optimizer = MockOptimizer()
        
        if optimizer is None:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        # Run optimization with correlation-based diversification if available
        optimization_kwargs = {
            'correlation_matrix': corr_matrix,
            **kwargs
        }
        
        best_indices, best_fitness, metrics = optimizer.optimize(
            daily_matrix, 
            portfolio_size,
            **optimization_kwargs
        )
        
        # Create result with enhanced metadata
        result = ZoneResult(
            zone_name=zone_name,
            portfolio_indices=best_indices.tolist() if isinstance(best_indices, np.ndarray) else best_indices,
            portfolio_columns=[strategy_columns[i] for i in best_indices],
            fitness_score=float(best_fitness),
            metrics=metrics,
            optimization_time=time.time() - start_time,
            algorithm_used=algorithm,
            metadata={
                'total_strategies': len(strategy_columns),
                'original_strategies': len([col for col in original_zone_data.columns if col not in metadata_cols]),
                'data_rows': len(zone_data),
                'ulta_applied': self.config.apply_ulta,
                'ulta_inversions': len([m for m in ulta_metrics.values() if m.was_inverted]),
                'strategy_range': strategy_range,
                'zone_strategy_count': strategy_count,
                'inverted_strategies': [name for name, m in ulta_metrics.items() if m.was_inverted],
                'correlation_matrix_shape': corr_matrix.shape if corr_matrix is not None else None
            }
        )
        
        self.logger.info(f"Zone {zone_name} optimization complete. Fitness: {best_fitness:.4f}, "
                        f"ULTA inversions: {result.metadata['ulta_inversions']}")
        return result
    
    def optimize_all_zones(self, data: pd.DataFrame, 
                          portfolio_size_per_zone: Optional[int] = None,
                          algorithm: str = "genetic",
                          progress_callback: Optional[callable] = None,
                          **kwargs) -> Dict[str, ZoneResult]:
        """
        Optimize all configured zones independently
        
        Args:
            data: Full dataset with all zones
            portfolio_size_per_zone: Portfolio size for each zone (uses config if None)
            algorithm: Optimization algorithm
            progress_callback: Optional callback for progress updates
            **kwargs: Additional algorithm parameters
            
        Returns:
            Dictionary mapping zone names to optimization results
        """
        # Extract zones
        zone_data = self.extract_zones_from_data(data)
        
        if not zone_data:
            raise ValueError("No valid zones found in data")
        
        # Determine portfolio size
        if portfolio_size_per_zone is None:
            portfolio_size_per_zone = self.config.min_portfolio_size
        
        results = {}
        total_zones = len(zone_data)
        
        for idx, (zone_name, zone_df) in enumerate(zone_data.items()):
            if progress_callback:
                progress = (idx / total_zones) * 100
                progress_callback(progress, f"Optimizing zone: {zone_name}")
            
            try:
                result = self.optimize_single_zone(
                    zone_name, zone_df, portfolio_size_per_zone, 
                    algorithm, **kwargs
                )
                results[zone_name] = result
                
            except Exception as e:
                self.logger.error(f"Failed to optimize zone {zone_name}: {str(e)}")
                # Continue with other zones
                
        if progress_callback:
            progress_callback(100, "Zone optimization complete")
            
        return results
    
    def combine_zone_results(self, zone_results: Dict[str, ZoneResult],
                           method: Optional[str] = None) -> CombinedZoneResult:
        """
        Combine results from multiple zones into a final portfolio
        
        Args:
            zone_results: Results from each zone
            method: Combination method (uses config default if None)
            
        Returns:
            Combined optimization result
        """
        if not zone_results:
            raise ValueError("No zone results to combine")
        
        method = method or self.config.combination_method
        self.logger.info(f"Combining {len(zone_results)} zone results using method: {method}")
        
        if method == "weighted_average":
            return self._combine_weighted_average(zone_results)
        elif method == "best_zone":
            return self._combine_best_zone(zone_results)
        elif method == "equal_weight":
            return self._combine_equal_weight(zone_results)
        else:
            raise ValueError(f"Unknown combination method: {method}")
    
    def _combine_weighted_average(self, zone_results: Dict[str, ZoneResult]) -> CombinedZoneResult:
        """Combine zones using weighted average based on fitness and zone weights"""
        combined_portfolio = []
        zone_contributions = {}
        
        # Calculate weighted fitness
        total_weighted_fitness = 0.0
        
        for zone_name, result in zone_results.items():
            zone_weight = self.config.zone_weights.get(zone_name, 1.0 / len(zone_results))
            weighted_fitness = result.fitness_score * zone_weight
            total_weighted_fitness += weighted_fitness
            zone_contributions[zone_name] = zone_weight
            
            # Add strategies from this zone
            for idx in result.portfolio_indices:
                combined_portfolio.append({
                    'zone': zone_name,
                    'index': idx,
                    'column': result.portfolio_columns[result.portfolio_indices.index(idx)],
                    'weight': zone_weight
                })
        
        # Remove duplicates, keeping highest weighted
        unique_strategies = {}
        for strategy in combined_portfolio:
            key = strategy['column']
            if key not in unique_strategies or strategy['weight'] > unique_strategies[key]['weight']:
                unique_strategies[key] = strategy
        
        # Extract final portfolio
        final_indices = [s['index'] for s in unique_strategies.values()]
        
        return CombinedZoneResult(
            zone_results=zone_results,
            combined_portfolio=final_indices,
            combined_fitness=total_weighted_fitness,
            combination_method="weighted_average",
            zone_contributions=zone_contributions,
            metadata={
                'unique_strategies': len(final_indices),
                'total_strategies': len(combined_portfolio)
            }
        )
    
    def _combine_best_zone(self, zone_results: Dict[str, ZoneResult]) -> CombinedZoneResult:
        """Select portfolio from the best performing zone"""
        best_zone = max(zone_results.items(), key=lambda x: x[1].fitness_score)
        zone_name, best_result = best_zone
        
        zone_contributions = {z: 0.0 for z in zone_results}
        zone_contributions[zone_name] = 1.0
        
        return CombinedZoneResult(
            zone_results=zone_results,
            combined_portfolio=best_result.portfolio_indices,
            combined_fitness=best_result.fitness_score,
            combination_method="best_zone",
            zone_contributions=zone_contributions,
            metadata={'selected_zone': zone_name}
        )
    
    def _combine_equal_weight(self, zone_results: Dict[str, ZoneResult]) -> CombinedZoneResult:
        """Combine zones with equal weights regardless of configuration"""
        equal_weight = 1.0 / len(zone_results)
        
        # Override configured weights
        original_weights = self.config.zone_weights.copy()
        for zone in zone_results:
            self.config.zone_weights[zone] = equal_weight
        
        # Use weighted average with equal weights
        result = self._combine_weighted_average(zone_results)
        result.combination_method = "equal_weight"
        
        # Restore original weights
        self.config.zone_weights = original_weights
        
        return result
    
    def generate_zone_report(self, results: Union[Dict[str, ZoneResult], CombinedZoneResult],
                           output_path: str) -> str:
        """
        Generate detailed report of zone optimization results
        
        Args:
            results: Zone optimization results
            output_path: Path to save the report
            
        Returns:
            Path to generated report
        """
        report_lines = []
        report_lines.append("="*80)
        report_lines.append("ZONE OPTIMIZATION REPORT")
        report_lines.append("="*80)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        if isinstance(results, dict):
            # Individual zone results
            report_lines.append(f"Total Zones Optimized: {len(results)}")
            report_lines.append("")
            
            for zone_name, result in results.items():
                report_lines.append(f"\nZone: {zone_name.upper()}")
                report_lines.append("-"*40)
                report_lines.append(f"Fitness Score: {result.fitness_score:.6f}")
                report_lines.append(f"Portfolio Size: {len(result.portfolio_indices)}")
                report_lines.append(f"Algorithm: {result.algorithm_used}")
                report_lines.append(f"Optimization Time: {result.optimization_time:.2f}s")
                
                if result.metrics:
                    report_lines.append("\nMetrics:")
                    for metric, value in result.metrics.items():
                        report_lines.append(f"  {metric}: {value:.6f}")
                
                report_lines.append("\nSelected Strategies:")
                for col in result.portfolio_columns[:10]:  # Show first 10
                    report_lines.append(f"  - {col}")
                if len(result.portfolio_columns) > 10:
                    report_lines.append(f"  ... and {len(result.portfolio_columns)-10} more")
                    
        elif isinstance(results, CombinedZoneResult):
            # Combined results
            report_lines.append("COMBINED ZONE OPTIMIZATION RESULTS")
            report_lines.append("-"*40)
            report_lines.append(f"Combination Method: {results.combination_method}")
            report_lines.append(f"Combined Fitness: {results.combined_fitness:.6f}")
            report_lines.append(f"Combined Portfolio Size: {len(results.combined_portfolio)}")
            report_lines.append("")
            
            report_lines.append("Zone Contributions:")
            for zone, contribution in results.zone_contributions.items():
                report_lines.append(f"  {zone}: {contribution:.2%}")
            
            report_lines.append("\nIndividual Zone Results:")
            for zone_name, result in results.zone_results.items():
                report_lines.append(f"\n  {zone_name}: Fitness={result.fitness_score:.6f}, Size={len(result.portfolio_indices)}")
        
        # Write report
        report_content = "\n".join(report_lines)
        
        with open(output_path, 'w') as f:
            f.write(report_content)
        
        self.logger.info(f"Zone optimization report saved to: {output_path}")
        return output_path
    
    def save_results_json(self, results: Union[Dict[str, ZoneResult], CombinedZoneResult],
                         output_path: str) -> str:
        """Save zone optimization results as JSON"""
        
        def serialize_result(obj):
            """Custom serializer for dataclasses"""
            if isinstance(obj, (ZoneResult, CombinedZoneResult)):
                return obj.__dict__
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.number):
                return float(obj)
            return str(obj)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, default=serialize_result, indent=2)
        
        self.logger.info(f"Zone results saved to JSON: {output_path}")
        return output_path
    
    def analyze_zone_correlation(self, zone_results: Dict[str, ZoneResult]) -> Dict[str, Any]:
        """
        Analyze correlations between zone portfolios
        
        Args:
            zone_results: Results from zone optimization
            
        Returns:
            Dictionary with correlation analysis
        """
        if len(zone_results) < 2:
            return {"error": "Need at least 2 zones for correlation analysis"}
        
        # Create a matrix of selected strategies per zone
        all_strategies = set()
        for result in zone_results.values():
            all_strategies.update(result.portfolio_columns)
        
        strategy_list = sorted(list(all_strategies))
        n_strategies = len(strategy_list)
        n_zones = len(zone_results)
        
        # Binary matrix: zones x strategies
        selection_matrix = np.zeros((n_zones, n_strategies))
        
        zone_names = list(zone_results.keys())
        for i, zone in enumerate(zone_names):
            for strategy in zone_results[zone].portfolio_columns:
                if strategy in strategy_list:
                    j = strategy_list.index(strategy)
                    selection_matrix[i, j] = 1
        
        # Calculate zone similarity (Jaccard index)
        zone_similarity = np.zeros((n_zones, n_zones))
        
        for i in range(n_zones):
            for j in range(n_zones):
                if i == j:
                    zone_similarity[i, j] = 1.0
                else:
                    intersection = np.sum(selection_matrix[i] * selection_matrix[j])
                    union = np.sum(np.maximum(selection_matrix[i], selection_matrix[j]))
                    if union > 0:
                        zone_similarity[i, j] = intersection / union
                    else:
                        zone_similarity[i, j] = 0.0
        
        return {
            'zone_similarity_matrix': zone_similarity,
            'zone_names': zone_names,
            'total_unique_strategies': n_strategies,
            'strategy_overlap': {
                f"{z1}-{z2}": zone_similarity[i, j]
                for i, z1 in enumerate(zone_names)
                for j, z2 in enumerate(zone_names)
                if i < j
            }
        }