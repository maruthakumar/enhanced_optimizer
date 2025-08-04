#!/usr/bin/env python3
"""
Test data generator for benchmark scenarios.
Creates standardized datasets matching production format.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import logging

# Add backend directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'backend'))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BenchmarkDataGenerator:
    """Generates standardized test datasets for benchmark scenarios."""
    
    def __init__(self, config_path: str = None):
        """Initialize with benchmark configuration."""
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'benchmark_config.json')
        
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.output_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
        os.makedirs(self.output_dir, exist_ok=True)
    
    def generate_strategy_returns(self, strategy_count: int, trading_days: int, 
                                correlation_pattern: str = "medium") -> np.ndarray:
        """Generate realistic strategy return data with specified correlation patterns."""
        np.random.seed(42)  # For reproducibility
        
        # Base return characteristics
        base_return = 0.001  # 0.1% daily return
        base_volatility = 0.02  # 2% daily volatility
        
        # Correlation matrix based on pattern
        if correlation_pattern == "low":
            correlation_strength = 0.1
        elif correlation_pattern == "medium":
            correlation_strength = 0.3
        else:  # high
            correlation_strength = 0.6
        
        # Generate correlated returns
        correlation_matrix = np.full((strategy_count, strategy_count), correlation_strength)
        np.fill_diagonal(correlation_matrix, 1.0)
        
        # Generate multivariate normal returns
        returns = np.random.multivariate_normal(
            mean=np.full(strategy_count, base_return),
            cov=correlation_matrix * (base_volatility ** 2),
            size=trading_days
        )
        
        # Add some outliers and realistic patterns
        # Add trend reversals
        if trading_days > 50:
            trend_change_point = trading_days // 2
            returns[trend_change_point:] *= 0.8
        
        # Add volatility clustering
        high_vol_periods = np.random.choice(trading_days, size=trading_days//10, replace=False)
        returns[high_vol_periods] *= 2.0
        
        return returns
    
    def generate_date_range(self, trading_days: int, start_date: str = "2023-01-01") -> List[str]:
        """Generate trading date range excluding weekends."""
        start = datetime.strptime(start_date, "%Y-%m-%d")
        dates = []
        current_date = start
        
        while len(dates) < trading_days:
            # Skip weekends (Saturday=5, Sunday=6)
            if current_date.weekday() < 5:
                dates.append(current_date.strftime("%Y-%m-%d"))
            current_date += timedelta(days=1)
        
        return dates
    
    def create_dataset(self, strategy_count: int, trading_days: int, 
                      correlation_pattern: str = "medium", 
                      output_filename: str = None) -> str:
        """Create a complete benchmark dataset."""
        logger.info(f"Generating dataset: {strategy_count} strategies, {trading_days} days, {correlation_pattern} correlation")
        
        # Generate returns data
        returns = self.generate_strategy_returns(strategy_count, trading_days, correlation_pattern)
        
        # Generate date range
        dates = self.generate_date_range(trading_days)
        
        # Create DataFrame matching production format
        data = {'Date': dates}
        
        # Add strategy columns (S1, S2, S3, etc.)
        for i in range(strategy_count):
            strategy_name = f'S{i+1}'
            # Convert returns to cumulative values starting from 10000
            cumulative_values = 10000 * np.cumprod(1 + returns[:, i])
            data[strategy_name] = cumulative_values
        
        df = pd.DataFrame(data)
        
        # Generate output filename if not provided
        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"benchmark_{strategy_count}s_{trading_days}d_{correlation_pattern}_{timestamp}.csv"
        
        output_path = os.path.join(self.output_dir, output_filename)
        df.to_csv(output_path, index=False)
        
        # Generate metadata
        metadata = {
            "filename": output_filename,
            "strategy_count": strategy_count,
            "trading_days": trading_days,
            "correlation_pattern": correlation_pattern,
            "file_size_mb": os.path.getsize(output_path) / (1024 * 1024),
            "generated_at": datetime.now().isoformat(),
            "data_range": {
                "start_date": dates[0],
                "end_date": dates[-1]
            },
            "statistics": {
                "mean_return": float(np.mean(returns)),
                "std_return": float(np.std(returns)),
                "min_value": float(df.select_dtypes(include=[np.number]).min().min()),
                "max_value": float(df.select_dtypes(include=[np.number]).max().max())
            }
        }
        
        metadata_path = os.path.join(self.output_dir, f"{output_filename.replace('.csv', '_metadata.json')}")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Dataset created: {output_path} ({metadata['file_size_mb']:.2f} MB)")
        return output_path
    
    def generate_all_benchmark_scenarios(self) -> List[str]:
        """Generate datasets for all configured benchmark scenarios."""
        generated_files = []
        
        for scenario in self.config['benchmark_scenarios']:
            output_filename = f"{scenario['name']}.csv"
            file_path = self.create_dataset(
                strategy_count=scenario['strategy_count'],
                trading_days=scenario['trading_days'],
                correlation_pattern="medium",  # Default correlation pattern
                output_filename=output_filename
            )
            generated_files.append(file_path)
        
        return generated_files
    
    def cleanup_old_data(self, retention_days: int = 7):
        """Remove old benchmark data files."""
        cutoff_time = datetime.now() - timedelta(days=retention_days)
        
        for filename in os.listdir(self.output_dir):
            file_path = os.path.join(self.output_dir, filename)
            if os.path.isfile(file_path):
                file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                if file_time < cutoff_time and filename.startswith('benchmark_'):
                    os.remove(file_path)
                    logger.info(f"Removed old benchmark file: {filename}")


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate benchmark test data")
    parser.add_argument('--strategies', type=int, default=2500, help='Number of strategies')
    parser.add_argument('--days', type=int, default=82, help='Number of trading days')
    parser.add_argument('--correlation', choices=['low', 'medium', 'high'], default='medium',
                       help='Correlation pattern')
    parser.add_argument('--output', type=str, help='Output filename')
    parser.add_argument('--all-scenarios', action='store_true', 
                       help='Generate all configured benchmark scenarios')
    parser.add_argument('--cleanup', action='store_true', help='Clean up old benchmark data')
    
    args = parser.parse_args()
    
    generator = BenchmarkDataGenerator()
    
    if args.cleanup:
        generator.cleanup_old_data()
    
    if args.all_scenarios:
        files = generator.generate_all_benchmark_scenarios()
        print(f"Generated {len(files)} benchmark datasets")
        for f in files:
            print(f"  - {f}")
    else:
        file_path = generator.create_dataset(
            strategy_count=args.strategies,
            trading_days=args.days,
            correlation_pattern=args.correlation,
            output_filename=args.output
        )
        print(f"Generated benchmark dataset: {file_path}")


if __name__ == "__main__":
    main()