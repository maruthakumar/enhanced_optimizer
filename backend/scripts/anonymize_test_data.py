#!/usr/bin/env python3
"""
Test Data Anonymization Script for Heavy Optimizer Platform

This script anonymizes production data for safe use in testing while preserving
all mathematical relationships and edge cases.
"""

import pandas as pd
import numpy as np
import argparse
import json
from datetime import datetime, timedelta
import hashlib
import os
from pathlib import Path


class TestDataAnonymizer:
    """Anonymize production data for testing purposes"""
    
    def __init__(self, date_shift_days=180, seed=42):
        """
        Initialize anonymizer
        
        Args:
            date_shift_days: Number of days to shift dates
            seed: Random seed for reproducibility
        """
        self.date_shift_days = date_shift_days
        self.seed = seed
        np.random.seed(seed)
        self.strategy_mapping = {}
        
    def anonymize_strategy_names(self, df, strategy_columns):
        """
        Replace strategy names with generic identifiers
        
        Args:
            df: DataFrame with strategy data
            strategy_columns: List of columns containing strategy data
            
        Returns:
            DataFrame with anonymized strategy names
        """
        # Create mapping for strategy names
        for i, col in enumerate(strategy_columns):
            if col not in self.strategy_mapping:
                self.strategy_mapping[col] = f"STRATEGY_{i+1:04d}"
        
        # Rename columns
        df_anon = df.copy()
        df_anon = df_anon.rename(columns=self.strategy_mapping)
        
        return df_anon
    
    def shift_dates(self, df, date_column='Date'):
        """
        Shift dates to obscure actual trading periods
        
        Args:
            df: DataFrame with date column
            date_column: Name of date column
            
        Returns:
            DataFrame with shifted dates
        """
        if date_column in df.columns:
            try:
                # Try to convert to datetime
                df[date_column] = pd.to_datetime(df[date_column])
                df[date_column] = df[date_column] - timedelta(days=self.date_shift_days)
                # Convert back to string format YYYY-MM-DD
                df[date_column] = df[date_column].dt.strftime('%Y-%m-%d')
            except:
                # If date conversion fails, leave as is
                print(f"Warning: Could not shift dates in column '{date_column}'")
        
        return df
    
    def identify_edge_cases(self, df, strategy_columns):
        """
        Identify and preserve edge cases in the data
        
        Args:
            df: DataFrame with strategy data
            strategy_columns: List of strategy column names
            
        Returns:
            Dictionary of edge case indices
        """
        edge_cases = {}
        
        # Find zero return days
        zero_returns = []
        for col in strategy_columns:
            if col in df.columns:
                zero_mask = df[col] == 0
                zero_returns.extend(df.index[zero_mask].tolist())
        edge_cases['zero_returns'] = list(set(zero_returns))
        
        # Find maximum drawdown events
        max_drawdowns = []
        for col in strategy_columns:
            if col in df.columns:
                returns = df[col]
                cumulative = returns.cumsum()
                running_max = cumulative.cummax()
                drawdown = cumulative - running_max
                
                # Find worst 5% drawdown days
                threshold = drawdown.quantile(0.05)
                max_drawdowns.extend(df.index[drawdown <= threshold].tolist())
        edge_cases['max_drawdowns'] = list(set(max_drawdowns))
        
        # Find high correlation pairs (this is preserved automatically)
        # Just document which strategies show high correlation
        
        return edge_cases
    
    def validate_anonymization(self, original_df, anon_df, strategy_columns):
        """
        Validate that mathematical relationships are preserved
        
        Args:
            original_df: Original DataFrame
            anon_df: Anonymized DataFrame
            strategy_columns: Original strategy column names
            
        Returns:
            Validation report
        """
        report = {
            'valid': True,
            'checks': {}
        }
        
        # Check correlations preserved
        orig_corr = original_df[strategy_columns].corr()
        anon_columns = [self.strategy_mapping.get(col, col) for col in strategy_columns]
        anon_corr = anon_df[anon_columns].corr()
        
        # Check if correlations match within tolerance
        corr_diff = np.abs(orig_corr.values - anon_corr.values).max()
        report['checks']['correlation_preserved'] = corr_diff < 1e-10
        
        # Check summary statistics
        orig_stats = original_df[strategy_columns].describe()
        anon_stats = anon_df[anon_columns].describe()
        
        stats_match = np.allclose(orig_stats.values, anon_stats.values, rtol=1e-10)
        report['checks']['statistics_preserved'] = stats_match
        
        # Check row count
        report['checks']['row_count_match'] = len(original_df) == len(anon_df)
        
        # Overall validation
        report['valid'] = all(report['checks'].values())
        
        return report
    
    def anonymize_file(self, input_path, output_path, mapping_path=None):
        """
        Anonymize a complete CSV file
        
        Args:
            input_path: Path to input CSV
            output_path: Path to output anonymized CSV
            mapping_path: Optional path to save strategy mapping
        """
        # Read the CSV
        df = pd.read_csv(input_path)
        
        # Identify strategy columns (exclude metadata columns)
        metadata_cols = ['Date', 'Day', 'start_time', 'end_time', 'market_regime', 
                        'Regime_Confidence_%', 'Market_regime_transition_threshold',
                        'capital', 'zone']
        strategy_columns = [col for col in df.columns if col not in metadata_cols]
        
        # Anonymize strategy names
        df_anon = self.anonymize_strategy_names(df, strategy_columns)
        
        # Shift dates
        df_anon = self.shift_dates(df_anon)
        
        # Identify edge cases for documentation
        edge_cases = self.identify_edge_cases(df, strategy_columns)
        
        # Validate anonymization
        validation = self.validate_anonymization(df, df_anon, strategy_columns)
        
        if not validation['valid']:
            raise ValueError(f"Anonymization validation failed: {validation}")
        
        # Save anonymized data
        df_anon.to_csv(output_path, index=False)
        print(f"Anonymized data saved to: {output_path}")
        
        # Save mapping if requested (NEVER commit this file)
        if mapping_path:
            mapping_data = {
                'strategy_mapping': self.strategy_mapping,
                'date_shift_days': self.date_shift_days,
                'edge_cases': edge_cases,
                'validation': validation
            }
            with open(mapping_path, 'w') as f:
                json.dump(mapping_data, f, indent=2)
            print(f"Mapping saved to: {mapping_path}")
            print("WARNING: Never commit the mapping file to version control!")
        
        # Print summary
        print(f"\nAnonymization Summary:")
        print(f"- Strategies anonymized: {len(strategy_columns)}")
        print(f"- Dates shifted by: {self.date_shift_days} days")
        print(f"- Zero return instances: {len(edge_cases['zero_returns'])}")
        print(f"- Max drawdown instances: {len(edge_cases['max_drawdowns'])}")
        print(f"- Validation passed: {validation['valid']}")


def main():
    parser = argparse.ArgumentParser(
        description='Anonymize production data for testing'
    )
    parser.add_argument(
        '--input', '-i', required=True,
        help='Input CSV file path'
    )
    parser.add_argument(
        '--output', '-o', required=True,
        help='Output anonymized CSV file path'
    )
    parser.add_argument(
        '--strategy-mapping', '-m',
        help='Path to save strategy mapping JSON (NEVER commit this)'
    )
    parser.add_argument(
        '--date-shift', '-d', type=int, default=180,
        help='Number of days to shift dates (default: 180)'
    )
    parser.add_argument(
        '--seed', '-s', type=int, default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    args = parser.parse_args()
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Initialize anonymizer
    anonymizer = TestDataAnonymizer(
        date_shift_days=args.date_shift,
        seed=args.seed
    )
    
    # Anonymize the file
    try:
        anonymizer.anonymize_file(
            input_path=args.input,
            output_path=args.output,
            mapping_path=args.strategy_mapping
        )
    except Exception as e:
        print(f"Error during anonymization: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())