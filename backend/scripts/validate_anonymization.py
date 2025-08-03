#!/usr/bin/env python3
"""
Validate that anonymized data preserves mathematical relationships
Compares original and anonymized datasets for correlation preservation
"""

import pandas as pd
import numpy as np
import sys
import json
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')


class AnonymizationValidator:
    """Validate anonymized data integrity"""
    
    def __init__(self, original_path, anonymized_path, mapping_path=None):
        """
        Initialize validator
        
        Args:
            original_path: Path to original data
            anonymized_path: Path to anonymized data
            mapping_path: Optional path to strategy mapping JSON
        """
        self.original_path = original_path
        self.anonymized_path = anonymized_path
        
        print("Loading original data...")
        self.original_df = pd.read_csv(original_path)
        
        print("Loading anonymized data...")
        self.anon_df = pd.read_csv(anonymized_path)
        
        # Load mapping if provided
        self.mapping = None
        if mapping_path:
            with open(mapping_path, 'r') as f:
                data = json.load(f)
                self.mapping = data.get('strategy_mapping', {})
        
        # Identify columns
        self.metadata_cols = ['Date', 'Day', 'start_time', 'end_time', 'market_regime', 
                             'Regime_Confidence_%', 'Market_regime_transition_threshold',
                             'capital', 'zone']
        
        self.original_strategies = [col for col in self.original_df.columns 
                                   if col not in self.metadata_cols]
        self.anon_strategies = [col for col in self.anon_df.columns 
                               if col not in self.metadata_cols and col.startswith('STRATEGY_')]
        
        print(f"Original strategies: {len(self.original_strategies)}")
        print(f"Anonymized strategies: {len(self.anon_strategies)}")
    
    def validate_basic_properties(self):
        """Validate basic dataset properties"""
        print("\n=== Basic Properties Validation ===")
        
        results = {
            'row_count_match': len(self.original_df) == len(self.anon_df),
            'original_rows': len(self.original_df),
            'anon_rows': len(self.anon_df),
            'strategy_count_match': len(self.original_strategies) == len(self.anon_strategies),
            'original_strategies': len(self.original_strategies),
            'anon_strategies': len(self.anon_strategies)
        }
        
        print(f"Row count: Original={results['original_rows']}, Anonymized={results['anon_rows']} - {'✓' if results['row_count_match'] else '✗'}")
        print(f"Strategy count: Original={results['original_strategies']}, Anonymized={results['anon_strategies']} - {'✓' if results['strategy_count_match'] else '✗'}")
        
        return results
    
    def validate_summary_statistics(self, sample_size=10):
        """Validate that summary statistics are preserved"""
        print("\n=== Summary Statistics Validation ===")
        
        # Sample strategies for comparison
        if len(self.original_strategies) > sample_size:
            sample_indices = np.linspace(0, len(self.original_strategies)-1, sample_size, dtype=int)
            sample_orig = [self.original_strategies[i] for i in sample_indices]
            sample_anon = [self.anon_strategies[i] for i in sample_indices]
        else:
            sample_orig = self.original_strategies
            sample_anon = self.anon_strategies
        
        stats_match = True
        detailed_stats = []
        
        for orig_col, anon_col in zip(sample_orig, sample_anon):
            orig_stats = self.original_df[orig_col].describe()
            anon_stats = self.anon_df[anon_col].describe()
            
            # Check if statistics match within tolerance
            stats_close = np.allclose(orig_stats.values, anon_stats.values, rtol=1e-10)
            stats_match = stats_match and stats_close
            
            detailed_stats.append({
                'original_strategy': orig_col,
                'anon_strategy': anon_col,
                'stats_match': stats_close,
                'mean_diff': abs(orig_stats['mean'] - anon_stats['mean']),
                'std_diff': abs(orig_stats['std'] - anon_stats['std'])
            })
        
        print(f"Summary statistics preserved: {'✓' if stats_match else '✗'}")
        if not stats_match:
            print("Mismatches found in:")
            for stat in detailed_stats:
                if not stat['stats_match']:
                    print(f"  - {stat['original_strategy']} -> {stat['anon_strategy']}")
        
        return {
            'all_stats_match': stats_match,
            'sample_size': len(sample_orig),
            'detailed_stats': detailed_stats[:5]  # Return first 5 for summary
        }
    
    def validate_correlations(self, sample_size=20):
        """Validate that correlations between strategies are preserved"""
        print("\n=== Correlation Preservation Validation ===")
        
        # Sample strategies for correlation testing
        if len(self.original_strategies) > sample_size:
            sample_indices = np.random.choice(len(self.original_strategies), 
                                            size=sample_size, replace=False)
            sample_orig = [self.original_strategies[i] for i in sample_indices]
            sample_anon = [self.anon_strategies[i] for i in sample_indices]
        else:
            sample_orig = self.original_strategies[:sample_size]
            sample_anon = self.anon_strategies[:sample_size]
        
        # Calculate correlation matrices
        orig_corr = self.original_df[sample_orig].corr()
        anon_corr = self.anon_df[sample_anon].corr()
        
        # Compare correlation matrices
        corr_diff = np.abs(orig_corr.values - anon_corr.values)
        max_diff = corr_diff.max()
        avg_diff = corr_diff.mean()
        
        # Check if correlations match within tolerance
        correlations_match = max_diff < 1e-10
        
        print(f"Correlations preserved: {'✓' if correlations_match else '✗'}")
        print(f"Max correlation difference: {max_diff:.2e}")
        print(f"Average correlation difference: {avg_diff:.2e}")
        
        # Find pairs with largest differences
        if max_diff > 1e-10:
            # Find indices of max difference
            max_idx = np.unravel_index(corr_diff.argmax(), corr_diff.shape)
            print(f"Largest difference between: {sample_orig[max_idx[0]]} and {sample_orig[max_idx[1]]}")
        
        return {
            'correlations_preserved': correlations_match,
            'max_correlation_diff': float(max_diff),
            'avg_correlation_diff': float(avg_diff),
            'sample_size': len(sample_orig)
        }
    
    def validate_portfolio_metrics(self):
        """Validate that portfolio-level metrics are preserved"""
        print("\n=== Portfolio Metrics Validation ===")
        
        # Calculate portfolio returns (sum of all strategies)
        orig_portfolio = self.original_df[self.original_strategies].sum(axis=1)
        anon_portfolio = self.anon_df[self.anon_strategies].sum(axis=1)
        
        # Compare portfolio statistics
        metrics = {}
        
        # Total return
        orig_total = orig_portfolio.sum()
        anon_total = anon_portfolio.sum()
        metrics['total_return_match'] = np.isclose(orig_total, anon_total, rtol=1e-10)
        metrics['total_return_diff'] = abs(orig_total - anon_total)
        
        # Maximum drawdown
        orig_cumsum = orig_portfolio.cumsum()
        anon_cumsum = anon_portfolio.cumsum()
        
        orig_dd = (orig_cumsum - orig_cumsum.expanding().max()).min()
        anon_dd = (anon_cumsum - anon_cumsum.expanding().max()).min()
        metrics['max_drawdown_match'] = np.isclose(orig_dd, anon_dd, rtol=1e-10)
        metrics['max_drawdown_diff'] = abs(orig_dd - anon_dd)
        
        # Win rate
        orig_win_rate = (orig_portfolio > 0).mean()
        anon_win_rate = (anon_portfolio > 0).mean()
        metrics['win_rate_match'] = np.isclose(orig_win_rate, anon_win_rate, rtol=1e-10)
        metrics['win_rate_diff'] = abs(orig_win_rate - anon_win_rate)
        
        print(f"Portfolio total return preserved: {'✓' if metrics['total_return_match'] else '✗'}")
        print(f"Portfolio max drawdown preserved: {'✓' if metrics['max_drawdown_match'] else '✗'}")
        print(f"Portfolio win rate preserved: {'✓' if metrics['win_rate_match'] else '✗'}")
        
        return metrics
    
    def validate_date_shift(self):
        """Validate that dates were properly shifted"""
        print("\n=== Date Shift Validation ===")
        
        if 'Date' not in self.original_df.columns or 'Date' not in self.anon_df.columns:
            print("Date column not found in one or both datasets")
            return {'date_shift_valid': False, 'reason': 'Date column missing'}
        
        try:
            # Convert dates
            orig_dates = pd.to_datetime(self.original_df['Date'])
            anon_dates = pd.to_datetime(self.anon_df['Date'])
            
            # Calculate difference
            date_diffs = (orig_dates - anon_dates).dt.days
            
            # Check if all differences are the same
            unique_diffs = date_diffs.unique()
            
            if len(unique_diffs) == 1:
                shift_days = unique_diffs[0]
                print(f"Dates shifted by {shift_days} days: ✓")
                return {
                    'date_shift_valid': True,
                    'shift_days': int(shift_days),
                    'consistent': True
                }
            else:
                print(f"Inconsistent date shifts found: ✗")
                return {
                    'date_shift_valid': False,
                    'reason': 'Inconsistent shifts',
                    'unique_shifts': unique_diffs.tolist()
                }
        except Exception as e:
            print(f"Error validating date shift: {e}")
            return {'date_shift_valid': False, 'reason': str(e)}
    
    def generate_validation_report(self):
        """Generate comprehensive validation report"""
        print("\n" + "="*50)
        print("ANONYMIZATION VALIDATION REPORT")
        print("="*50)
        
        report = {
            'original_file': self.original_path,
            'anonymized_file': self.anonymized_path,
            'validation_results': {}
        }
        
        # Run all validations
        report['validation_results']['basic_properties'] = self.validate_basic_properties()
        report['validation_results']['summary_statistics'] = self.validate_summary_statistics()
        report['validation_results']['correlations'] = self.validate_correlations()
        report['validation_results']['portfolio_metrics'] = self.validate_portfolio_metrics()
        report['validation_results']['date_shift'] = self.validate_date_shift()
        
        # Overall verdict
        all_passed = all([
            report['validation_results']['basic_properties']['row_count_match'],
            report['validation_results']['basic_properties']['strategy_count_match'],
            report['validation_results']['summary_statistics']['all_stats_match'],
            report['validation_results']['correlations']['correlations_preserved'],
            report['validation_results']['portfolio_metrics']['total_return_match'],
            report['validation_results']['date_shift']['date_shift_valid']
        ])
        
        report['overall_validation'] = {
            'passed': all_passed,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        print("\n" + "="*50)
        print(f"OVERALL VALIDATION: {'PASSED ✓' if all_passed else 'FAILED ✗'}")
        print("="*50)
        
        return report


def main():
    if len(sys.argv) < 3:
        print("Usage: python validate_anonymization.py original.csv anonymized.csv [mapping.json]")
        sys.exit(1)
    
    original_file = sys.argv[1]
    anonymized_file = sys.argv[2]
    mapping_file = sys.argv[3] if len(sys.argv) > 3 else None
    
    validator = AnonymizationValidator(original_file, anonymized_file, mapping_file)
    report = validator.generate_validation_report()
    
    # Save report (convert numpy types to native Python types)
    report_path = anonymized_file.replace('.csv', '_validation_report.json')
    
    # Convert numpy types to native Python types
    def convert_numpy_types(obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(i) for i in obj]
        return obj
    
    report = convert_numpy_types(report)
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nValidation report saved to: {report_path}")
    
    # Exit with appropriate code
    sys.exit(0 if report['overall_validation']['passed'] else 1)


if __name__ == "__main__":
    main()