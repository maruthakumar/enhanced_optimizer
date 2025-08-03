#!/usr/bin/env python3
"""
Validation Framework for Parquet/Arrow/cuDF Migration
Ensures results match HeavyDB implementation within tolerance
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import logging
from datetime import datetime
from pathlib import Path
import argparse
from typing import Dict, List, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add backend to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class MigrationValidator:
    """
    Validates Parquet/cuDF results against HeavyDB baseline
    """
    
    def __init__(self, tolerance: float = 0.001):
        """
        Initialize validator
        
        Args:
            tolerance: Maximum allowed percentage difference (0.001 = 0.1%)
        """
        self.tolerance = tolerance
        self.validation_results = {
            'timestamp': datetime.now().isoformat(),
            'tolerance': tolerance,
            'tests': []
        }
    
    def run_both_workflows(self, csv_path: str, portfolio_size: int = 35) -> Tuple[dict, dict]:
        """
        Run both HeavyDB and Parquet workflows on same data
        
        Returns:
            Tuple of (heavydb_results, parquet_results)
        """
        logger.info(f"Running validation on {csv_path}")
        
        # Create output directories
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        heavydb_output = f"/mnt/optimizer_share/output/validation_heavydb_{timestamp}"
        parquet_output = f"/mnt/optimizer_share/output/validation_parquet_{timestamp}"
        
        os.makedirs(heavydb_output, exist_ok=True)
        os.makedirs(parquet_output, exist_ok=True)
        
        # Run HeavyDB workflow
        logger.info("Running HeavyDB workflow...")
        heavydb_cmd = (
            f"python3 csv_only_heavydb_workflow.py "
            f"--input {csv_path} "
            f"--output {heavydb_output} "
            f"--portfolio-size {portfolio_size}"
        )
        
        heavydb_status = os.system(heavydb_cmd)
        if heavydb_status != 0:
            raise RuntimeError("HeavyDB workflow failed")
        
        # Load HeavyDB results
        heavydb_results = self._load_workflow_results(heavydb_output)
        
        # Run Parquet/cuDF workflow
        logger.info("Running Parquet/cuDF workflow...")
        parquet_cmd = (
            f"python3 parquet_cudf_workflow.py "
            f"--input {csv_path} "
            f"--output {parquet_output} "
            f"--portfolio-size {portfolio_size}"
        )
        
        parquet_status = os.system(parquet_cmd)
        if parquet_status != 0:
            raise RuntimeError("Parquet workflow failed")
        
        # Load Parquet results
        parquet_results = self._load_workflow_results(parquet_output)
        
        return heavydb_results, parquet_results
    
    def _load_workflow_results(self, output_dir: str) -> dict:
        """Load results from workflow output directory"""
        results = {}
        
        # Load execution summary
        summary_path = os.path.join(output_dir, 'execution_summary_*.json')
        summary_files = list(Path(output_dir).glob('execution_summary_*.json'))
        if summary_files:
            with open(summary_files[0], 'r') as f:
                results['summary'] = json.load(f)
        
        # Load algorithm results
        results['algorithms'] = {}
        for algo_file in Path(output_dir).glob('*_result.json'):
            algo_name = algo_file.stem.replace('_result', '')
            with open(algo_file, 'r') as f:
                results['algorithms'][algo_name] = json.load(f)
        
        # Load correlation matrix if available
        corr_files = list(Path(output_dir).glob('correlation_matrix_*.csv'))
        if corr_files:
            results['correlations'] = pd.read_csv(corr_files[0], index_col=0)
        
        return results
    
    def validate_fitness_scores(self, heavydb_results: dict, parquet_results: dict) -> dict:
        """
        Validate fitness scores match within tolerance
        
        Returns:
            Validation results dictionary
        """
        validation = {
            'test': 'fitness_scores',
            'status': 'passed',
            'details': []
        }
        
        for algo_name in heavydb_results.get('algorithms', {}):
            if algo_name not in parquet_results.get('algorithms', {}):
                validation['details'].append({
                    'algorithm': algo_name,
                    'status': 'missing_in_parquet'
                })
                continue
            
            heavydb_algo = heavydb_results['algorithms'][algo_name]
            parquet_algo = parquet_results['algorithms'][algo_name]
            
            # Compare metrics
            metrics_comparison = {}
            for metric in ['total_roi', 'max_drawdown', 'win_rate', 'profit_factor', 'fitness_score']:
                if metric in heavydb_algo and metric in parquet_algo:
                    heavydb_val = heavydb_algo[metric]
                    parquet_val = parquet_algo[metric]
                    
                    # Calculate percentage difference
                    if heavydb_val != 0:
                        pct_diff = abs(parquet_val - heavydb_val) / abs(heavydb_val)
                    else:
                        pct_diff = 0 if parquet_val == 0 else float('inf')
                    
                    passed = pct_diff <= self.tolerance
                    
                    metrics_comparison[metric] = {
                        'heavydb': heavydb_val,
                        'parquet': parquet_val,
                        'pct_diff': pct_diff,
                        'passed': passed
                    }
                    
                    if not passed:
                        validation['status'] = 'failed'
            
            validation['details'].append({
                'algorithm': algo_name,
                'metrics': metrics_comparison
            })
        
        self.validation_results['tests'].append(validation)
        return validation
    
    def validate_correlations(self, heavydb_results: dict, parquet_results: dict) -> dict:
        """
        Validate correlation matrices match
        
        Returns:
            Validation results dictionary
        """
        validation = {
            'test': 'correlations',
            'status': 'passed',
            'details': {}
        }
        
        if 'correlations' not in heavydb_results or 'correlations' not in parquet_results:
            validation['status'] = 'skipped'
            validation['reason'] = 'Correlation data not available'
            self.validation_results['tests'].append(validation)
            return validation
        
        heavydb_corr = heavydb_results['correlations']
        parquet_corr = parquet_results['correlations']
        
        # Ensure same shape
        if heavydb_corr.shape != parquet_corr.shape:
            validation['status'] = 'failed'
            validation['details']['shape_mismatch'] = {
                'heavydb': heavydb_corr.shape,
                'parquet': parquet_corr.shape
            }
        else:
            # Compare values
            diff_matrix = np.abs(heavydb_corr.values - parquet_corr.values)
            max_diff = np.max(diff_matrix)
            mean_diff = np.mean(diff_matrix)
            
            validation['details'] = {
                'max_difference': float(max_diff),
                'mean_difference': float(mean_diff),
                'shape': heavydb_corr.shape
            }
            
            if max_diff > self.tolerance:
                validation['status'] = 'failed'
                # Find locations of largest differences
                large_diffs = np.where(diff_matrix > self.tolerance)
                validation['details']['large_differences'] = [
                    {
                        'row': int(row),
                        'col': int(col),
                        'heavydb': float(heavydb_corr.iloc[row, col]),
                        'parquet': float(parquet_corr.iloc[row, col]),
                        'diff': float(diff_matrix[row, col])
                    }
                    for row, col in zip(large_diffs[0][:5], large_diffs[1][:5])
                ]
        
        self.validation_results['tests'].append(validation)
        return validation
    
    def validate_performance(self, heavydb_results: dict, parquet_results: dict) -> dict:
        """
        Compare performance metrics
        
        Returns:
            Performance comparison dictionary
        """
        performance = {
            'test': 'performance',
            'details': {}
        }
        
        # Compare execution times
        if 'summary' in heavydb_results and 'summary' in parquet_results:
            heavydb_time = heavydb_results['summary'].get('total_execution_time', 0)
            parquet_time = parquet_results['summary'].get('total_execution_time', 0)
            
            speedup = heavydb_time / parquet_time if parquet_time > 0 else 0
            
            performance['details'] = {
                'heavydb_time': heavydb_time,
                'parquet_time': parquet_time,
                'speedup': speedup,
                'improvement_pct': (speedup - 1) * 100
            }
        
        self.validation_results['tests'].append(performance)
        return performance
    
    def save_validation_report(self, output_path: str):
        """Save validation report to file"""
        with open(output_path, 'w') as f:
            json.dump(self.validation_results, f, indent=2)
        
        # Also create a summary report
        summary_path = output_path.replace('.json', '_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("Parquet/Arrow/cuDF Migration Validation Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Timestamp: {self.validation_results['timestamp']}\n")
            f.write(f"Tolerance: {self.validation_results['tolerance']} ({self.tolerance * 100}%)\n\n")
            
            # Summary of tests
            passed_tests = sum(1 for test in self.validation_results['tests'] 
                             if test.get('status') == 'passed')
            total_tests = len(self.validation_results['tests'])
            
            f.write(f"Tests Passed: {passed_tests}/{total_tests}\n\n")
            
            # Details of each test
            for test in self.validation_results['tests']:
                f.write(f"\n{test['test'].upper()}\n")
                f.write("-" * 30 + "\n")
                f.write(f"Status: {test.get('status', 'unknown')}\n")
                
                if test['test'] == 'fitness_scores':
                    for algo_detail in test.get('details', []):
                        f.write(f"\n{algo_detail['algorithm']}:\n")
                        for metric, data in algo_detail.get('metrics', {}).items():
                            f.write(f"  {metric}: {data['passed']} ")
                            f.write(f"(HeavyDB: {data['heavydb']:.4f}, ")
                            f.write(f"Parquet: {data['parquet']:.4f}, ")
                            f.write(f"Diff: {data['pct_diff']*100:.3f}%)\n")
                
                elif test['test'] == 'performance':
                    details = test.get('details', {})
                    f.write(f"HeavyDB Time: {details.get('heavydb_time', 0):.2f}s\n")
                    f.write(f"Parquet Time: {details.get('parquet_time', 0):.2f}s\n")
                    f.write(f"Speedup: {details.get('speedup', 0):.2f}x\n")
                    f.write(f"Improvement: {details.get('improvement_pct', 0):.1f}%\n")
        
        logger.info(f"Validation report saved to {output_path}")

def main():
    """Main validation entry point"""
    parser = argparse.ArgumentParser(description='Validate Parquet/cuDF migration')
    parser.add_argument('--input', '-i', required=True, help='Input CSV file')
    parser.add_argument('--portfolio-size', '-p', type=int, default=35, help='Portfolio size')
    parser.add_argument('--tolerance', '-t', type=float, default=0.001, help='Tolerance (0.001 = 0.1%)')
    parser.add_argument('--output', '-o', help='Output report path')
    
    args = parser.parse_args()
    
    # Validate input
    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)
    
    # Create validator
    validator = MigrationValidator(tolerance=args.tolerance)
    
    try:
        # Run both workflows
        heavydb_results, parquet_results = validator.run_both_workflows(
            args.input, 
            args.portfolio_size
        )
        
        # Validate fitness scores
        logger.info("Validating fitness scores...")
        validator.validate_fitness_scores(heavydb_results, parquet_results)
        
        # Validate correlations
        logger.info("Validating correlations...")
        validator.validate_correlations(heavydb_results, parquet_results)
        
        # Compare performance
        logger.info("Comparing performance...")
        validator.validate_performance(heavydb_results, parquet_results)
        
        # Save report
        if args.output:
            report_path = args.output
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = f"/mnt/optimizer_share/output/validation_report_{timestamp}.json"
        
        validator.save_validation_report(report_path)
        
        # Check overall status
        all_passed = all(test.get('status') == 'passed' 
                        for test in validator.validation_results['tests']
                        if test.get('status') != 'skipped')
        
        if all_passed:
            logger.info("✅ All validation tests passed!")
            sys.exit(0)
        else:
            logger.error("❌ Some validation tests failed. Check report for details.")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Validation failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()