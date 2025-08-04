#!/usr/bin/env python3
"""
Check for performance regressions in CI/CD pipeline
Compares current benchmark results against baselines
"""

import json
import sys
from pathlib import Path
from datetime import datetime


class PerformanceRegressionChecker:
    """Check for performance regressions in benchmark results"""
    
    def __init__(self, baseline_file='benchmark_baselines.json', tolerance=0.20):
        """
        Initialize regression checker
        
        Args:
            baseline_file: Path to baseline performance metrics
            tolerance: Maximum allowed regression (default 20%)
        """
        self.baseline_file = Path(baseline_file)
        self.tolerance = tolerance
        self.regressions = []
        
    def load_baselines(self):
        """Load baseline performance metrics"""
        if not self.baseline_file.exists():
            print(f"Warning: No baseline file found at {self.baseline_file}")
            return {}
            
        with open(self.baseline_file, 'r') as f:
            return json.load(f)
    
    def load_current_results(self, results_file):
        """Load current benchmark results"""
        results_path = Path(results_file)
        if not results_path.exists():
            raise FileNotFoundError(f"Results file not found: {results_file}")
            
        with open(results_path, 'r') as f:
            return json.load(f)
    
    def check_regression(self, test_name, baseline_time, current_time):
        """
        Check if current performance shows regression
        
        Args:
            test_name: Name of the test
            baseline_time: Baseline execution time
            current_time: Current execution time
            
        Returns:
            Tuple of (has_regression, regression_percent)
        """
        if baseline_time == 0:
            return False, 0.0
            
        regression = (current_time - baseline_time) / baseline_time
        has_regression = regression > self.tolerance
        
        if has_regression:
            self.regressions.append({
                'test': test_name,
                'baseline': baseline_time,
                'current': current_time,
                'regression_percent': regression * 100
            })
            
        return has_regression, regression * 100
    
    def generate_report(self, baselines, current_results):
        """Generate performance comparison report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'tolerance': f"{self.tolerance * 100}%",
            'total_tests': 0,
            'regressions_found': 0,
            'improvements_found': 0,
            'details': []
        }
        
        for test_name, current_data in current_results.items():
            report['total_tests'] += 1
            
            if test_name not in baselines:
                # New test, no baseline to compare
                detail = {
                    'test': test_name,
                    'status': 'NEW',
                    'current_time': current_data.get('mean', current_data.get('time', 0)),
                    'message': 'No baseline available for comparison'
                }
            else:
                baseline_time = baselines[test_name].get('time', baselines[test_name].get('mean', 0))
                current_time = current_data.get('mean', current_data.get('time', 0))
                
                has_regression, regression_percent = self.check_regression(
                    test_name, baseline_time, current_time
                )
                
                if has_regression:
                    status = 'REGRESSION'
                    report['regressions_found'] += 1
                elif regression_percent < -10:  # More than 10% improvement
                    status = 'IMPROVED'
                    report['improvements_found'] += 1
                else:
                    status = 'OK'
                
                detail = {
                    'test': test_name,
                    'status': status,
                    'baseline_time': baseline_time,
                    'current_time': current_time,
                    'change_percent': regression_percent
                }
            
            report['details'].append(detail)
        
        return report
    
    def print_report(self, report):
        """Print formatted performance report"""
        print("\n" + "="*60)
        print("PERFORMANCE REGRESSION CHECK REPORT")
        print("="*60)
        print(f"Timestamp: {report['timestamp']}")
        print(f"Tolerance: {report['tolerance']}")
        print(f"Total Tests: {report['total_tests']}")
        print(f"Regressions Found: {report['regressions_found']}")
        print(f"Improvements Found: {report['improvements_found']}")
        print("\n" + "-"*60)
        
        # Group by status
        for status in ['REGRESSION', 'IMPROVED', 'OK', 'NEW']:
            tests = [d for d in report['details'] if d['status'] == status]
            if tests:
                print(f"\n{status}:")
                for test in tests:
                    if status == 'NEW':
                        print(f"  - {test['test']}: {test['current_time']:.3f}s (new test)")
                    else:
                        change = test.get('change_percent', 0)
                        print(f"  - {test['test']}: {test['baseline_time']:.3f}s -> "
                              f"{test['current_time']:.3f}s ({change:+.1f}%)")
        
        print("\n" + "="*60)
        
        if report['regressions_found'] > 0:
            print("\n❌ PERFORMANCE REGRESSIONS DETECTED!")
            print("The following tests exceeded the acceptable performance threshold:")
            for reg in self.regressions:
                print(f"  - {reg['test']}: {reg['regression_percent']:.1f}% slower")
        else:
            print("\n✅ All performance tests passed!")
    
    def check(self, results_file):
        """
        Main method to check for regressions
        
        Args:
            results_file: Path to current benchmark results
            
        Returns:
            True if no regressions found, False otherwise
        """
        baselines = self.load_baselines()
        current_results = self.load_current_results(results_file)
        
        report = self.generate_report(baselines, current_results)
        self.print_report(report)
        
        # Save report
        report_file = Path('performance_regression_report.json')
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\nDetailed report saved to: {report_file}")
        
        return len(self.regressions) == 0


def main():
    """Main entry point for CI/CD integration"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Check for performance regressions')
    parser.add_argument('--results', '-r', required=True,
                        help='Path to current benchmark results file')
    parser.add_argument('--baseline', '-b', default='benchmark_baselines.json',
                        help='Path to baseline performance metrics')
    parser.add_argument('--tolerance', '-t', type=float, default=0.20,
                        help='Maximum allowed regression (default: 0.20 = 20%%)')
    parser.add_argument('--update-baseline', action='store_true',
                        help='Update baseline with current results')
    
    args = parser.parse_args()
    
    checker = PerformanceRegressionChecker(
        baseline_file=args.baseline,
        tolerance=args.tolerance
    )
    
    if args.update_baseline:
        # Update baseline with current results
        current_results = checker.load_current_results(args.results)
        baseline_path = Path(args.baseline)
        
        # Merge with existing baselines
        existing = checker.load_baselines()
        for test_name, data in current_results.items():
            existing[test_name] = {
                'time': data.get('mean', data.get('time', 0)),
                'updated': datetime.now().isoformat()
            }
        
        with open(baseline_path, 'w') as f:
            json.dump(existing, f, indent=2)
        
        print(f"✅ Baseline updated: {baseline_path}")
        return 0
    
    # Check for regressions
    success = checker.check(args.results)
    
    # Exit with appropriate code for CI/CD
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()