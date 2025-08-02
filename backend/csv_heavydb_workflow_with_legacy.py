#!/usr/bin/env python3
"""
Heavy Optimizer Platform - CSV Workflow with Legacy System Comparison
Extends the main workflow to include side-by-side comparison with legacy system
"""

import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from csv_only_heavydb_workflow import CSVOnlyHeavyDBOptimizer
from legacy_system_integration import LegacySystemIntegration
import logging
import json
from pathlib import Path
from datetime import datetime
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class CSVWorkflowWithLegacyComparison(CSVOnlyHeavyDBOptimizer):
    """Extended workflow that includes legacy system comparison"""
    
    def __init__(self, enable_legacy_comparison=False):
        super().__init__()
        self.enable_legacy_comparison = enable_legacy_comparison
        if enable_legacy_comparison:
            self.legacy_integration = LegacySystemIntegration()
            print("🔄 Legacy System Comparison: ENABLED")
    
    def run_optimization_with_comparison(self, csv_file_path, portfolio_size):
        """Run optimization with optional legacy system comparison"""
        # Run the new optimization first
        success = self.run_optimization(csv_file_path, portfolio_size)
        
        if not success:
            return False
        
        if self.enable_legacy_comparison:
            print("\n" + "=" * 80)
            print("🏛️ RUNNING LEGACY SYSTEM FOR COMPARISON")
            print("=" * 80)
            
            try:
                # Get the results from the new system
                # This is simplified - in reality we'd need to parse the output
                new_results = {
                    'best_algorithm': self.last_results.get('best_algorithm', ''),
                    'best_fitness': self.last_results.get('best_fitness', 0),
                    'algorithm_results': self.last_results.get('algorithm_results', {})
                }
                
                # Run legacy comparison
                comparison_results = self.legacy_integration.run_legacy_comparison(
                    csv_file_path,
                    portfolio_size,
                    new_results
                )
                
                # Display comparison summary
                self._display_comparison_summary(comparison_results)
                
                # Save detailed comparison
                self._save_comparison_results(comparison_results)
                
            except Exception as e:
                self.logger.error(f"Legacy comparison failed: {e}")
                print(f"⚠️ Legacy comparison failed: {e}")
                # Don't fail the whole run if comparison fails
        
        return True
    
    def run_optimization(self, csv_file_path, portfolio_size):
        """Override to capture results for comparison"""
        # Store results for comparison
        self.last_results = {}
        
        print("=" * 80)
        print("🚀 HEAVY OPTIMIZER PLATFORM - WITH LEGACY COMPARISON")
        print("=" * 80)
        
        try:
            # Load CSV data
            loaded_data, load_time = self.load_csv_data(csv_file_path)
            
            # Preprocess data
            processed_data, preprocess_time = self.preprocess_data(loaded_data)
            
            # Execute algorithms
            algorithm_results, algorithm_time = self.execute_algorithms_with_heavydb(
                processed_data, portfolio_size
            )
            
            # Store results for comparison
            self.last_results = algorithm_results
            
            # Generate output
            output_dir, output_time = self.generate_reference_compatible_output(
                csv_file_path, portfolio_size, processed_data, algorithm_results
            )
            
            # Final summary
            total_time = time.time() - self.start_time
            
            print("\n" + "=" * 80)
            print("✅ NEW SYSTEM OPTIMIZATION COMPLETED")
            print("=" * 80)
            print(f"📊 Total Execution Time: {total_time:.3f}s")
            print(f"   - Data Loading: {load_time:.3f}s")
            print(f"   - Preprocessing: {preprocess_time:.3f}s")
            print(f"   - Algorithm Execution: {algorithm_time:.3f}s")
            print(f"   - Output Generation: {output_time:.3f}s")
            print(f"📁 Results saved to: {output_dir}")
            
            return True
            
        except Exception as e:
            print(f"\n❌ OPTIMIZATION FAILED: {e}")
            self.logger.error("Optimization failed", exc_info=True)
            return False
    
    def _display_comparison_summary(self, comparison_results):
        """Display a summary of the comparison results"""
        if 'comparison_report' not in comparison_results:
            return
        
        report = comparison_results['comparison_report']
        summary = report.get('summary', {})
        
        print("\n📊 COMPARISON SUMMARY")
        print("=" * 60)
        print(f"Legacy System Fitness: {summary.get('legacy_fitness', 'N/A')}")
        print(f"New System Fitness: {summary.get('new_fitness', 'N/A')}")
        print(f"Legacy Algorithm: {summary.get('legacy_algorithm', 'N/A')}")
        print(f"New Algorithm: {summary.get('new_algorithm', 'N/A')}")
        
        if 'relative_difference' in summary:
            diff_pct = summary['relative_difference'] * 100
            print(f"Relative Difference: {diff_pct:.4f}%")
        
        if summary.get('fitness_parity'):
            print("✅ Fitness Parity: VALIDATED (within tolerance)")
        else:
            print("❌ Fitness Parity: FAILED (exceeds tolerance)")
        
        # Show deviations if any
        deviations = report.get('deviations', [])
        if deviations:
            print(f"\n⚠️ Found {len(deviations)} deviations:")
            for dev in deviations[:3]:  # Show first 3
                print(f"  - {dev['type']}: Legacy={dev['legacy_value']}, New={dev['new_value']}")
    
    def _save_comparison_results(self, comparison_results):
        """Save detailed comparison results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("/mnt/optimizer_share/output/legacy_comparison")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save full comparison results
        output_file = output_dir / f"full_comparison_{timestamp}.json"
        with open(output_file, 'w') as f:
            json.dump(comparison_results, f, indent=2, default=str)
        
        print(f"\n📄 Detailed comparison saved to: {output_file}")


def main():
    """Main entry point with legacy comparison support"""
    parser = argparse.ArgumentParser(
        description='Heavy Optimizer Platform - With Legacy Comparison'
    )
    parser.add_argument('--input', '-i', type=str, help='Input CSV file path')
    parser.add_argument('--portfolio-size', '-p', type=int, help='Target portfolio size')
    parser.add_argument('--test', action='store_true', help='Run with test dataset')
    parser.add_argument('--legacy-comparison', '-l', action='store_true', 
                       help='Enable legacy system comparison')
    
    args = parser.parse_args()
    
    # Handle test mode
    if args.test:
        csv_file = '/mnt/optimizer_share/input/Python_Multi_Consolidated_20250726_161921.csv'
        portfolio_size = args.portfolio_size or 37  # Use 37 for legacy comparison
        print(f"🧪 Running in TEST mode with {csv_file}")
    else:
        if not args.input or not args.portfolio_size:
            parser.error("--input and --portfolio-size are required unless --test is used")
        csv_file = args.input
        portfolio_size = args.portfolio_size
    
    # Add missing import
    import time
    
    # Create optimizer with legacy comparison if requested
    optimizer = CSVWorkflowWithLegacyComparison(
        enable_legacy_comparison=args.legacy_comparison
    )
    
    # Run optimization with comparison
    success = optimizer.run_optimization_with_comparison(csv_file, portfolio_size)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()