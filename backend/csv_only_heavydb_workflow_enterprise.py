#!/usr/bin/env python3
"""
CSV-Only HeavyDB Workflow with Enterprise CSV Loading

Enhanced version using the enterprise CSV loader with all required features:
- Streaming for large files
- Progress tracking
- Data validation
- Batch processing
- GPU memory monitoring
- Audit trail
"""

import os
import sys
import time
import argparse
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

# Add backend to path
sys.path.append(str(Path(__file__).parent))

from dal.csv_dal_enhanced import EnhancedCSVDAL
from lib.csv_loader import EnterpriseCSVLoader, LoadMetrics
from algorithms import AlgorithmFactory


class EnterpriseCSVWorkflow:
    """Enhanced CSV workflow with enterprise features"""
    
    def __init__(self):
        """Initialize enterprise workflow"""
        self.logger = logging.getLogger(__name__)
        
        # Initialize enhanced DAL
        self.dal = EnhancedCSVDAL()
        self.dal.connect()
        
        # Algorithm factory for modular algorithms
        config_path = "/mnt/optimizer_share/backend/config/algorithm_config.ini"
        self.algorithm_factory = AlgorithmFactory(config_path)
        
        # Workflow metrics
        self.workflow_metrics = {
            'start_time': None,
            'csv_load_metrics': None,
            'algorithm_metrics': {},
            'output_metrics': {}
        }
        
    def run(self, input_file: str, portfolio_size: int = 35):
        """
        Run the complete workflow with enterprise CSV loading
        
        Args:
            input_file: Path to input CSV file
            portfolio_size: Target portfolio size
        """
        self.workflow_metrics['start_time'] = time.time()
        
        print("="*80)
        print("🚀 Heavy Optimizer Platform - Enterprise CSV Workflow")
        print("="*80)
        print(f"📁 Input File: {input_file}")
        print(f"📊 Portfolio Size: {portfolio_size}")
        print(f"⏰ Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        
        try:
            # Step 1: Load CSV with enterprise features
            print("\n📥 Step 1: Loading CSV with Enterprise Features...")
            csv_metrics = self._load_csv_enterprise(input_file)
            
            if not csv_metrics or csv_metrics.status != "SUCCESS":
                raise Exception("CSV loading failed")
            
            # Step 2: Data preprocessing
            print("\n🔧 Step 2: Data Preprocessing...")
            processed_data = self._preprocess_data('trading_data')
            
            # Step 3: Run optimization algorithms
            print("\n🧬 Step 3: Running Optimization Algorithms...")
            algorithm_results = self._run_algorithms(processed_data, portfolio_size)
            
            # Step 4: Generate outputs
            print("\n📄 Step 4: Generating Output Files...")
            output_dir = self._generate_outputs(
                input_file, portfolio_size, processed_data, algorithm_results
            )
            
            # Step 5: Display summary
            self._display_summary(csv_metrics, algorithm_results, output_dir)
            
            print("\n✅ Workflow completed successfully!")
            
        except Exception as e:
            self.logger.error(f"Workflow failed: {e}")
            print(f"\n❌ Workflow failed: {e}")
            raise
        
        finally:
            self.dal.disconnect()
    
    def _load_csv_enterprise(self, input_file: str) -> LoadMetrics:
        """Load CSV using enterprise loader with progress tracking"""
        
        # Progress tracking
        def progress_callback(progress: dict):
            percentage = progress.get('percentage', 0)
            rows = progress.get('rows_processed', 0)
            
            # Update progress bar
            bar_length = 40
            filled = int(bar_length * percentage / 100)
            bar = '█' * filled + '░' * (bar_length - filled)
            
            print(f'\r  Progress: [{bar}] {percentage:.1f}% - {rows:,} rows processed', 
                  end='', flush=True)
        
        # Load CSV with all features
        success = self.dal.load_csv_to_heavydb(
            filepath=input_file,
            table_name='trading_data',
            progress_callback=progress_callback
        )
        
        print()  # New line after progress bar
        
        if not success:
            raise Exception("Failed to load CSV file")
        
        # Get load metrics
        metrics_list = self.dal.get_load_metrics(limit=1)
        if metrics_list:
            metrics = metrics_list[0]
            self.workflow_metrics['csv_load_metrics'] = metrics
            
            # Display load statistics
            print(f"\n  ✅ Load Statistics:")
            print(f"     - Total Rows: {metrics.total_rows:,}")
            print(f"     - Total Columns: {metrics.total_columns}")
            print(f"     - Load Time: {metrics.load_time_seconds:.2f}s")
            print(f"     - Throughput: {metrics.throughput_mbps:.2f} MB/s")
            print(f"     - Peak Memory: {metrics.peak_memory_mb:.1f} MB")
            
            if metrics.validation_errors > 0:
                print(f"     - ⚠️  Validation Errors: {metrics.validation_errors}")
            
            if metrics.gpu_memory_used_mb > 0:
                print(f"     - GPU Memory: {metrics.gpu_memory_used_mb:.1f} MB")
            
            return metrics
        
        return None
    
    def _preprocess_data(self, table_name: str) -> dict:
        """Preprocess data with ULTA and correlation analysis"""
        
        # Apply ULTA transformation
        print("  📈 Applying ULTA transformation...")
        ulta_success = self.dal.apply_ulta_transformation(table_name)
        
        if not ulta_success:
            raise Exception("ULTA transformation failed")
        
        # Compute correlation matrix
        print("  📊 Computing correlation matrix...")
        corr_matrix = self.dal.compute_correlation_matrix(f"{table_name}_ulta")
        
        if corr_matrix is None:
            raise Exception("Correlation computation failed")
        
        # Get table info
        table_info = self.dal.get_table_info(f"{table_name}_ulta")
        
        print(f"  ✅ Preprocessing complete:")
        print(f"     - Strategies after ULTA: {table_info['columns']}")
        print(f"     - Correlation matrix shape: {corr_matrix.shape}")
        
        return {
            'table_name': f"{table_name}_ulta",
            'table_info': table_info,
            'correlation_matrix': corr_matrix
        }
    
    def _run_algorithms(self, processed_data: dict, portfolio_size: int) -> dict:
        """Run optimization algorithms using modular architecture"""
        
        # Get data from DAL
        table_name = processed_data['table_name']
        df = self.dal.tables[table_name]
        
        # Convert to numpy array
        daily_matrix = df.values
        
        # Define fitness function
        def fitness_function(data, portfolio):
            # Extract portfolio returns
            portfolio_returns = data[:, portfolio].mean(axis=1)
            
            # Calculate cumulative returns
            cumulative_returns = np.cumsum(portfolio_returns)
            
            # Calculate ROI
            roi = cumulative_returns[-1]
            
            # Calculate maximum drawdown
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdowns = running_max - cumulative_returns
            max_drawdown = np.max(drawdowns) if np.max(drawdowns) > 0 else 1e-6
            
            # Base fitness (ROI/Drawdown ratio)
            base_fitness = roi / max_drawdown if max_drawdown > 1e-6 else roi * 100
            
            # Correlation penalty
            if len(portfolio) > 1:
                corr_matrix = processed_data['correlation_matrix']
                correlations = []
                for i in range(len(portfolio)):
                    for j in range(i + 1, len(portfolio)):
                        correlations.append(corr_matrix[portfolio[i], portfolio[j]])
                
                avg_correlation = np.mean(correlations) if correlations else 0
                penalty = 10.0 * avg_correlation
            else:
                penalty = 0
            
            return base_fitness - penalty
        
        # Run all algorithms
        algorithm_names = [
            'genetic_algorithm', 'particle_swarm_optimization',
            'simulated_annealing', 'differential_evolution',
            'ant_colony_optimization', 'hill_climbing',
            'bayesian_optimization', 'random_search'
        ]
        
        results = {}
        
        for algo_name in algorithm_names:
            print(f"\n  🔄 Running {algo_name}...")
            
            try:
                # Create algorithm instance
                algorithm = self.algorithm_factory.create_algorithm(algo_name)
                
                # Run optimization
                start_time = time.time()
                result = algorithm.optimize(
                    daily_matrix=daily_matrix,
                    portfolio_size=portfolio_size,
                    fitness_function=fitness_function
                )
                
                results[algo_name] = result
                
                print(f"     ✅ Completed: Fitness={result['best_fitness']:.6f}, "
                      f"Time={result['execution_time']:.3f}s")
                
            except Exception as e:
                self.logger.error(f"Algorithm {algo_name} failed: {e}")
                print(f"     ❌ Failed: {e}")
        
        # Find best algorithm
        best_algorithm = max(results.keys(), 
                           key=lambda k: results[k]['best_fitness'])
        best_result = results[best_algorithm]
        
        print(f"\n  🏆 Best Algorithm: {best_algorithm}")
        print(f"     - Best Fitness: {best_result['best_fitness']:.6f}")
        print(f"     - Portfolio Size: {len(best_result['best_portfolio'])}")
        
        self.workflow_metrics['algorithm_metrics'] = results
        
        return {
            'results': results,
            'best_algorithm': best_algorithm,
            'best_result': best_result
        }
    
    def _generate_outputs(self, input_file: str, portfolio_size: int,
                         processed_data: dict, algorithm_results: dict) -> str:
        """Generate output files"""
        
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("/mnt/optimizer_share/output") / f"run_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Summary file
        summary_file = output_dir / f"optimization_summary_{timestamp}.txt"
        with open(summary_file, 'w') as f:
            f.write("Heavy Optimizer Platform - Enterprise CSV Workflow Results\n")
            f.write("="*60 + "\n")
            f.write(f"Execution Timestamp: {timestamp}\n")
            f.write(f"Input File: {Path(input_file).name}\n")
            f.write(f"Portfolio Size: {portfolio_size}\n")
            f.write(f"Best Algorithm: {algorithm_results['best_algorithm']}\n")
            f.write(f"Best Fitness: {algorithm_results['best_result']['best_fitness']:.6f}\n")
            
            # CSV load metrics
            if self.workflow_metrics['csv_load_metrics']:
                metrics = self.workflow_metrics['csv_load_metrics']
                f.write(f"\nCSV Load Statistics:\n")
                f.write(f"  - Total Rows: {metrics.total_rows:,}\n")
                f.write(f"  - Load Time: {metrics.load_time_seconds:.2f}s\n")
                f.write(f"  - Throughput: {metrics.throughput_mbps:.2f} MB/s\n")
                f.write(f"  - Peak Memory: {metrics.peak_memory_mb:.1f} MB\n")
        
        # 2. Algorithm results CSV
        results_file = output_dir / "algorithm_results.csv"
        results_data = []
        
        for algo_name, result in algorithm_results['results'].items():
            results_data.append({
                'Algorithm': algo_name,
                'Fitness': result['best_fitness'],
                'ExecutionTime': result['execution_time'],
                'PortfolioSize': len(result['best_portfolio'])
            })
        
        pd.DataFrame(results_data).to_csv(results_file, index=False)
        
        # 3. Best portfolio file
        best_result = algorithm_results['best_result']
        portfolio_file = output_dir / f"best_portfolio_{timestamp}.json"
        
        with open(portfolio_file, 'w') as f:
            json.dump({
                'algorithm': algorithm_results['best_algorithm'],
                'fitness': best_result['best_fitness'],
                'portfolio': best_result['best_portfolio'],
                'execution_time': best_result['execution_time']
            }, f, indent=2)
        
        print(f"\n  ✅ Output files generated in: {output_dir}")
        
        return str(output_dir)
    
    def _display_summary(self, csv_metrics: LoadMetrics, 
                        algorithm_results: dict, output_dir: str):
        """Display workflow summary"""
        
        total_time = time.time() - self.workflow_metrics['start_time']
        
        print("\n" + "="*80)
        print("📊 WORKFLOW SUMMARY")
        print("="*80)
        
        # CSV Loading
        print(f"\n📥 CSV Loading:")
        print(f"   - File Size: {csv_metrics.file_size_mb:.2f} MB")
        print(f"   - Rows Loaded: {csv_metrics.total_rows:,}")
        print(f"   - Load Time: {csv_metrics.load_time_seconds:.2f}s")
        print(f"   - Throughput: {csv_metrics.throughput_mbps:.2f} MB/s")
        
        # Algorithm Performance
        print(f"\n🧬 Algorithm Performance:")
        results = algorithm_results['results']
        
        # Sort by fitness
        sorted_algos = sorted(results.items(), 
                            key=lambda x: x[1]['best_fitness'], 
                            reverse=True)
        
        for algo_name, result in sorted_algos[:5]:  # Top 5
            print(f"   - {algo_name:30s}: {result['best_fitness']:8.4f} "
                  f"({result['execution_time']:6.3f}s)")
        
        # Overall metrics
        print(f"\n⏱️  Total Workflow Time: {total_time:.2f}s")
        print(f"📁 Output Directory: {output_dir}")
        
        print("\n" + "="*80)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Heavy Optimizer Platform - Enterprise CSV Workflow'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to input CSV file'
    )
    parser.add_argument(
        '--portfolio-size',
        type=int,
        default=35,
        help='Target portfolio size (default: 35)'
    )
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run workflow
    workflow = EnterpriseCSVWorkflow()
    workflow.run(args.input, args.portfolio_size)


if __name__ == '__main__':
    main()