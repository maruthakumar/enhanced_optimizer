#!/usr/bin/env python3
"""
Real Performance Benchmark - Measure Actual Speedup
Tests sequential vs parallel execution with real SENSEX data
"""

import os
import sys
import time
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime

# Import our components
from parallel_algorithm_orchestrator import ParallelAlgorithmOrchestrator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RealPerformanceBenchmark:
    def __init__(self):
        self.results = {
            'benchmark_timestamp': datetime.now().isoformat(),
            'dataset_info': {},
            'sequential_results': {},
            'parallel_results': {},
            'performance_comparison': {}
        }
        
    def load_production_data(self):
        """Load real production SENSEX dataset"""
        logger.info("📊 Loading production SENSEX dataset...")
        
        dataset_path = "/mnt/optimizer_share/input/SENSEX_test_dataset.xlsx"
        
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Production dataset not found: {dataset_path}")
        
        # Load Excel file
        df = pd.read_excel(dataset_path)
        logger.info(f"✅ Dataset loaded: {len(df)} rows, {len(df.columns)} columns")
        
        # Extract strategy columns
        reserved_columns = ['Date', 'Day', 'Zone', 'date', 'day', 'zone']
        strategy_columns = [col for col in df.columns if col not in reserved_columns]
        
        # Create daily returns matrix
        daily_matrix = np.zeros((len(df), len(strategy_columns)))
        for i, col in enumerate(strategy_columns):
            daily_matrix[:, i] = pd.to_numeric(df[col], errors='coerce').fillna(0).values
        
        self.results['dataset_info'] = {
            'file_path': dataset_path,
            'file_size_mb': os.path.getsize(dataset_path) / (1024 * 1024),
            'total_rows': len(df),
            'total_strategies': len(strategy_columns),
            'data_shape': daily_matrix.shape
        }
        
        logger.info(f"✅ Data processed: {len(strategy_columns)} strategies, {len(df)} days")
        return daily_matrix, strategy_columns
    
    def run_sequential_benchmark(self, daily_matrix, strategy_columns, portfolio_size=35):
        """Run algorithms sequentially to measure baseline performance"""
        logger.info("🔄 Running sequential algorithm benchmark...")
        
        start_time = time.time()
        sequential_results = {}
        
        # Create orchestrator but run algorithms one by one
        orchestrator = ParallelAlgorithmOrchestrator()
        
        algorithms = [
            'genetic_algorithm',
            'particle_swarm_optimization',
            'simulated_annealing', 
            'differential_evolution',
            'ant_colony_optimization',
            'bayesian_optimization',
            'random_search'
        ]
        
        for algorithm in algorithms:
            alg_start = time.time()
            
            # Run individual algorithm
            if algorithm == 'genetic_algorithm':
                result = orchestrator._run_genetic_algorithm(daily_matrix, portfolio_size)
            elif algorithm == 'particle_swarm_optimization':
                result = orchestrator._run_pso(daily_matrix, portfolio_size)
            elif algorithm == 'simulated_annealing':
                result = orchestrator._run_simulated_annealing(daily_matrix, portfolio_size)
            elif algorithm == 'differential_evolution':
                result = orchestrator._run_differential_evolution(daily_matrix, portfolio_size)
            elif algorithm == 'ant_colony_optimization':
                result = orchestrator._run_ant_colony(daily_matrix, portfolio_size)
            elif algorithm == 'bayesian_optimization':
                result = orchestrator._run_bayesian_optimization(daily_matrix, portfolio_size)
            elif algorithm == 'random_search':
                result = orchestrator._run_random_search(daily_matrix, portfolio_size)
            
            alg_time = time.time() - alg_start
            result['execution_time'] = alg_time
            sequential_results[algorithm] = result
            
            logger.info(f"✅ {algorithm}: {alg_time:.3f}s, fitness: {result.get('best_fitness', 0):.6f}")
        
        total_sequential_time = time.time() - start_time
        
        # Find best sequential result
        best_sequential = max(sequential_results.items(), 
                            key=lambda x: x[1].get('best_fitness', 0))
        
        self.results['sequential_results'] = {
            'total_execution_time': total_sequential_time,
            'individual_results': sequential_results,
            'best_algorithm': best_sequential[0],
            'best_fitness': best_sequential[1].get('best_fitness', 0),
            'algorithms_completed': len(sequential_results)
        }
        
        logger.info(f"✅ Sequential benchmark completed: {total_sequential_time:.3f}s")
        logger.info(f"🏆 Best sequential: {best_sequential[0]} (fitness: {best_sequential[1].get('best_fitness', 0):.6f})")
        
        return total_sequential_time
    
    def run_parallel_benchmark(self, daily_matrix, strategy_columns, portfolio_size=35):
        """Run algorithms in parallel to measure optimized performance"""
        logger.info("⚡ Running parallel algorithm benchmark...")
        
        start_time = time.time()
        
        # Use the parallel orchestrator
        orchestrator = ParallelAlgorithmOrchestrator()
        parallel_result = orchestrator.execute_all_algorithms_parallel(
            daily_matrix, strategy_columns, portfolio_size
        )
        
        total_parallel_time = time.time() - start_time
        
        self.results['parallel_results'] = {
            'total_execution_time': total_parallel_time,
            'orchestrator_time': parallel_result.get('total_execution_time', 0),
            'algorithms_executed': parallel_result.get('algorithms_executed', 0),
            'success_rate': parallel_result.get('success_rate', 0),
            'best_algorithm': parallel_result.get('best_algorithm', 'None'),
            'best_fitness': parallel_result.get('best_fitness', 0),
            'parallel_efficiency': parallel_result.get('parallel_efficiency', 0),
            'individual_results': parallel_result.get('individual_results', {})
        }
        
        logger.info(f"✅ Parallel benchmark completed: {total_parallel_time:.3f}s")
        logger.info(f"🏆 Best parallel: {parallel_result.get('best_algorithm', 'None')} (fitness: {parallel_result.get('best_fitness', 0):.6f})")
        
        return total_parallel_time
    
    def calculate_performance_improvement(self, sequential_time, parallel_time):
        """Calculate actual performance improvement"""
        logger.info("📊 Calculating performance improvement...")
        
        if parallel_time > 0:
            speedup = sequential_time / parallel_time
            efficiency = (speedup / 7) * 100  # 7 algorithms
            time_saved = sequential_time - parallel_time
            improvement_percent = ((sequential_time - parallel_time) / sequential_time) * 100
        else:
            speedup = 0
            efficiency = 0
            time_saved = 0
            improvement_percent = 0
        
        self.results['performance_comparison'] = {
            'sequential_time_seconds': sequential_time,
            'parallel_time_seconds': parallel_time,
            'actual_speedup': speedup,
            'parallel_efficiency_percent': efficiency,
            'time_saved_seconds': time_saved,
            'improvement_percent': improvement_percent,
            'meets_24x_claim': speedup >= 24.0
        }
        
        logger.info(f"📊 Performance Analysis:")
        logger.info(f"   Sequential Time: {sequential_time:.3f}s")
        logger.info(f"   Parallel Time: {parallel_time:.3f}s")
        logger.info(f"   Actual Speedup: {speedup:.1f}x")
        logger.info(f"   Parallel Efficiency: {efficiency:.1f}%")
        logger.info(f"   Time Saved: {time_saved:.3f}s")
        logger.info(f"   Improvement: {improvement_percent:.1f}%")
        
        return speedup
    
    def run_complete_benchmark(self):
        """Run complete performance benchmark"""
        logger.info("🚀 Starting Real Performance Benchmark")
        logger.info("=" * 80)
        
        try:
            # Load production data
            daily_matrix, strategy_columns = self.load_production_data()
            
            # Run sequential benchmark
            sequential_time = self.run_sequential_benchmark(daily_matrix, strategy_columns)
            
            # Run parallel benchmark  
            parallel_time = self.run_parallel_benchmark(daily_matrix, strategy_columns)
            
            # Calculate improvement
            speedup = self.calculate_performance_improvement(sequential_time, parallel_time)
            
            # Save results
            results_file = f"real_performance_benchmark_{int(time.time())}.json"
            with open(results_file, 'w') as f:
                json.dump(self.results, f, indent=2)
            
            logger.info("=" * 80)
            logger.info("🎉 REAL PERFORMANCE BENCHMARK COMPLETED")
            logger.info(f"📄 Results saved to: {results_file}")
            logger.info(f"⚡ Actual Speedup Achieved: {speedup:.1f}x")
            
            return self.results
            
        except Exception as e:
            logger.error(f"❌ Benchmark failed: {e}")
            return None

def main():
    """Main execution function"""
    benchmark = RealPerformanceBenchmark()
    results = benchmark.run_complete_benchmark()
    
    if results:
        print(f"\n🎯 REAL PERFORMANCE BENCHMARK COMPLETE")
        print(f"Actual Speedup: {results['performance_comparison']['actual_speedup']:.1f}x")
        print(f"Parallel Efficiency: {results['performance_comparison']['parallel_efficiency_percent']:.1f}%")
        print(f"Time Saved: {results['performance_comparison']['time_saved_seconds']:.3f}s")
    else:
        print("\n❌ Benchmark failed - check logs for details")

if __name__ == "__main__":
    main()
