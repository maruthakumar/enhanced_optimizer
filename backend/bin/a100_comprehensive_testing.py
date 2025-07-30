#!/usr/bin/env python3
"""
Comprehensive A100 Performance Testing Framework
End-to-end testing with SENSEX dataset and multiple portfolio sizes
"""

import sys
import os
import time
import numpy as np
import pandas as pd
import json
import logging
import subprocess
from datetime import datetime
from typing import Dict, List, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('A100ComprehensiveTesting')

class A100PerformanceMonitor:
    """Advanced performance monitoring for A100 GPU"""
    
    def __init__(self):
        self.monitoring_active = False
        self.performance_data = []
        
    def start_monitoring(self):
        """Start continuous GPU monitoring"""
        self.monitoring_active = True
        self.performance_data = []
        logger.info("A100 performance monitoring started")
    
    def stop_monitoring(self):
        """Stop GPU monitoring and return data"""
        self.monitoring_active = False
        logger.info("A100 performance monitoring stopped")
        return self.performance_data
    
    def get_gpu_metrics(self) -> Dict[str, Any]:
        """Get comprehensive A100 GPU metrics"""
        try:
            # Get memory info
            memory_result = subprocess.run([
                'nvidia-smi', '--query-gpu=memory.used,memory.total,memory.free',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True)
            
            # Get utilization info
            util_result = subprocess.run([
                'nvidia-smi', '--query-gpu=utilization.gpu,utilization.memory,temperature.gpu,power.draw',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True)
            
            if memory_result.returncode == 0 and util_result.returncode == 0:
                memory_used, memory_total, memory_free = memory_result.stdout.strip().split(', ')
                gpu_util, mem_util, temperature, power_draw = util_result.stdout.strip().split(', ')
                
                metrics = {
                    'timestamp': time.time(),
                    'memory_used_mb': float(memory_used),
                    'memory_total_mb': float(memory_total),
                    'memory_free_mb': float(memory_free),
                    'memory_utilization': float(memory_used) / float(memory_total),
                    'gpu_utilization': float(gpu_util) / 100.0,
                    'memory_bandwidth_utilization': float(mem_util) / 100.0,
                    'temperature_c': float(temperature),
                    'power_draw_w': float(power_draw),
                    'memory_efficiency': (float(memory_used) / float(memory_total)) * (float(mem_util) / 100.0)
                }
                
                if self.monitoring_active:
                    self.performance_data.append(metrics)
                
                return metrics
        except Exception as e:
            logger.debug(f"Could not get A100 GPU metrics: {e}")
        
        return {
            'timestamp': time.time(),
            'memory_used_mb': 0,
            'memory_total_mb': 40960,  # A100 40GB
            'memory_utilization': 0.0,
            'gpu_utilization': 0.0,
            'memory_bandwidth_utilization': 0.0,
            'temperature_c': 0,
            'power_draw_w': 0,
            'memory_efficiency': 0.0
        }

class A100ComprehensiveTester:
    """Comprehensive A100 testing framework"""
    
    def __init__(self):
        self.performance_monitor = A100PerformanceMonitor()
        self.test_results = {}
        self.sensex_data = None
        
        # Load SENSEX dataset
        self.load_sensex_dataset()
        
        # Initialize A100 optimizer
        try:
            from a100_optimized_gpu_optimizer import A100OptimizedGPUOptimizer
            self.a100_optimizer = A100OptimizedGPUOptimizer(connection_pool_size=5)
            logger.info("A100 Optimized GPU Optimizer initialized")
        except ImportError as e:
            logger.error(f"Failed to import A100 optimizer: {e}")
            self.a100_optimizer = None
    
    def load_sensex_dataset(self) -> bool:
        """Load the actual SENSEX dataset"""
        try:
            sensex_file = '/home/administrator/Optimizer/SENSEX_test_dataset.xlsx'
            if os.path.exists(sensex_file):
                df = pd.read_excel(sensex_file)
                logger.info(f"Loaded SENSEX dataset: {len(df)} rows, {len(df.columns)} columns")
                
                # Extract strategy columns
                reserved_columns = ['Date', 'Day', 'Zone', 'date', 'day', 'zone']
                strategy_columns = [col for col in df.columns if col not in reserved_columns]
                
                # Create daily matrix
                daily_matrix = np.zeros((len(df), len(strategy_columns)))
                for i, col in enumerate(strategy_columns):
                    daily_matrix[:, i] = pd.to_numeric(df[col], errors='coerce').fillna(0).values
                
                self.sensex_data = daily_matrix
                logger.info(f"SENSEX daily matrix created: {daily_matrix.shape} with {len(strategy_columns)} strategies")
                return True
            else:
                logger.error(f"SENSEX dataset not found at {sensex_file}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to load SENSEX dataset: {e}")
            return False
    
    def test_single_algorithm_a100(self, algorithm_name: str, portfolio_size: int, metric: str) -> Dict[str, Any]:
        """Test a single algorithm with A100 optimization"""
        
        if self.a100_optimizer is None or self.sensex_data is None:
            return {'error': 'A100 optimizer or SENSEX data not available'}
        
        logger.info(f"Testing {algorithm_name} with portfolio size {portfolio_size}, metric {metric}")
        
        # Start performance monitoring
        self.performance_monitor.start_monitoring()
        
        try:
            # Get algorithm function
            if hasattr(self.a100_optimizer, f'a100_{algorithm_name}'):
                algorithm_func = getattr(self.a100_optimizer, f'a100_{algorithm_name}')
                
                # Record initial GPU state
                initial_metrics = self.performance_monitor.get_gpu_metrics()
                
                # Execute algorithm
                start_time = time.time()
                
                if algorithm_name == 'genetic_algorithm':
                    result = algorithm_func(self.sensex_data, portfolio_size, metric, generations=30, population_size=60)
                elif algorithm_name == 'particle_swarm_optimization':
                    result = algorithm_func(self.sensex_data, portfolio_size, metric, iterations=30, swarm_size=40)
                elif algorithm_name == 'differential_evolution':
                    result = algorithm_func(self.sensex_data, portfolio_size, metric, generations=30, population_size=60)
                elif algorithm_name == 'simulated_annealing':
                    result = algorithm_func(self.sensex_data, portfolio_size, metric, iterations=300, initial_temperature=100.0)
                elif algorithm_name == 'ant_colony_optimization':
                    result = algorithm_func(self.sensex_data, portfolio_size, metric, iterations=15, num_ants=25)
                elif algorithm_name == 'hill_climbing':
                    result = algorithm_func(self.sensex_data, portfolio_size, metric, iterations=50)
                elif algorithm_name == 'bayesian_optimization':
                    result = algorithm_func(self.sensex_data, portfolio_size, metric, iterations=25)
                else:
                    result = algorithm_func(self.sensex_data, portfolio_size, metric)
                
                execution_time = time.time() - start_time
                
                # Record final GPU state
                final_metrics = self.performance_monitor.get_gpu_metrics()
                
                # Stop monitoring and get performance data
                performance_data = self.performance_monitor.stop_monitoring()
                
                # Calculate performance metrics
                peak_memory = max([data['memory_used_mb'] for data in performance_data]) if performance_data else final_metrics['memory_used_mb']
                avg_gpu_util = np.mean([data['gpu_utilization'] for data in performance_data]) if performance_data else final_metrics['gpu_utilization']
                avg_memory_bandwidth = np.mean([data['memory_bandwidth_utilization'] for data in performance_data]) if performance_data else final_metrics['memory_bandwidth_utilization']
                peak_temperature = max([data['temperature_c'] for data in performance_data]) if performance_data else final_metrics['temperature_c']
                avg_power_draw = np.mean([data['power_draw_w'] for data in performance_data]) if performance_data else final_metrics['power_draw_w']
                
                test_result = {
                    'algorithm': algorithm_name,
                    'portfolio_size': portfolio_size,
                    'metric': metric,
                    'fitness': result.fitness,
                    'execution_time': execution_time,
                    'gpu_accelerated': result.gpu_accelerated,
                    'a100_optimized': result.a100_optimized,
                    'gpu_speedup': result.gpu_speedup,
                    'tensor_cores_used': result.tensor_cores_used,
                    'cuda_streams_used': result.cuda_streams_used,
                    'peak_memory_mb': peak_memory,
                    'memory_utilization': peak_memory / 40960,  # A100 40GB
                    'avg_gpu_utilization': avg_gpu_util,
                    'avg_memory_bandwidth_utilization': avg_memory_bandwidth,
                    'peak_temperature_c': peak_temperature,
                    'avg_power_draw_w': avg_power_draw,
                    'memory_efficiency': avg_memory_bandwidth * (peak_memory / 40960),
                    'performance_data': performance_data,
                    'success': True
                }
                
                logger.info(f"✅ {algorithm_name}: fitness={result.fitness:.6f}, time={execution_time:.2f}s, "
                          f"speedup={result.gpu_speedup:.1f}x, peak_mem={peak_memory:.0f}MB")
                
                return test_result
                
            else:
                return {'error': f'Algorithm {algorithm_name} not found', 'success': False}
                
        except Exception as e:
            self.performance_monitor.stop_monitoring()
            logger.error(f"❌ {algorithm_name} failed: {e}")
            return {'error': str(e), 'success': False}
    
    def run_comprehensive_a100_tests(self) -> Dict[str, Any]:
        """Run comprehensive A100 tests with multiple portfolio sizes"""
        
        logger.info("🚀 STARTING COMPREHENSIVE A100 PERFORMANCE TESTS")
        logger.info("=" * 80)
        
        if self.sensex_data is None:
            return {'error': 'SENSEX dataset not available'}
        
        # Test configuration
        algorithms = [
            'genetic_algorithm',
            'particle_swarm_optimization',
            'differential_evolution',
            'simulated_annealing',
            'hill_climbing',
            'bayesian_optimization'
        ]
        
        portfolio_sizes = [10, 20, 30, 50]
        metrics = ['ratio', 'roi']  # Test with 2 metrics for comprehensive coverage
        
        comprehensive_results = {
            'test_configuration': {
                'algorithms': algorithms,
                'portfolio_sizes': portfolio_sizes,
                'metrics': metrics,
                'dataset_shape': self.sensex_data.shape,
                'total_tests': len(algorithms) * len(portfolio_sizes) * len(metrics)
            },
            'individual_results': {},
            'performance_summary': {},
            'a100_optimization_analysis': {}
        }
        
        total_tests = len(algorithms) * len(portfolio_sizes) * len(metrics)
        completed_tests = 0
        successful_tests = 0
        
        logger.info(f"Total tests to run: {total_tests}")
        logger.info(f"Dataset: {self.sensex_data.shape[0]} days, {self.sensex_data.shape[1]} strategies")
        
        # Run all test combinations
        for algorithm in algorithms:
            comprehensive_results['individual_results'][algorithm] = {}
            
            for portfolio_size in portfolio_sizes:
                comprehensive_results['individual_results'][algorithm][portfolio_size] = {}
                
                for metric in metrics:
                    test_key = f"{algorithm}_size_{portfolio_size}_metric_{metric}"
                    
                    logger.info(f"[{completed_tests + 1}/{total_tests}] Testing {test_key}")
                    
                    # Run individual test
                    test_result = self.test_single_algorithm_a100(algorithm, portfolio_size, metric)
                    
                    comprehensive_results['individual_results'][algorithm][portfolio_size][metric] = test_result
                    
                    if test_result.get('success', False):
                        successful_tests += 1
                        logger.info(f"✅ {test_key} completed successfully")
                    else:
                        logger.error(f"❌ {test_key} failed: {test_result.get('error', 'Unknown error')}")
                    
                    completed_tests += 1
                    
                    # Brief pause between tests to allow GPU to cool down
                    time.sleep(2)
        
        # Calculate performance summary
        comprehensive_results['performance_summary'] = self._calculate_performance_summary(
            comprehensive_results['individual_results']
        )
        
        # A100 optimization analysis
        comprehensive_results['a100_optimization_analysis'] = self._analyze_a100_optimizations(
            comprehensive_results['individual_results']
        )
        
        # Overall test summary
        comprehensive_results['test_summary'] = {
            'total_tests': total_tests,
            'completed_tests': completed_tests,
            'successful_tests': successful_tests,
            'success_rate': (successful_tests / total_tests) * 100 if total_tests > 0 else 0,
            'test_duration': time.time(),
            'dataset_info': {
                'shape': self.sensex_data.shape,
                'strategies': self.sensex_data.shape[1],
                'trading_days': self.sensex_data.shape[0]
            }
        }
        
        logger.info(f"🏆 Comprehensive A100 tests completed:")
        logger.info(f"   Success rate: {comprehensive_results['test_summary']['success_rate']:.1f}%")
        logger.info(f"   Successful tests: {successful_tests}/{total_tests}")
        
        return comprehensive_results
    
    def _calculate_performance_summary(self, individual_results: Dict) -> Dict[str, Any]:
        """Calculate comprehensive performance summary"""
        
        performance_summary = {
            'algorithm_performance': {},
            'portfolio_size_analysis': {},
            'metric_comparison': {},
            'a100_utilization': {}
        }
        
        # Collect all successful results
        successful_results = []
        for algorithm, sizes in individual_results.items():
            for size, metrics in sizes.items():
                for metric, result in metrics.items():
                    if result.get('success', False):
                        successful_results.append({
                            'algorithm': algorithm,
                            'portfolio_size': size,
                            'metric': metric,
                            **result
                        })
        
        if not successful_results:
            return performance_summary
        
        # Algorithm performance analysis
        for algorithm in set(r['algorithm'] for r in successful_results):
            algorithm_results = [r for r in successful_results if r['algorithm'] == algorithm]
            
            if algorithm_results:
                performance_summary['algorithm_performance'][algorithm] = {
                    'avg_execution_time': np.mean([r['execution_time'] for r in algorithm_results]),
                    'avg_fitness': np.mean([r['fitness'] for r in algorithm_results]),
                    'avg_gpu_speedup': np.mean([r.get('gpu_speedup', 1.0) for r in algorithm_results]),
                    'avg_memory_utilization': np.mean([r.get('memory_utilization', 0.0) for r in algorithm_results]),
                    'avg_gpu_utilization': np.mean([r.get('avg_gpu_utilization', 0.0) for r in algorithm_results]),
                    'peak_memory_mb': max([r.get('peak_memory_mb', 0) for r in algorithm_results]),
                    'tests_completed': len(algorithm_results)
                }
        
        # Portfolio size analysis
        for size in set(r['portfolio_size'] for r in successful_results):
            size_results = [r for r in successful_results if r['portfolio_size'] == size]
            
            if size_results:
                performance_summary['portfolio_size_analysis'][size] = {
                    'avg_execution_time': np.mean([r['execution_time'] for r in size_results]),
                    'avg_fitness': np.mean([r['fitness'] for r in size_results]),
                    'avg_memory_usage': np.mean([r.get('peak_memory_mb', 0) for r in size_results]),
                    'scalability_factor': np.mean([r.get('gpu_speedup', 1.0) for r in size_results])
                }
        
        # Overall A100 utilization
        performance_summary['a100_utilization'] = {
            'avg_memory_utilization': np.mean([r.get('memory_utilization', 0.0) for r in successful_results]),
            'peak_memory_usage_mb': max([r.get('peak_memory_mb', 0) for r in successful_results]),
            'avg_gpu_utilization': np.mean([r.get('avg_gpu_utilization', 0.0) for r in successful_results]),
            'avg_memory_bandwidth_utilization': np.mean([r.get('avg_memory_bandwidth_utilization', 0.0) for r in successful_results]),
            'avg_power_draw_w': np.mean([r.get('avg_power_draw_w', 0) for r in successful_results]),
            'memory_efficiency': np.mean([r.get('memory_efficiency', 0.0) for r in successful_results])
        }
        
        return performance_summary
    
    def _analyze_a100_optimizations(self, individual_results: Dict) -> Dict[str, Any]:
        """Analyze A100-specific optimizations"""
        
        optimization_analysis = {
            'tensor_core_utilization': {},
            'memory_optimization': {},
            'cuda_streams_efficiency': {},
            'bottleneck_analysis': {}
        }
        
        # Collect successful results
        successful_results = []
        for algorithm, sizes in individual_results.items():
            for size, metrics in sizes.items():
                for metric, result in metrics.items():
                    if result.get('success', False):
                        successful_results.append(result)
        
        if successful_results:
            # Tensor core analysis
            tensor_core_results = [r for r in successful_results if r.get('tensor_cores_used', False)]
            optimization_analysis['tensor_core_utilization'] = {
                'algorithms_using_tensor_cores': len(tensor_core_results),
                'total_algorithms': len(successful_results),
                'utilization_rate': len(tensor_core_results) / len(successful_results) if successful_results else 0
            }
            
            # Memory optimization analysis
            optimization_analysis['memory_optimization'] = {
                'peak_memory_usage_mb': max([r.get('peak_memory_mb', 0) for r in successful_results]),
                'avg_memory_efficiency': np.mean([r.get('memory_efficiency', 0.0) for r in successful_results]),
                'memory_utilization_distribution': {
                    'low_usage': len([r for r in successful_results if r.get('memory_utilization', 0) < 0.3]),
                    'medium_usage': len([r for r in successful_results if 0.3 <= r.get('memory_utilization', 0) < 0.7]),
                    'high_usage': len([r for r in successful_results if r.get('memory_utilization', 0) >= 0.7])
                }
            }
            
            # CUDA streams efficiency
            optimization_analysis['cuda_streams_efficiency'] = {
                'avg_streams_used': np.mean([r.get('cuda_streams_used', 1) for r in successful_results]),
                'max_streams_used': max([r.get('cuda_streams_used', 1) for r in successful_results])
            }
        
        return optimization_analysis

def main():
    """Main function for comprehensive A100 testing"""
    print("🚀 A100 COMPREHENSIVE PERFORMANCE TESTING")
    print("=" * 100)
    
    tester = A100ComprehensiveTester()
    
    if tester.sensex_data is None:
        print("❌ SENSEX dataset not available - cannot proceed with testing")
        return False
    
    if tester.a100_optimizer is None:
        print("❌ A100 optimizer not available - cannot proceed with testing")
        return False
    
    # Run comprehensive tests
    results = tester.run_comprehensive_a100_tests()
    
    # Save detailed results
    results_file = '/tmp/a100_comprehensive_test_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Detailed A100 test results saved to {results_file}")
    
    # Print summary
    if 'test_summary' in results:
        summary = results['test_summary']
        print(f"\n📊 A100 COMPREHENSIVE TEST SUMMARY:")
        print(f"   Success Rate: {summary['success_rate']:.1f}%")
        print(f"   Successful Tests: {summary['successful_tests']}/{summary['total_tests']}")
        print(f"   Dataset: {summary['dataset_info']['trading_days']} days, {summary['dataset_info']['strategies']} strategies")
        
        if 'a100_utilization' in results.get('performance_summary', {}):
            util = results['performance_summary']['a100_utilization']
            print(f"\n🔥 A100 UTILIZATION METRICS:")
            print(f"   Peak Memory Usage: {util['peak_memory_usage_mb']:.0f}MB / 40960MB ({util['avg_memory_utilization']:.1%})")
            print(f"   Average GPU Utilization: {util['avg_gpu_utilization']:.1%}")
            print(f"   Memory Bandwidth Utilization: {util['avg_memory_bandwidth_utilization']:.1%}")
            print(f"   Average Power Draw: {util['avg_power_draw_w']:.0f}W")
        
        return summary['success_rate'] > 80  # Consider successful if >80% tests pass
    
    return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
