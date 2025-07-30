#!/usr/bin/env python3
"""
GPU-Enhanced Production Portfolio Optimizer
Integrates GPU acceleration with the existing production system
"""

import sys
import os
import time
import logging
import numpy as np
from typing import Dict, List, Any, Optional

# Add production path
sys.path.insert(0, '/opt/heavydb_optimizer/bin')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('GPUProductionOptimizer')

class GPUProductionEnhancedOptimizer:
    """Production-ready GPU-enhanced portfolio optimizer"""
    
    def __init__(self, connection_pool_size: int = 3, enable_gpu: bool = True):
        self.connection_pool_size = connection_pool_size
        self.enable_gpu = enable_gpu
        
        # Initialize GPU optimizer if available
        self.gpu_optimizer = None
        if enable_gpu:
            try:
                from gpu_enhanced_optimizer import GPUEnhancedOptimizer
                self.gpu_optimizer = GPUEnhancedOptimizer(connection_pool_size)
                logger.info("GPU acceleration enabled")
            except ImportError as e:
                logger.warning(f"GPU optimizer not available, using CPU fallback: {e}")
                self.enable_gpu = False
        
        # Initialize CPU optimizer as fallback
        try:
            from production_enhanced_optimizer import FixedCompleteEnhancedOptimizer
            self.cpu_optimizer = FixedCompleteEnhancedOptimizer(connection_pool_size)
            logger.info("CPU optimizer initialized as fallback")
        except ImportError as e:
            logger.error(f"CPU optimizer not available: {e}")
            self.cpu_optimizer = None
    
    def genetic_algorithm(self, daily_matrix: np.ndarray, portfolio_size: int, 
                         metric: str, generations: int = 30, **kwargs) -> Any:
        """Genetic Algorithm with GPU acceleration"""
        if self.enable_gpu and self.gpu_optimizer:
            try:
                return self.gpu_optimizer.gpu_genetic_algorithm(
                    daily_matrix, portfolio_size, metric, generations, 
                    kwargs.get('population_size', 50)
                )
            except Exception as e:
                logger.warning(f"GPU genetic algorithm failed, using CPU: {e}")
        
        # CPU fallback
        if self.cpu_optimizer:
            return self.cpu_optimizer.genetic_algorithm(
                daily_matrix, portfolio_size, metric, generations=generations
            )
        else:
            raise RuntimeError("No optimizer available")
    
    def particle_swarm_optimization(self, daily_matrix: np.ndarray, portfolio_size: int,
                                   metric: str, iterations: int = 30, **kwargs) -> Any:
        """Particle Swarm Optimization with GPU acceleration"""
        if self.enable_gpu and self.gpu_optimizer:
            try:
                return self.gpu_optimizer.gpu_particle_swarm_optimization(
                    daily_matrix, portfolio_size, metric, iterations,
                    kwargs.get('swarm_size', 30)
                )
            except Exception as e:
                logger.warning(f"GPU PSO failed, using CPU: {e}")
        
        # CPU fallback
        if self.cpu_optimizer:
            return self.cpu_optimizer.particle_swarm_optimization(
                daily_matrix, portfolio_size, metric, iterations=iterations
            )
        else:
            raise RuntimeError("No optimizer available")
    
    def differential_evolution(self, daily_matrix: np.ndarray, portfolio_size: int,
                              metric: str, generations: int = 30, **kwargs) -> Any:
        """Differential Evolution with GPU acceleration"""
        if self.enable_gpu and self.gpu_optimizer:
            try:
                return self.gpu_optimizer.gpu_differential_evolution(
                    daily_matrix, portfolio_size, metric, generations,
                    kwargs.get('population_size', 50)
                )
            except Exception as e:
                logger.warning(f"GPU differential evolution failed, using CPU: {e}")
        
        # CPU fallback
        if self.cpu_optimizer:
            return self.cpu_optimizer.differential_evolution(
                daily_matrix, portfolio_size, metric, generations=generations
            )
        else:
            raise RuntimeError("No optimizer available")
    
    def simulated_annealing(self, daily_matrix: np.ndarray, portfolio_size: int,
                           metric: str, iterations: int = 500, **kwargs) -> Any:
        """Simulated Annealing with GPU acceleration"""
        if self.enable_gpu and self.gpu_optimizer:
            try:
                return self.gpu_optimizer.gpu_simulated_annealing(
                    daily_matrix, portfolio_size, metric, iterations,
                    kwargs.get('initial_temperature', 100.0)
                )
            except Exception as e:
                logger.warning(f"GPU simulated annealing failed, using CPU: {e}")
        
        # CPU fallback
        if self.cpu_optimizer:
            return self.cpu_optimizer.simulated_annealing(
                daily_matrix, portfolio_size, metric, iterations=iterations
            )
        else:
            raise RuntimeError("No optimizer available")
    
    def ant_colony_optimization(self, daily_matrix: np.ndarray, portfolio_size: int,
                               metric: str, iterations: int = 20, **kwargs) -> Any:
        """Ant Colony Optimization with GPU acceleration"""
        if self.enable_gpu and self.gpu_optimizer:
            try:
                return self.gpu_optimizer.gpu_ant_colony_optimization(
                    daily_matrix, portfolio_size, metric, iterations,
                    kwargs.get('num_ants', 30)
                )
            except Exception as e:
                logger.warning(f"GPU ACO failed, using CPU: {e}")
        
        # CPU fallback
        if self.cpu_optimizer:
            return self.cpu_optimizer.ant_colony_optimization(
                daily_matrix, portfolio_size, metric, iterations=iterations
            )
        else:
            raise RuntimeError("No optimizer available")
    
    def hill_climbing(self, daily_matrix: np.ndarray, portfolio_size: int,
                     metric: str, iterations: int = 100, **kwargs) -> Any:
        """Hill Climbing with GPU acceleration"""
        if self.enable_gpu and self.gpu_optimizer:
            try:
                return self.gpu_optimizer.gpu_hill_climbing(
                    daily_matrix, portfolio_size, metric, iterations
                )
            except Exception as e:
                logger.warning(f"GPU hill climbing failed, using CPU: {e}")
        
        # CPU fallback
        if self.cpu_optimizer:
            return self.cpu_optimizer.hill_climbing(
                daily_matrix, portfolio_size, metric, iterations=iterations
            )
        else:
            raise RuntimeError("No optimizer available")
    
    def bayesian_optimization(self, daily_matrix: np.ndarray, portfolio_size: int,
                             metric: str, iterations: int = 30, **kwargs) -> Any:
        """Bayesian Optimization with GPU acceleration"""
        if self.enable_gpu and self.gpu_optimizer:
            try:
                return self.gpu_optimizer.gpu_bayesian_optimization(
                    daily_matrix, portfolio_size, metric, iterations
                )
            except Exception as e:
                logger.warning(f"GPU Bayesian optimization failed, using CPU: {e}")
        
        # CPU fallback
        if self.cpu_optimizer:
            return self.cpu_optimizer.bayesian_optimization(
                daily_matrix, portfolio_size, metric, iterations=iterations
            )
        else:
            raise RuntimeError("No optimizer available")
    
    def optimize_parallel(self, daily_matrix: np.ndarray, portfolio_size: int, metric: str,
                         algorithms: Optional[List[str]] = None, max_workers: int = 5) -> Dict[str, Any]:
        """Parallel optimization with GPU acceleration"""
        
        # Use GPU parallel execution if available
        if self.enable_gpu and self.gpu_optimizer:
            try:
                logger.info("Using GPU-accelerated parallel execution")
                return self.gpu_optimizer.gpu_optimize_parallel(
                    daily_matrix, portfolio_size, metric, algorithms, max_workers
                )
            except Exception as e:
                logger.warning(f"GPU parallel execution failed, using CPU: {e}")
        
        # CPU fallback
        if self.cpu_optimizer:
            logger.info("Using CPU parallel execution")
            return self.cpu_optimizer.optimize_parallel(
                daily_matrix, portfolio_size, metric, algorithms
            )
        else:
            raise RuntimeError("No optimizer available")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for GPU vs CPU"""
        metrics = {
            'gpu_enabled': self.enable_gpu,
            'gpu_available': self.gpu_optimizer is not None,
            'cpu_available': self.cpu_optimizer is not None,
            'connection_pool_size': self.connection_pool_size
        }
        
        if self.enable_gpu and self.gpu_optimizer:
            # Add GPU-specific metrics
            metrics.update({
                'gpu_algorithms': [
                    'gpu_genetic_algorithm',
                    'gpu_particle_swarm_optimization',
                    'gpu_differential_evolution',
                    'gpu_simulated_annealing',
                    'gpu_ant_colony_optimization',
                    'gpu_hill_climbing',
                    'gpu_bayesian_optimization'
                ],
                'expected_speedup': {
                    'genetic_algorithm': '5x',
                    'particle_swarm_optimization': '6x',
                    'differential_evolution': '6x',
                    'simulated_annealing': '5x',
                    'ant_colony_optimization': '6x',
                    'hill_climbing': '5x',
                    'bayesian_optimization': '5x'
                }
            })
        
        return metrics
    
    def benchmark_gpu_performance(self, daily_matrix: np.ndarray, portfolio_size: int = 20,
                                 metric: str = 'ratio') -> Dict[str, Any]:
        """Benchmark GPU vs CPU performance"""
        if not (self.enable_gpu and self.gpu_optimizer):
            return {'error': 'GPU acceleration not available'}
        
        try:
            return self.gpu_optimizer.benchmark_gpu_vs_cpu(daily_matrix, portfolio_size, metric)
        except Exception as e:
            return {'error': f'Benchmark failed: {e}'}

# Backward compatibility - create alias for existing production code
FixedCompleteEnhancedOptimizer = GPUProductionEnhancedOptimizer

def main():
    """Test the GPU production optimizer"""
    print("🚀 GPU-ENHANCED PRODUCTION OPTIMIZER TEST")
    print("=" * 60)
    
    # Create test data
    test_data = np.random.randn(100, 50)  # 100 days, 50 strategies
    
    # Initialize optimizer
    optimizer = GPUProductionEnhancedOptimizer(connection_pool_size=3, enable_gpu=True)
    
    # Get performance metrics
    metrics = optimizer.get_performance_metrics()
    print(f"GPU Enabled: {metrics['gpu_enabled']}")
    print(f"GPU Available: {metrics['gpu_available']}")
    print(f"CPU Available: {metrics['cpu_available']}")
    
    # Test a quick algorithm
    try:
        print("\nTesting genetic algorithm...")
        start_time = time.time()
        result = optimizer.genetic_algorithm(test_data, 10, 'ratio', generations=5)
        execution_time = time.time() - start_time
        
        print(f"✅ Success: fitness={result.fitness:.6f}, time={execution_time:.2f}s")
        if hasattr(result, 'gpu_accelerated'):
            print(f"GPU Accelerated: {result.gpu_accelerated}")
            print(f"GPU Speedup: {result.gpu_speedup:.1f}x")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
    
    print("\n🎉 GPU Production Optimizer Ready")

if __name__ == "__main__":
    main()
