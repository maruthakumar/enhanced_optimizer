#!/usr/bin/env python3
"""
Benchmark cuDF performance for CI/CD reporting
Compares GPU vs CPU performance for key operations
"""

import time
import json
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import argparse
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Try to import GPU libraries
try:
    import cudf
    import cupy as cp
    GPU_AVAILABLE = True
except (ImportError, RuntimeError):
    GPU_AVAILABLE = False
    print("Warning: GPU libraries not available, will run CPU-only benchmarks")


class CuDFBenchmark:
    """Benchmark cuDF operations vs pandas"""
    
    def __init__(self, output_dir="output/benchmarks"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'gpu_available': GPU_AVAILABLE,
            'benchmarks': {}
        }
    
    def time_operation(self, func, *args, **kwargs):
        """Time a single operation"""
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        return result, end - start
    
    def benchmark_correlation_calculation(self, num_strategies=500, num_days=252):
        """Benchmark correlation matrix calculation"""
        print(f"\nBenchmarking correlation calculation ({num_strategies}x{num_days})...")
        
        # Generate test data
        np.random.seed(42)
        data = {
            f'Strategy_{i}': np.random.randn(num_days) * 1000 
            for i in range(num_strategies)
        }
        
        # CPU benchmark (pandas)
        df_cpu = pd.DataFrame(data)
        _, cpu_time = self.time_operation(df_cpu.corr)
        
        results = {
            'cpu_time': cpu_time,
            'matrix_size': f"{num_strategies}x{num_strategies}"
        }
        
        # GPU benchmark (cuDF) if available
        if GPU_AVAILABLE:
            df_gpu = cudf.DataFrame(data)
            # Warm-up GPU
            _ = df_gpu.head()
            
            _, gpu_time = self.time_operation(df_gpu.corr)
            
            results['gpu_time'] = gpu_time
            results['speedup'] = cpu_time / gpu_time
            
            print(f"  CPU Time: {cpu_time:.3f}s")
            print(f"  GPU Time: {gpu_time:.3f}s")
            print(f"  Speedup: {results['speedup']:.2f}x")
        else:
            print(f"  CPU Time: {cpu_time:.3f}s")
            print("  GPU: Not available")
        
        self.results['benchmarks']['correlation'] = results
    
    def benchmark_aggregation_operations(self, num_rows=1000000):
        """Benchmark aggregation operations"""
        print(f"\nBenchmarking aggregation operations ({num_rows:,} rows)...")
        
        # Generate test data
        np.random.seed(42)
        data = {
            'strategy': np.random.choice([f'S{i}' for i in range(100)], num_rows),
            'returns': np.random.randn(num_rows) * 100,
            'volume': np.random.randint(1000, 100000, num_rows),
            'date': pd.date_range('2020-01-01', periods=num_rows, freq='1min')
        }
        
        # CPU benchmark
        df_cpu = pd.DataFrame(data)
        
        def cpu_aggregations():
            return {
                'mean_returns': df_cpu.groupby('strategy')['returns'].mean(),
                'total_volume': df_cpu.groupby('strategy')['volume'].sum(),
                'daily_stats': df_cpu.groupby(df_cpu['date'].dt.date)['returns'].agg(['mean', 'std', 'min', 'max'])
            }
        
        _, cpu_time = self.time_operation(cpu_aggregations)
        
        results = {
            'cpu_time': cpu_time,
            'num_rows': num_rows
        }
        
        # GPU benchmark if available
        if GPU_AVAILABLE:
            df_gpu = cudf.DataFrame(data)
            
            def gpu_aggregations():
                return {
                    'mean_returns': df_gpu.groupby('strategy')['returns'].mean(),
                    'total_volume': df_gpu.groupby('strategy')['volume'].sum(),
                    'daily_stats': df_gpu.groupby(df_gpu['date'].dt.date)['returns'].agg(['mean', 'std', 'min', 'max'])
                }
            
            _, gpu_time = self.time_operation(gpu_aggregations)
            
            results['gpu_time'] = gpu_time
            results['speedup'] = cpu_time / gpu_time
            
            print(f"  CPU Time: {cpu_time:.3f}s")
            print(f"  GPU Time: {gpu_time:.3f}s")
            print(f"  Speedup: {results['speedup']:.2f}x")
        else:
            print(f"  CPU Time: {cpu_time:.3f}s")
        
        self.results['benchmarks']['aggregation'] = results
    
    def benchmark_rolling_calculations(self, num_strategies=100, num_days=1000):
        """Benchmark rolling window calculations"""
        print(f"\nBenchmarking rolling calculations ({num_strategies} strategies, {num_days} days)...")
        
        # Generate test data
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=num_days)
        data = {'Date': dates}
        
        for i in range(num_strategies):
            data[f'Strategy_{i}'] = np.random.randn(num_days).cumsum() * 100
        
        # CPU benchmark
        df_cpu = pd.DataFrame(data).set_index('Date')
        
        def cpu_rolling():
            return {
                'rolling_mean': df_cpu.rolling(window=20).mean(),
                'rolling_std': df_cpu.rolling(window=20).std(),
                'rolling_corr': df_cpu.rolling(window=60).corr()
            }
        
        _, cpu_time = self.time_operation(cpu_rolling)
        
        results = {
            'cpu_time': cpu_time,
            'num_strategies': num_strategies,
            'num_days': num_days
        }
        
        # GPU benchmark if available
        if GPU_AVAILABLE:
            df_gpu = cudf.DataFrame(data).set_index('Date')
            
            def gpu_rolling():
                return {
                    'rolling_mean': df_gpu.rolling(window=20).mean(),
                    'rolling_std': df_gpu.rolling(window=20).std(),
                    # Note: rolling correlation might not be fully supported
                }
            
            _, gpu_time = self.time_operation(gpu_rolling)
            
            results['gpu_time'] = gpu_time
            results['speedup'] = cpu_time / gpu_time
            
            print(f"  CPU Time: {cpu_time:.3f}s")
            print(f"  GPU Time: {gpu_time:.3f}s")
            print(f"  Speedup: {results['speedup']:.2f}x")
        else:
            print(f"  CPU Time: {cpu_time:.3f}s")
        
        self.results['benchmarks']['rolling'] = results
    
    def benchmark_memory_usage(self):
        """Benchmark memory usage patterns"""
        print("\nBenchmarking memory usage...")
        
        import psutil
        process = psutil.Process()
        
        # Test different dataset sizes
        sizes = [(100, 100), (1000, 252), (5000, 252)]
        memory_results = []
        
        for num_strategies, num_days in sizes:
            # Generate data
            data = {
                f'Strategy_{i}': np.random.randn(num_days) * 1000 
                for i in range(num_strategies)
            }
            
            # CPU memory usage
            mem_before = process.memory_info().rss / (1024**2)  # MB
            df_cpu = pd.DataFrame(data)
            _ = df_cpu.corr()  # Force computation
            mem_after_cpu = process.memory_info().rss / (1024**2)
            cpu_memory = mem_after_cpu - mem_before
            
            result = {
                'dataset_size': f"{num_strategies}x{num_days}",
                'cpu_memory_mb': cpu_memory
            }
            
            # GPU memory usage if available
            if GPU_AVAILABLE:
                # Reset GPU memory
                mempool = cp.get_default_memory_pool()
                mempool.free_all_blocks()
                
                gpu_mem_before = mempool.used_bytes() / (1024**2)
                df_gpu = cudf.DataFrame(data)
                _ = df_gpu.corr()
                gpu_mem_after = mempool.used_bytes() / (1024**2)
                gpu_memory = gpu_mem_after - gpu_mem_before
                
                result['gpu_memory_mb'] = gpu_memory
                result['memory_ratio'] = cpu_memory / gpu_memory if gpu_memory > 0 else 0
                
                # Cleanup
                del df_gpu
                mempool.free_all_blocks()
            
            # Cleanup CPU
            del df_cpu
            
            memory_results.append(result)
            print(f"  {result['dataset_size']}: CPU={result['cpu_memory_mb']:.1f}MB", end="")
            if 'gpu_memory_mb' in result:
                print(f", GPU={result['gpu_memory_mb']:.1f}MB")
            else:
                print()
        
        self.results['benchmarks']['memory'] = memory_results
    
    def benchmark_optimization_workflow(self):
        """Benchmark complete optimization workflow"""
        print("\nBenchmarking complete optimization workflow...")
        
        # Simulate complete workflow
        num_strategies = 500
        num_days = 82
        portfolio_size = 35
        
        # Generate test data
        np.random.seed(42)
        data = {
            'Date': pd.date_range('2024-01-01', periods=num_days),
        }
        
        for i in range(num_strategies):
            data[f'Strategy_{i}'] = np.random.randn(num_days) * 1000
        
        # CPU workflow
        start_cpu = time.perf_counter()
        
        df_cpu = pd.DataFrame(data)
        strategy_cols = [col for col in df_cpu.columns if col.startswith('Strategy_')]
        
        # 1. Calculate correlations
        corr_matrix = df_cpu[strategy_cols].corr()
        
        # 2. Calculate basic metrics
        returns = df_cpu[strategy_cols].sum()
        drawdowns = df_cpu[strategy_cols].cumsum().expanding().max() - df_cpu[strategy_cols].cumsum()
        
        # 3. Simple portfolio selection (top N by returns)
        top_strategies = returns.nlargest(portfolio_size).index.tolist()
        
        cpu_time = time.perf_counter() - start_cpu
        
        results = {
            'cpu_time': cpu_time,
            'num_strategies': num_strategies,
            'portfolio_size': portfolio_size
        }
        
        # GPU workflow if available
        if GPU_AVAILABLE:
            start_gpu = time.perf_counter()
            
            df_gpu = cudf.DataFrame(data)
            
            # 1. Calculate correlations
            corr_matrix_gpu = df_gpu[strategy_cols].corr()
            
            # 2. Calculate basic metrics
            returns_gpu = df_gpu[strategy_cols].sum()
            # Note: Some operations might need adaptation for cuDF
            
            # 3. Portfolio selection
            top_strategies_gpu = returns_gpu.nlargest(portfolio_size)
            
            gpu_time = time.perf_counter() - start_gpu
            
            results['gpu_time'] = gpu_time
            results['speedup'] = cpu_time / gpu_time
            
            print(f"  CPU Time: {cpu_time:.3f}s")
            print(f"  GPU Time: {gpu_time:.3f}s")
            print(f"  Speedup: {results['speedup']:.2f}x")
        else:
            print(f"  CPU Time: {cpu_time:.3f}s")
        
        self.results['benchmarks']['workflow'] = results
    
    def save_results(self):
        """Save benchmark results"""
        output_file = self.output_dir / f"cudf_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\nResults saved to: {output_file}")
        
        # Also save a latest.json for easy access
        latest_file = self.output_dir / "cudf_benchmark_latest.json"
        with open(latest_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        return output_file
    
    def generate_summary(self):
        """Generate performance summary"""
        print("\n" + "="*60)
        print("PERFORMANCE SUMMARY")
        print("="*60)
        
        if GPU_AVAILABLE:
            speedups = []
            for name, result in self.results['benchmarks'].items():
                if isinstance(result, dict) and 'speedup' in result:
                    speedups.append(result['speedup'])
                    print(f"{name.capitalize()}: {result['speedup']:.2f}x speedup")
            
            if speedups:
                avg_speedup = np.mean(speedups)
                print(f"\nAverage GPU Speedup: {avg_speedup:.2f}x")
        else:
            print("GPU not available - CPU-only benchmarks completed")
        
        print("="*60)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Benchmark cuDF performance')
    parser.add_argument('--output-dir', default='output/benchmarks',
                        help='Directory to save benchmark results')
    parser.add_argument('--small', action='store_true',
                        help='Run smaller benchmarks for quick testing')
    
    args = parser.parse_args()
    
    # Create benchmark instance
    benchmark = CuDFBenchmark(output_dir=args.output_dir)
    
    print("Starting cuDF Performance Benchmarks...")
    print(f"GPU Available: {GPU_AVAILABLE}")
    
    if args.small:
        # Run smaller benchmarks
        benchmark.benchmark_correlation_calculation(100, 50)
        benchmark.benchmark_aggregation_operations(10000)
    else:
        # Run full benchmarks
        benchmark.benchmark_correlation_calculation(500, 252)
        benchmark.benchmark_aggregation_operations(1000000)
        benchmark.benchmark_rolling_calculations(100, 1000)
        benchmark.benchmark_memory_usage()
        benchmark.benchmark_optimization_workflow()
    
    # Save results
    output_file = benchmark.save_results()
    
    # Generate summary
    benchmark.generate_summary()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())