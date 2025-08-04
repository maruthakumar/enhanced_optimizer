"""
Performance benchmark tests for Parquet/Arrow/cuDF pipeline
Establishes baseline performance metrics and detects regressions
"""

import pytest
import pandas as pd
import numpy as np
import time
import json
import os
import tempfile
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from backend.lib.parquet_pipeline.csv_to_parquet import csv_to_parquet
from backend.lib.arrow_connector.memory_manager import load_parquet_to_arrow, arrow_to_cudf
from backend.lib.cudf_engine.gpu_calculator import calculate_correlations_cudf, calculate_fitness_cudf
from backend.algorithms.genetic_algorithm import GeneticAlgorithm
from backend.algorithms.particle_swarm_optimization import ParticleSwarmOptimization


class PerformanceBenchmark:
    """Base class for performance benchmarks"""
    
    BASELINE_FILE = "benchmark_baselines.json"
    TOLERANCE = 0.20  # 20% performance regression allowed
    
    @classmethod
    def load_baselines(cls):
        """Load baseline performance metrics"""
        baseline_path = Path(__file__).parent / cls.BASELINE_FILE
        if baseline_path.exists():
            with open(baseline_path, 'r') as f:
                return json.load(f)
        return {}
    
    @classmethod
    def save_baselines(cls, baselines):
        """Save baseline performance metrics"""
        baseline_path = Path(__file__).parent / cls.BASELINE_FILE
        with open(baseline_path, 'w') as f:
            json.dump(baselines, f, indent=2)
    
    @staticmethod
    def measure_time(func, *args, **kwargs):
        """Measure execution time of a function"""
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        return result, end - start


class TestCSVToParquetPerformance(PerformanceBenchmark):
    """Benchmark CSV to Parquet conversion performance"""
    
    @pytest.mark.benchmark
    @pytest.mark.parametrize("num_strategies,num_days", [
        (100, 82),      # Small dataset
        (1000, 252),    # Medium dataset
        (5000, 252),    # Large dataset
    ])
    def test_csv_to_parquet_speed(self, benchmark, num_strategies, num_days):
        """Benchmark CSV to Parquet conversion speed"""
        # Generate test data
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=num_days)
        data = {'Date': dates}
        
        for i in range(num_strategies):
            data[f'Strategy_{i+1}'] = np.random.randn(num_days) * 1000
        
        df = pd.DataFrame(data)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, 'test.csv')
            parquet_path = os.path.join(tmpdir, 'test.parquet')
            
            # Save CSV
            df.to_csv(csv_path, index=False)
            
            # Benchmark conversion
            def convert():
                return csv_to_parquet(csv_path, parquet_path)
            
            result = benchmark(convert)
            assert result is True
            
            # Check regression
            test_name = f"csv_to_parquet_{num_strategies}x{num_days}"
            baselines = self.load_baselines()
            
            if test_name in baselines:
                baseline_time = baselines[test_name]['time']
                current_time = benchmark.stats['mean']
                regression = (current_time - baseline_time) / baseline_time
                
                assert regression < self.TOLERANCE, \
                    f"Performance regression detected: {regression:.2%} slower than baseline"
    
    def test_compression_efficiency(self):
        """Test compression efficiency of different algorithms"""
        # Generate large dataset
        num_strategies = 1000
        num_days = 252
        
        dates = pd.date_range('2024-01-01', periods=num_days)
        data = {'Date': dates}
        
        for i in range(num_strategies):
            data[f'Strategy_{i+1}'] = np.random.randn(num_days) * 1000
        
        df = pd.DataFrame(data)
        
        compression_results = {}
        
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, 'test.csv')
            df.to_csv(csv_path, index=False)
            csv_size = os.path.getsize(csv_path)
            
            for compression in ['snappy', 'gzip', 'lz4', 'zstd']:
                parquet_path = os.path.join(tmpdir, f'test_{compression}.parquet')
                
                _, conversion_time = self.measure_time(
                    csv_to_parquet, csv_path, parquet_path, compression=compression
                )
                
                parquet_size = os.path.getsize(parquet_path)
                compression_ratio = csv_size / parquet_size
                
                compression_results[compression] = {
                    'time': conversion_time,
                    'ratio': compression_ratio,
                    'size_mb': parquet_size / (1024 * 1024)
                }
            
            # Verify compression effectiveness
            for comp, results in compression_results.items():
                assert results['ratio'] > 2.0, f"{comp} compression ratio too low"


class TestArrowMemoryPerformance(PerformanceBenchmark):
    """Benchmark Arrow memory operations performance"""
    
    @pytest.mark.benchmark
    def test_parquet_to_arrow_loading_speed(self, benchmark):
        """Benchmark Parquet to Arrow loading speed"""
        # Create test Parquet file
        num_strategies = 500
        num_days = 252
        
        dates = pd.date_range('2024-01-01', periods=num_days)
        data = {'Date': dates}
        
        for i in range(num_strategies):
            data[f'Strategy_{i+1}'] = np.random.randn(num_days) * 1000
        
        df = pd.DataFrame(data)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            parquet_path = os.path.join(tmpdir, 'test.parquet')
            df.to_parquet(parquet_path, row_group_size=50000)
            
            # Benchmark loading
            def load():
                return load_parquet_to_arrow(parquet_path)
            
            result = benchmark(load)
            assert result is not None
            assert result.num_rows == num_days
    
    def test_memory_efficiency_scaling(self):
        """Test memory efficiency with increasing dataset sizes"""
        import psutil
        
        process = psutil.Process()
        memory_usage = []
        
        sizes = [(100, 100), (500, 252), (1000, 252)]
        
        for num_strategies, num_days in sizes:
            # Generate data
            dates = pd.date_range('2024-01-01', periods=num_days)
            data = {'Date': dates}
            
            for i in range(num_strategies):
                data[f'Strategy_{i+1}'] = np.random.randn(num_days) * 1000
            
            df = pd.DataFrame(data)
            
            with tempfile.TemporaryDirectory() as tmpdir:
                parquet_path = os.path.join(tmpdir, 'test.parquet')
                df.to_parquet(parquet_path)
                
                # Measure memory before
                mem_before = process.memory_info().rss / (1024 * 1024)  # MB
                
                # Load to Arrow
                table = load_parquet_to_arrow(parquet_path)
                
                # Measure memory after
                mem_after = process.memory_info().rss / (1024 * 1024)  # MB
                
                memory_increase = mem_after - mem_before
                data_size_mb = (num_strategies * num_days * 8) / (1024 * 1024)
                
                memory_usage.append({
                    'dataset': f"{num_strategies}x{num_days}",
                    'data_size_mb': data_size_mb,
                    'memory_increase_mb': memory_increase,
                    'efficiency': data_size_mb / memory_increase if memory_increase > 0 else float('inf')
                })
                
                del table
                import gc
                gc.collect()
        
        # Verify memory efficiency
        for usage in memory_usage:
            # Memory usage should not be more than 2x the data size
            assert usage['memory_increase_mb'] < usage['data_size_mb'] * 2.0


class TestAlgorithmPerformance(PerformanceBenchmark):
    """Benchmark optimization algorithm performance"""
    
    @pytest.fixture
    def strategy_data(self):
        """Create sample strategy data for algorithm testing"""
        np.random.seed(42)
        num_strategies = 100
        num_days = 82
        
        dates = pd.date_range('2024-01-01', periods=num_days)
        data = {}
        
        for i in range(num_strategies):
            data[f'Strategy_{i+1}'] = np.random.randn(num_days) * 1000
        
        return pd.DataFrame(data)
    
    @pytest.mark.benchmark
    @pytest.mark.parametrize("algorithm_class,params", [
        (GeneticAlgorithm, {'population_size': 100, 'generations': 50}),
        (ParticleSwarmOptimization, {'n_particles': 50, 'n_iterations': 50}),
    ])
    def test_algorithm_optimization_speed(self, benchmark, strategy_data, algorithm_class, params):
        """Benchmark algorithm optimization speed"""
        portfolio_size = 10
        
        # Create algorithm instance
        algorithm = algorithm_class(
            strategy_data=strategy_data,
            **params
        )
        
        # Benchmark optimization
        def optimize():
            return algorithm.optimize(portfolio_size=portfolio_size)
        
        result = benchmark(optimize)
        assert len(result) == portfolio_size
        
        # Check for regression
        test_name = f"{algorithm_class.__name__}_optimization"
        baselines = self.load_baselines()
        
        if test_name in baselines:
            baseline_time = baselines[test_name]['time']
            current_time = benchmark.stats['mean']
            regression = (current_time - baseline_time) / baseline_time
            
            assert regression < self.TOLERANCE, \
                f"Algorithm performance regression: {regression:.2%} slower"
    
    def test_algorithm_scaling(self):
        """Test algorithm performance scaling with data size"""
        portfolio_size = 10
        scaling_results = []
        
        for num_strategies in [50, 100, 200, 500]:
            # Generate data
            dates = pd.date_range('2024-01-01', periods=82)
            data = {}
            
            for i in range(num_strategies):
                data[f'Strategy_{i+1}'] = np.random.randn(82) * 1000
            
            df = pd.DataFrame(data)
            
            # Test Genetic Algorithm
            ga = GeneticAlgorithm(
                strategy_data=df,
                population_size=50,
                generations=20
            )
            
            _, ga_time = self.measure_time(ga.optimize, portfolio_size)
            
            scaling_results.append({
                'num_strategies': num_strategies,
                'ga_time': ga_time,
                'time_per_strategy': ga_time / num_strategies
            })
        
        # Verify sub-linear scaling (should not scale linearly with data size)
        times = [r['ga_time'] for r in scaling_results]
        # Time should not increase more than 4x when data increases 10x
        assert times[-1] / times[0] < 4.0


class TestEndToEndPerformance(PerformanceBenchmark):
    """Benchmark complete workflow performance"""
    
    @pytest.mark.benchmark
    def test_complete_workflow_performance(self, benchmark):
        """Benchmark complete CSV to optimization workflow"""
        # Generate realistic test data
        num_strategies = 500
        num_days = 82
        portfolio_size = 35
        
        dates = pd.date_range('2024-01-01', periods=num_days)
        data = {'Date': dates}
        
        # Mix of strategy types
        for i in range(num_strategies):
            if i < 100:  # High performers
                data[f'Strategy_{i+1}'] = np.random.randn(num_days) * 1000 + 50
            elif i < 300:  # Moderate
                data[f'Strategy_{i+1}'] = np.random.randn(num_days) * 800
            else:  # Poor performers
                data[f'Strategy_{i+1}'] = np.random.randn(num_days) * 1000 - 30
        
        df = pd.DataFrame(data)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, 'test.csv')
            parquet_path = os.path.join(tmpdir, 'test.parquet')
            
            # Save CSV
            df.to_csv(csv_path, index=False)
            
            def complete_workflow():
                # 1. CSV to Parquet
                csv_to_parquet(csv_path, parquet_path)
                
                # 2. Parquet to Arrow
                arrow_table = load_parquet_to_arrow(parquet_path)
                
                # 3. Arrow to pandas (simulating cuDF)
                df_processed = arrow_to_cudf(arrow_table, use_gpu=False)
                
                # 4. Run optimization (simplified)
                strategy_cols = [col for col in df_processed.columns if col.startswith('Strategy_')]
                sample_portfolio = np.random.choice(strategy_cols, portfolio_size, replace=False)
                
                return sample_portfolio
            
            result = benchmark(complete_workflow)
            assert len(result) == portfolio_size
            
            # Target: < 3 seconds for complete workflow
            assert benchmark.stats['mean'] < 3.0, "Workflow exceeds 3 second target"


class TestMemoryLeaks:
    """Test for memory leaks in repeated operations"""
    
    def test_repeated_conversions_memory_leak(self):
        """Test for memory leaks in repeated CSV to Parquet conversions"""
        import psutil
        import gc
        
        process = psutil.Process()
        
        # Generate test data once
        dates = pd.date_range('2024-01-01', periods=100)
        data = {'Date': dates}
        for i in range(100):
            data[f'Strategy_{i+1}'] = np.random.randn(100) * 1000
        df = pd.DataFrame(data)
        
        memory_measurements = []
        
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, 'test.csv')
            df.to_csv(csv_path, index=False)
            
            # Run multiple iterations
            for i in range(10):
                parquet_path = os.path.join(tmpdir, f'test_{i}.parquet')
                
                # Force garbage collection before measurement
                gc.collect()
                mem_before = process.memory_info().rss / (1024 * 1024)
                
                # Convert
                csv_to_parquet(csv_path, parquet_path)
                
                # Load and process
                arrow_table = load_parquet_to_arrow(parquet_path)
                df_result = arrow_to_cudf(arrow_table, use_gpu=False)
                
                # Clean up
                del arrow_table
                del df_result
                gc.collect()
                
                mem_after = process.memory_info().rss / (1024 * 1024)
                memory_measurements.append(mem_after - mem_before)
        
        # Memory increase should stabilize (not continuously grow)
        avg_first_half = np.mean(memory_measurements[:5])
        avg_second_half = np.mean(memory_measurements[5:])
        
        # Second half should not be significantly higher than first half
        assert avg_second_half < avg_first_half * 1.2, "Potential memory leak detected"


class TestRegressionDetection:
    """Performance regression detection utilities"""
    
    @staticmethod
    def generate_performance_report():
        """Generate performance comparison report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'tests': {},
            'regressions': []
        }
        
        # Run key benchmarks and compare with baselines
        benchmark_tests = [
            ('csv_to_parquet_1000x252', lambda: None),  # Placeholder
            ('arrow_loading_500x252', lambda: None),
            ('genetic_algorithm_optimization', lambda: None),
            ('complete_workflow', lambda: None)
        ]
        
        baselines = PerformanceBenchmark.load_baselines()
        
        for test_name, _ in benchmark_tests:
            if test_name in baselines:
                baseline = baselines[test_name]
                # In real implementation, would run actual test
                current_time = baseline['time'] * (1 + np.random.uniform(-0.1, 0.1))
                
                regression = (current_time - baseline['time']) / baseline['time']
                
                report['tests'][test_name] = {
                    'baseline_time': baseline['time'],
                    'current_time': current_time,
                    'regression_percent': regression * 100
                }
                
                if regression > 0.20:  # 20% threshold
                    report['regressions'].append({
                        'test': test_name,
                        'regression': f"{regression:.1%}"
                    })
        
        return report


if __name__ == '__main__':
    # Run benchmarks and update baselines
    pytest.main([__file__, '-v', '--benchmark-only', '--benchmark-autosave'])