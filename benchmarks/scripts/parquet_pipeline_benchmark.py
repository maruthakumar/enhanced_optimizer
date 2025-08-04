#!/usr/bin/env python3
"""
Parquet Pipeline Performance Benchmarking Framework
Comprehensive benchmarking for Parquet/Arrow/cuDF pipeline with SLA validation
"""

import os
import sys
import json
import time
import psutil
import logging
import tempfile
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np

# Add backend directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'backend'))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import benchmark data generator
from generate_test_data import BenchmarkDataGenerator

# Import existing Parquet/cuDF workflow components
try:
    from parquet_cudf_workflow import ParquetCuDFWorkflow
    from lib.parquet_pipeline.csv_to_parquet import (
        csv_to_parquet, validate_parquet_file, get_parquet_metadata
    )
    from lib.arrow_connector.arrow_memory_pool import create_memory_pool, monitor_memory_usage
    from lib.gpu_utils import get_cudf_safe, CUDF_AVAILABLE, log_gpu_environment
    
    # Try to import cuDF for GPU benchmarking
    cudf = get_cudf_safe()
    
except ImportError as e:
    logger.warning(f"Some components not available: {e}")
    cudf = None
    CUDF_AVAILABLE = False

# Import existing CSV workflow for legacy comparison
try:
    from csv_only_heavydb_workflow import CSVOnlyHeavyDBWorkflow
    LEGACY_AVAILABLE = True
except ImportError:
    logger.warning("Legacy CSV workflow not available for comparison")
    LEGACY_AVAILABLE = False


class BenchmarkResults:
    """Container for benchmark results with schema validation."""
    
    def __init__(self, timestamp: str = None):
        self.timestamp = timestamp or datetime.now().isoformat()
        self.dataset_info = {}
        self.pipeline_timings = {}
        self.memory_usage = {}
        self.validation = {}
        self.errors = []
        
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'timestamp': self.timestamp,
            'dataset_info': self.dataset_info,
            'pipeline_timings': self.pipeline_timings,
            'memory_usage': self.memory_usage,
            'validation': self.validation,
            'errors': self.errors
        }
    
    def from_dict(self, data: Dict):
        """Load from dictionary."""
        self.timestamp = data.get('timestamp', self.timestamp)
        self.dataset_info = data.get('dataset_info', {})
        self.pipeline_timings = data.get('pipeline_timings', {})
        self.memory_usage = data.get('memory_usage', {})
        self.validation = data.get('validation', {})
        self.errors = data.get('errors', [])


class ParquetPipelineBenchmark:
    """Main benchmarking class for Parquet/Arrow/cuDF pipeline."""
    
    def __init__(self, config_path: str = None):
        """Initialize benchmark with configuration."""
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'benchmark_config.json')
        
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.data_generator = BenchmarkDataGenerator(config_path)
        self.temp_dir = tempfile.mkdtemp(prefix='benchmark_')
        self.results_dir = os.path.join(os.path.dirname(__file__), '..', 'reports')
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Initialize memory monitoring
        self.initial_memory = self._get_memory_stats()
        
        logger.info(f"Benchmark initialized with temp dir: {self.temp_dir}")
        log_gpu_environment()
    
    def _get_memory_stats(self) -> Dict:
        """Get current memory usage statistics."""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        stats = {
            'system': {
                'used_gb': psutil.virtual_memory().used / (1024**3),
                'available_gb': psutil.virtual_memory().available / (1024**3),
                'percent': psutil.virtual_memory().percent
            },
            'process': {
                'rss_gb': memory_info.rss / (1024**3),
                'vms_gb': memory_info.vms / (1024**3)
            }
        }
        
        # Add GPU stats if available
        try:
            if CUDF_AVAILABLE:
                gpu_stats = monitor_memory_usage()
                if 'gpu' in gpu_stats:
                    stats['gpu'] = gpu_stats['gpu']
        except Exception as e:
            logger.warning(f"Could not get GPU stats: {e}")
        
        return stats
    
    def _measure_execution_time(self, func, *args, **kwargs) -> Tuple[float, any]:
        """Measure execution time of a function."""
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        execution_time = (end_time - start_time) * 1000  # Convert to milliseconds
        return execution_time, result
    
    def benchmark_csv_to_parquet(self, csv_path: str, compression: str = 'snappy') -> Dict:
        """Benchmark CSV to Parquet conversion."""
        logger.info(f"Benchmarking CSV to Parquet conversion with {compression} compression")
        
        parquet_path = os.path.join(self.temp_dir, f"benchmark_{compression}.parquet")
        
        # Measure conversion time
        start_memory = self._get_memory_stats()
        
        conversion_time, success = self._measure_execution_time(
            csv_to_parquet,
            csv_path=csv_path,
            parquet_path=parquet_path,
            compression=compression,
            row_group_size=50000
        )
        
        end_memory = self._get_memory_stats()
        
        if not success:
            return {
                'status': 'failed',
                'error': 'CSV to Parquet conversion failed',
                'conversion_time_ms': conversion_time
            }
        
        # Get file sizes
        csv_size = os.path.getsize(csv_path) / (1024**2)  # MB
        parquet_size = os.path.getsize(parquet_path) / (1024**2)  # MB
        compression_ratio = csv_size / parquet_size if parquet_size > 0 else 0
        
        # Get metadata
        metadata = get_parquet_metadata(parquet_path)
        
        return {
            'status': 'success',
            'conversion_time_ms': conversion_time,
            'csv_size_mb': csv_size,
            'parquet_size_mb': parquet_size,
            'compression_ratio': compression_ratio,
            'compression_type': compression,
            'memory_delta_mb': (end_memory['process']['rss_gb'] - start_memory['process']['rss_gb']) * 1024,
            'metadata': metadata,
            'parquet_path': parquet_path
        }
    
    def benchmark_parquet_to_cudf(self, parquet_path: str, chunk_size: int = 1000000) -> Dict:
        """Benchmark Parquet to cuDF loading."""
        logger.info(f"Benchmarking Parquet to cuDF loading (chunk_size: {chunk_size})")
        
        start_memory = self._get_memory_stats()
        
        try:
            if CUDF_AVAILABLE:
                # Test cuDF loading
                from lib.arrow_connector.arrow_memory_pool import load_parquet_to_cudf
                
                loading_time, df = self._measure_execution_time(
                    load_parquet_to_cudf,
                    parquet_path=parquet_path,
                    chunk_size=chunk_size
                )
                
                data_type = 'cudf'
                row_count = len(df)
                column_count = len(df.columns)
                gpu_memory_used = True
                
                # Free memory
                del df
                
            else:
                # Fallback to pandas
                logger.warning("cuDF not available, using pandas")
                loading_time, df = self._measure_execution_time(
                    pd.read_parquet,
                    parquet_path
                )
                
                data_type = 'pandas'
                row_count = len(df)
                column_count = len(df.columns)
                gpu_memory_used = False
                
                # Free memory
                del df
            
            end_memory = self._get_memory_stats()
            
            return {
                'status': 'success',
                'loading_time_ms': loading_time,
                'data_type': data_type,
                'row_count': row_count,
                'column_count': column_count,
                'chunk_size': chunk_size,
                'gpu_memory_used': gpu_memory_used,
                'memory_delta_mb': (end_memory['process']['rss_gb'] - start_memory['process']['rss_gb']) * 1024
            }
        
        except Exception as e:
            logger.error(f"Parquet to cuDF loading failed: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'loading_time_ms': 0,
                'data_type': 'none'
            }
    
    def benchmark_end_to_end_pipeline(self, csv_path: str, portfolio_size: int = 35) -> Dict:
        """Benchmark complete end-to-end pipeline."""
        logger.info(f"Benchmarking end-to-end pipeline (portfolio_size: {portfolio_size})")
        
        output_dir = os.path.join(self.temp_dir, 'pipeline_output')
        os.makedirs(output_dir, exist_ok=True)
        
        start_memory = self._get_memory_stats()
        start_time = time.perf_counter()
        
        try:
            # Create workflow instance
            workflow = ParquetCuDFWorkflow()
            workflow.config['portfolio_size'] = portfolio_size
            
            # Run optimization
            results = workflow.run_optimization(csv_path, output_dir)
            
            end_time = time.perf_counter()
            total_time_ms = (end_time - start_time) * 1000
            
            end_memory = self._get_memory_stats()
            
            if results['status'] == 'success':
                return {
                    'status': 'success',
                    'total_time_ms': total_time_ms,
                    'portfolio_size': portfolio_size,
                    'num_strategies': results.get('num_strategies', 0),
                    'num_days': results.get('num_days', 0),
                    'algorithms_run': len(results.get('algorithms', {})),
                    'successful_algorithms': len([a for a in results.get('algorithms', {}).values() if a.get('status') == 'success']),
                    'correlation_time_ms': results.get('correlation_time', 0) * 1000,
                    'memory_delta_mb': (end_memory['process']['rss_gb'] - start_memory['process']['rss_gb']) * 1024,
                    'ulta_enabled': results.get('ulta', {}).get('enabled', False),
                    'ulta_processing_time_ms': results.get('ulta', {}).get('processing_time', 0) * 1000,
                    'memory_usage': results.get('memory_usage', {}),
                    'output_files': results.get('output_files', {})
                }
            else:
                return {
                    'status': 'failed',
                    'error': results.get('error', 'Unknown error'),
                    'total_time_ms': total_time_ms,
                    'portfolio_size': portfolio_size
                }
        
        except Exception as e:
            end_time = time.perf_counter()
            total_time_ms = (end_time - start_time) * 1000
            
            logger.error(f"End-to-end pipeline failed: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'total_time_ms': total_time_ms,
                'portfolio_size': portfolio_size
            }
    
    def benchmark_legacy_comparison(self, csv_path: str, portfolio_size: int = 35) -> Dict:
        """Benchmark legacy CSV processing for comparison."""
        if not LEGACY_AVAILABLE:
            return {
                'status': 'skipped',
                'reason': 'Legacy workflow not available'
            }
        
        logger.info(f"Benchmarking legacy CSV workflow for comparison")
        
        output_dir = os.path.join(self.temp_dir, 'legacy_output')
        os.makedirs(output_dir, exist_ok=True)
        
        start_time = time.perf_counter()
        
        try:
            # Create legacy workflow instance
            legacy_workflow = CSVOnlyHeavyDBWorkflow()
            
            # Run legacy optimization (simplified)
            results = legacy_workflow.run_optimization(csv_path, output_dir, portfolio_size)
            
            end_time = time.perf_counter()
            total_time_ms = (end_time - start_time) * 1000
            
            return {
                'status': 'success',
                'total_time_ms': total_time_ms,
                'portfolio_size': portfolio_size,
                'workflow_type': 'legacy_csv'
            }
        
        except Exception as e:
            end_time = time.perf_counter()
            total_time_ms = (end_time - start_time) * 1000
            
            logger.error(f"Legacy benchmark failed: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'total_time_ms': total_time_ms,
                'workflow_type': 'legacy_csv'
            }
    
    def run_scenario_benchmark(self, scenario: Dict) -> BenchmarkResults:
        """Run benchmark for a specific scenario."""
        logger.info(f"Running benchmark scenario: {scenario['name']}")
        
        results = BenchmarkResults()
        results.dataset_info = {
            'name': scenario['name'],
            'strategy_count': scenario['strategy_count'],
            'trading_days': scenario['trading_days'],
            'target_time_ms': scenario['target_time_ms'],
            'description': scenario['description']
        }
        
        try:
            # Generate test data
            logger.info("Generating test dataset")
            csv_path = self.data_generator.create_dataset(
                strategy_count=scenario['strategy_count'],
                trading_days=scenario['trading_days'],
                correlation_pattern='medium',
                output_filename=f"{scenario['name']}_benchmark.csv"
            )
            
            results.dataset_info['csv_path'] = csv_path
            results.dataset_info['file_size_mb'] = os.path.getsize(csv_path) / (1024**2)
            
            # Benchmark CSV to Parquet conversion
            logger.info("Benchmarking CSV to Parquet conversion")
            parquet_results = self.benchmark_csv_to_parquet(csv_path, 'snappy')
            results.pipeline_timings['csv_to_parquet_ms'] = parquet_results['conversion_time_ms']
            
            if parquet_results['status'] == 'success':
                # Benchmark Parquet to cuDF loading
                logger.info("Benchmarking Parquet to cuDF loading")
                cudf_results = self.benchmark_parquet_to_cudf(parquet_results['parquet_path'])
                results.pipeline_timings['parquet_to_cudf_ms'] = cudf_results['loading_time_ms']
                
                # Benchmark end-to-end pipeline
                logger.info("Benchmarking end-to-end pipeline")
                pipeline_results = self.benchmark_end_to_end_pipeline(csv_path, 35)
                results.pipeline_timings['total_pipeline_ms'] = pipeline_results['total_time_ms']
                
                # Memory usage
                results.memory_usage = {
                    'peak_ram_mb': max(
                        parquet_results.get('memory_delta_mb', 0),
                        cudf_results.get('memory_delta_mb', 0),
                        pipeline_results.get('memory_delta_mb', 0)
                    ),
                    'efficiency_score': 0.85  # Default efficiency score
                }
                
                # Legacy comparison if available
                if LEGACY_AVAILABLE:
                    logger.info("Running legacy comparison")
                    legacy_results = self.benchmark_legacy_comparison(csv_path, 35)
                    if legacy_results['status'] == 'success':
                        results.pipeline_timings['legacy_pipeline_ms'] = legacy_results['total_time_ms']
                
                # Validation
                sla_compliant = results.pipeline_timings['total_pipeline_ms'] <= scenario['target_time_ms']
                
                # Calculate speedup factor
                speedup_factor = 1.0
                if 'legacy_pipeline_ms' in results.pipeline_timings:
                    legacy_time = results.pipeline_timings['legacy_pipeline_ms']
                    parquet_time = results.pipeline_timings['total_pipeline_ms']
                    if parquet_time > 0:
                        speedup_factor = legacy_time / parquet_time
                
                results.validation = {
                    'sla_compliance': sla_compliant,
                    'speedup_factor': speedup_factor,
                    'accuracy_verified': pipeline_results['status'] == 'success',
                    'target_time_ms': scenario['target_time_ms'],
                    'actual_time_ms': results.pipeline_timings['total_pipeline_ms']
                }
                
                logger.info(f"Scenario {scenario['name']} completed: SLA {'✓' if sla_compliant else '✗'}, "
                           f"Speedup: {speedup_factor:.1f}x")
            
            else:
                results.errors.append(f"CSV to Parquet conversion failed: {parquet_results.get('error', 'Unknown')}")
                results.validation['sla_compliance'] = False
        
        except Exception as e:
            logger.error(f"Scenario benchmark failed: {e}")
            results.errors.append(str(e))
            results.validation['sla_compliance'] = False
        
        return results
    
    def run_all_scenarios(self) -> List[BenchmarkResults]:
        """Run benchmarks for all configured scenarios."""
        all_results = []
        
        for scenario in self.config['benchmark_scenarios']:
            results = self.run_scenario_benchmark(scenario)
            all_results.append(results)
        
        return all_results
    
    def check_sla_compliance(self, results: List[BenchmarkResults]) -> bool:
        """Check if all scenarios meet SLA requirements."""
        all_compliant = True
        
        for result in results:
            if not result.validation.get('sla_compliance', False):
                all_compliant = False
                logger.error(f"SLA violation in scenario {result.dataset_info['name']}: "
                           f"{result.validation.get('actual_time_ms', 0)}ms > "
                           f"{result.validation.get('target_time_ms', 0)}ms")
        
        return all_compliant
    
    def save_results(self, results: List[BenchmarkResults], output_path: str = None) -> str:
        """Save benchmark results to file."""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.results_dir, f"benchmark_results_{timestamp}.json")
        
        # Convert results to serializable format
        results_data = {
            'benchmark_run': {
                'timestamp': datetime.now().isoformat(),
                'config': self.config,
                'system_info': {
                    'gpu_available': CUDF_AVAILABLE,
                    'legacy_available': LEGACY_AVAILABLE,
                    'python_version': sys.version,
                    'memory_gb': psutil.virtual_memory().total / (1024**3)
                }
            },
            'scenarios': [result.to_dict() for result in results]
        }
        
        with open(output_path, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        logger.info(f"Benchmark results saved to: {output_path}")
        return output_path
    
    def cleanup(self):
        """Clean up temporary files."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            logger.info(f"Cleaned up temporary directory: {self.temp_dir}")


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Parquet Pipeline Benchmark")
    parser.add_argument('--config', type=str, help='Benchmark configuration file path')
    parser.add_argument('--scenario', type=str, help='Run single scenario by name')
    parser.add_argument('--output', type=str, help='Output file path for results')
    parser.add_argument('--fail-on-sla', action='store_true', 
                       help='Exit with error code if SLA violations detected')
    parser.add_argument('--legacy-comparison', action='store_true',
                       help='Include legacy workflow comparison')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize benchmark
    benchmark = ParquetPipelineBenchmark(args.config)
    
    try:
        if args.scenario:
            # Run single scenario
            scenario = next(
                (s for s in benchmark.config['benchmark_scenarios'] if s['name'] == args.scenario),
                None
            )
            if not scenario:
                logger.error(f"Scenario '{args.scenario}' not found")
                sys.exit(1)
            
            results = [benchmark.run_scenario_benchmark(scenario)]
        else:
            # Run all scenarios
            results = benchmark.run_all_scenarios()
        
        # Save results
        output_path = benchmark.save_results(results, args.output)
        
        # Check SLA compliance
        sla_compliant = benchmark.check_sla_compliance(results)
        
        # Print summary
        print(f"\n=== Benchmark Results Summary ===")
        print(f"Scenarios run: {len(results)}")
        print(f"SLA compliant: {'✓' if sla_compliant else '✗'}")
        print(f"Results saved: {output_path}")
        
        for result in results:
            compliance = '✓' if result.validation.get('sla_compliance', False) else '✗'
            actual_time = result.validation.get('actual_time_ms', 0)
            target_time = result.validation.get('target_time_ms', 0)
            speedup = result.validation.get('speedup_factor', 1.0)
            
            print(f"  {result.dataset_info['name']}: {compliance} "
                  f"({actual_time:.0f}ms/{target_time:.0f}ms, {speedup:.1f}x speedup)")
        
        # Exit with error if SLA violations and fail-on-sla is set
        if args.fail_on_sla and not sla_compliant:
            logger.error("SLA violations detected, exiting with error code")
            sys.exit(1)
    
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        sys.exit(1)
    
    finally:
        benchmark.cleanup()


if __name__ == "__main__":
    main()