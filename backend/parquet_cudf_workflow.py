#!/usr/bin/env python3
"""
Heavy Optimizer Platform - Parquet/Arrow/cuDF Accelerated Workflow
Modern GPU-accelerated data processing pipeline for portfolio optimization
"""

import time
import numpy as np
import pandas as pd
import os
import sys
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
from typing import List, Dict, Callable, Optional, Tuple
import logging
import argparse
import json
matplotlib.use('Agg')  # Use non-interactive backend for server execution

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add backend to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import new Parquet/Arrow/cuDF modules
from lib.parquet_pipeline import (
    csv_to_parquet,
    detect_csv_schema,
    validate_parquet_file,
    optimize_parquet_storage
)

from lib.arrow_connector import (
    load_parquet_to_arrow,
    arrow_to_cudf,
    create_memory_pool,
    monitor_memory_usage,
    load_parquet_to_cudf
)

from lib.cudf_engine import (
    calculate_correlations_cudf,
    calculate_fitness_cudf,
    calculate_roi_cudf,
    calculate_drawdown_cudf,
    calculate_win_rate_cudf,
    calculate_profit_factor_cudf,
    calculate_sharpe_ratio_cudf,
    calculate_sortino_ratio_cudf,
    calculate_calmar_ratio_cudf
)

# Check if cuDF is available
try:
    import cudf
    CUDF_AVAILABLE = True
    logger.info("✅ cuDF available for GPU acceleration")
except (ImportError, RuntimeError) as e:
    CUDF_AVAILABLE = False
    logger.warning(f"⚠️ cuDF not available ({str(e)}), falling back to CPU mode")

# Import algorithms (reuse existing implementations)
from algorithms.genetic_algorithm import GeneticAlgorithm
from algorithms.particle_swarm_optimization import ParticleSwarmOptimization
from algorithms.simulated_annealing import SimulatedAnnealing
from algorithms.differential_evolution import DifferentialEvolution
from algorithms.ant_colony_optimization import AntColonyOptimization
from algorithms.hill_climbing import HillClimbing
from algorithms.bayesian_optimization import BayesianOptimization
from algorithms.random_search import RandomSearch

# Import output generation engine
from output_generation_engine import OutputGenerationEngine

class ParquetCuDFWorkflow:
    """
    Main workflow class using Parquet/Arrow/cuDF instead of HeavyDB
    """
    
    def __init__(self, config_path: str = None):
        """Initialize workflow with configuration"""
        self.config = self._load_config(config_path)
        self.memory_pool = create_memory_pool(size_gb=4.0)
        self.parquet_dir = Path("/mnt/optimizer_share/data/parquet/strategies")
        self.parquet_dir.mkdir(parents=True, exist_ok=True)
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from file or use defaults"""
        default_config = {
            'portfolio_size': 35,
            'use_gpu': True,
            'chunk_size': 1000000,
            'compression': 'snappy',
            'correlation_threshold': 0.7,
            'fitness_weights': {
                'roi_dd_ratio_weight': 0.35,
                'total_roi_weight': 0.25,
                'max_drawdown_weight': 0.15,
                'win_rate_weight': 0.15,
                'profit_factor_weight': 0.10
            },
            'algorithm_timeouts': {
                'genetic_algorithm': 300,
                'particle_swarm': 300,
                'simulated_annealing': 300,
                'differential_evolution': 300,
                'ant_colony': 300,
                'hill_climbing': 60,
                'bayesian_optimization': 300,
                'random_search': 60
            }
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                loaded_config = json.load(f)
                default_config.update(loaded_config)
        
        return default_config
    
    def process_csv_to_parquet(self, csv_path: str) -> str:
        """
        Convert CSV to Parquet format with optimization
        
        Args:
            csv_path: Path to input CSV file
            
        Returns:
            Path to generated Parquet file
        """
        logger.info(f"Converting CSV to Parquet: {csv_path}")
        
        # Generate Parquet filename
        csv_name = Path(csv_path).stem
        parquet_path = self.parquet_dir / f"{csv_name}.parquet"
        
        # Check if Parquet already exists and is newer than CSV
        if parquet_path.exists():
            csv_mtime = os.path.getmtime(csv_path)
            parquet_mtime = os.path.getmtime(parquet_path)
            if parquet_mtime > csv_mtime:
                logger.info(f"Using existing Parquet file: {parquet_path}")
                return str(parquet_path)
        
        # Convert CSV to Parquet
        success = csv_to_parquet(
            csv_path=csv_path,
            parquet_path=str(parquet_path),
            compression=self.config['compression'],
            row_group_size=50000
        )
        
        if not success:
            raise RuntimeError(f"Failed to convert CSV to Parquet")
        
        # Optimize storage with date partitioning
        optimized_path = self.parquet_dir / f"{csv_name}_partitioned"
        if not optimized_path.exists():
            optimize_parquet_storage(
                parquet_path=str(parquet_path),
                output_path=str(optimized_path),
                partitions=['Date']
            )
        
        return str(parquet_path)
    
    def load_strategy_data(self, parquet_path: str) -> Tuple['cudf.DataFrame', List[str]]:
        """
        Load strategy data from Parquet to cuDF DataFrame
        
        Returns:
            Tuple of (cuDF DataFrame, list of strategy column names)
        """
        logger.info("Loading strategy data from Parquet")
        
        if CUDF_AVAILABLE and self.config['use_gpu']:
            # Load directly to cuDF
            df = load_parquet_to_cudf(
                parquet_path=parquet_path,
                chunk_size=self.config['chunk_size']
            )
            
            # Identify strategy columns
            strategy_cols = [col for col in df.columns 
                           if col not in ['Date', 'start_time', 'end_time', 
                                        'market_regime', 'Regime_Confidence_%',
                                        'Market_regime_transition_threshold', 
                                        'capital', 'zone']]
            
            logger.info(f"Loaded {len(df)} rows with {len(strategy_cols)} strategies")
            return df, strategy_cols
        else:
            # Fallback to pandas
            logger.warning("Using CPU mode - loading with pandas")
            df = pd.read_parquet(parquet_path)
            
            # Identify strategy columns
            strategy_cols = [col for col in df.columns 
                           if col not in ['Date', 'start_time', 'end_time', 
                                        'market_regime', 'Regime_Confidence_%',
                                        'Market_regime_transition_threshold', 
                                        'capital', 'zone']]
            
            logger.info(f"Loaded {len(df)} rows with {len(strategy_cols)} strategies")
            return df, strategy_cols
    
    def calculate_correlations(self, df, strategy_cols: List[str]):
        """Calculate correlation matrix using cuDF or pandas"""
        if CUDF_AVAILABLE and self.config['use_gpu'] and hasattr(df, '__module__') and 'cudf' in df.__module__:
            return calculate_correlations_cudf(df, strategy_cols)
        else:
            # Fallback to pandas correlation
            return df[strategy_cols].corr(method='pearson')
    
    def calculate_fitness(self, df, portfolio: List[str]) -> Dict[str, float]:
        """Calculate fitness metrics for a portfolio"""
        if CUDF_AVAILABLE and self.config['use_gpu'] and hasattr(df, '__module__') and 'cudf' in df.__module__:
            return calculate_fitness_cudf(df, portfolio, self.config['fitness_weights'])
        else:
            # Fallback to pandas calculations
            portfolio_returns = df[portfolio].sum(axis=1)
            
            metrics = {}
            
            # Total ROI
            metrics['total_roi'] = float(portfolio_returns.sum())
            
            # Maximum Drawdown
            cumulative_returns = portfolio_returns.cumsum()
            running_max = cumulative_returns.cummax()
            drawdown = cumulative_returns - running_max
            metrics['max_drawdown'] = float(drawdown.min())
            
            # Win Rate
            wins = (portfolio_returns > 0).sum()
            total_days = len(portfolio_returns)
            metrics['win_rate'] = float(wins) / total_days if total_days > 0 else 0
            
            # Profit Factor
            gains = portfolio_returns[portfolio_returns > 0].sum()
            losses = -portfolio_returns[portfolio_returns < 0].sum()
            metrics['profit_factor'] = float(gains) / float(losses) if losses > 0 else float('inf')
            
            # ROI/Drawdown Ratio
            roi_dd_ratio = metrics['total_roi'] / abs(metrics['max_drawdown']) if metrics['max_drawdown'] != 0 else float('inf')
            metrics['roi_dd_ratio'] = roi_dd_ratio
            
            # Calculate weighted fitness score
            fitness = 0
            for metric, value in metrics.items():
                weight = self.config['fitness_weights'].get(f'{metric}_weight', 0)
                if metric == 'max_drawdown':
                    fitness += weight * (1 / (1 + abs(value)))
                elif metric == 'roi_dd_ratio' and value == float('inf'):
                    fitness += weight * 1000
                else:
                    fitness += weight * value
            
            metrics['fitness_score'] = fitness
            
            return metrics
    
    def run_optimization(self, csv_path: str, output_dir: str) -> dict:
        """
        Run complete optimization workflow
        
        Args:
            csv_path: Path to input CSV file
            output_dir: Directory for output files
            
        Returns:
            Dictionary with optimization results
        """
        start_time = time.time()
        results = {
            'start_time': datetime.now().isoformat(),
            'input_file': csv_path,
            'output_dir': output_dir
        }
        
        try:
            # Step 1: Convert CSV to Parquet
            logger.info("Step 1: Converting CSV to Parquet")
            parquet_path = self.process_csv_to_parquet(csv_path)
            results['parquet_path'] = parquet_path
            
            # Step 2: Load data to cuDF
            logger.info("Step 2: Loading data to cuDF/pandas")
            df, strategy_cols = self.load_strategy_data(parquet_path)
            results['num_strategies'] = len(strategy_cols)
            results['num_days'] = len(df)
            
            # Step 3: Calculate correlations
            logger.info("Step 3: Calculating correlations")
            corr_start = time.time()
            correlation_matrix = self.calculate_correlations(df, strategy_cols)
            results['correlation_time'] = time.time() - corr_start
            
            # Step 4: Run optimization algorithms
            logger.info("Step 4: Running optimization algorithms")
            
            # Initialize algorithms with GPU support based on configuration
            use_gpu = CUDF_AVAILABLE and self.config['use_gpu']
            algorithms = {
                'genetic_algorithm': GeneticAlgorithm(use_gpu=use_gpu),
                'particle_swarm': ParticleSwarmOptimization(use_gpu=use_gpu),
                'simulated_annealing': SimulatedAnnealing(use_gpu=use_gpu),
                'differential_evolution': DifferentialEvolution(use_gpu=use_gpu),
                'ant_colony': AntColonyOptimization(use_gpu=use_gpu),
                'hill_climbing': HillClimbing(use_gpu=use_gpu),
                'bayesian_optimization': BayesianOptimization(use_gpu=use_gpu),
                'random_search': RandomSearch(use_gpu=use_gpu)
            }
            
            # Run each algorithm
            algorithm_results = {}
            for name, algorithm in algorithms.items():
                logger.info(f"Running {name}")
                algo_start = time.time()
                
                try:
                    # Set timeout
                    timeout = self.config['algorithm_timeouts'].get(name, 300)
                    
                    # Run optimization with retrofitted interface
                    # Algorithms now accept either numpy arrays or cuDF DataFrames
                    result = algorithm.optimize(
                        data=df if use_gpu else df[strategy_cols].values,
                        portfolio_size=self.config['portfolio_size']
                    )
                    
                    # Extract results from the new format
                    if result['status'] == 'success' or 'best_portfolio' in result:
                        # Get portfolio strategy names
                        if isinstance(result['best_portfolio'][0], str):
                            # Already strategy names
                            portfolio_names = result['best_portfolio']
                        else:
                            # Indices, convert to names
                            portfolio_names = [strategy_cols[i] for i in result['best_portfolio']]
                        
                        algorithm_results[name] = {
                            'portfolio': portfolio_names,
                            'metrics': result.get('detailed_metrics', {}),
                            'fitness_score': result['best_fitness'],
                            'execution_time': result['execution_time'],
                            'iterations': result.get('iterations', 'N/A'),
                            'data_type': result.get('data_type', 'unknown'),
                            'gpu_accelerated': result.get('gpu_accelerated', False),
                            'status': 'success'
                        }
                    else:
                        algorithm_results[name] = {
                            'status': 'failed',
                            'error': 'Algorithm did not return valid results',
                            'execution_time': time.time() - algo_start
                        }
                    
                except Exception as e:
                    logger.error(f"Algorithm {name} failed: {str(e)}")
                    algorithm_results[name] = {
                        'status': 'failed',
                        'error': str(e),
                        'execution_time': time.time() - algo_start
                    }
            
            results['algorithms'] = algorithm_results
            
            # Step 5: Generate output
            logger.info("Step 5: Generating output files")
            output_engine = OutputGenerationEngine(output_dir)
            
            # Prepare data for output generation
            successful_algorithms = [(name, res) for name, res in algorithm_results.items() if res['status'] == 'success']
            
            if not successful_algorithms:
                raise RuntimeError("No algorithms completed successfully")
            
            best_algorithm = max(
                successful_algorithms,
                key=lambda x: x[1].get('fitness_score', x[1].get('metrics', {}).get('fitness_score', 0))
            )
            
            best_portfolio_names = best_algorithm[1]['portfolio']
            
            # Convert data back to pandas for output generation
            if hasattr(df, 'to_pandas'):
                df_pandas = df.to_pandas()
            else:
                df_pandas = df
            
            output_results = output_engine.generate_all_outputs(
                df_pandas[best_portfolio_names],
                df_pandas,
                best_portfolio_names,
                algorithm_results,
                correlation_matrix
            )
            
            results['output_files'] = output_results
            results['total_time'] = time.time() - start_time
            results['status'] = 'success'
            
            # Log memory usage
            memory_stats = monitor_memory_usage()
            results['memory_usage'] = memory_stats
            logger.info(f"Memory usage - System: {memory_stats['system']['used_gb']:.2f}GB")
            if 'gpu' in memory_stats:
                logger.info(f"GPU memory: {memory_stats['gpu']['used_gb']:.2f}GB")
            
        except Exception as e:
            logger.error(f"Workflow failed: {str(e)}")
            results['status'] = 'failed'
            results['error'] = str(e)
            results['total_time'] = time.time() - start_time
        
        finally:
            # Cleanup
            if hasattr(self, 'memory_pool'):
                self.memory_pool.cleanup()
        
        return results

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Heavy Optimizer Platform - Parquet/cuDF Workflow')
    parser.add_argument('--input', '-i', required=True, help='Input CSV file path')
    parser.add_argument('--output', '-o', help='Output directory (default: auto-generated)')
    parser.add_argument('--portfolio-size', '-p', type=int, default=35, help='Portfolio size')
    parser.add_argument('--config', '-c', help='Configuration file path')
    parser.add_argument('--no-gpu', action='store_true', help='Disable GPU acceleration')
    parser.add_argument('--compare-legacy', action='store_true', help='Compare against legacy system results')
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)
    
    # Create output directory
    if args.output:
        output_dir = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"/mnt/optimizer_share/output/run_{timestamp}"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create workflow configuration
    config = {}
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    config['portfolio_size'] = args.portfolio_size
    if args.no_gpu:
        config['use_gpu'] = False
    
    # Save configuration
    config_path = os.path.join(output_dir, 'workflow_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Run workflow
    workflow = ParquetCuDFWorkflow(config_path)
    results = workflow.run_optimization(args.input, output_dir)
    
    # Save results
    results_path = os.path.join(output_dir, 'workflow_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Print summary
    if results['status'] == 'success':
        logger.info(f"✅ Optimization completed successfully in {results['total_time']:.2f} seconds")
        logger.info(f"📁 Results saved to: {output_dir}")
        
        # Run legacy comparison if requested
        if args.compare_legacy:
            logger.info("🔍 Running legacy system comparison...")
            try:
                from legacy_integration_orchestrator import LegacyIntegrationOrchestrator
                
                orchestrator = LegacyIntegrationOrchestrator()
                comparison_results = orchestrator.run_complete_comparison(
                    input_csv=args.input,
                    portfolio_sizes=[args.portfolio_size],
                    timeout_minutes=30
                )
                
                summary = comparison_results.get('summary', {})
                logger.info(f"📊 Comparison Status: {summary.get('overall_status', 'Unknown')}")
                logger.info(f"📊 Fitness Match Rate: {summary.get('fitness_match_rate', 0):.1f}%")
                
            except Exception as e:
                logger.error(f"❌ Legacy comparison failed: {e}")
    else:
        logger.error(f"❌ Optimization failed: {results.get('error', 'Unknown error')}")
        sys.exit(1)

if __name__ == "__main__":
    main()