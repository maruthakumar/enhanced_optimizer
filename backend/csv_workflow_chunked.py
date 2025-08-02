#!/usr/bin/env python3
"""
Enhanced CSV Workflow with Chunked Processing
Handles large datasets (25,544 strategies) efficiently
"""

import os
import sys
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from csv_only_heavydb_workflow import CSVOnlyHeavyDBOptimizer
from chunked_processor import ChunkedProcessor
from lib.heavydb_connector import get_connection, HEAVYDB_AVAILABLE
from algorithms import (
    GeneticAlgorithm,
    ParticleSwarmOptimization,
    SimulatedAnnealing,
    DifferentialEvolution,
    AntColonyOptimization,
    HillClimbing,
    BayesianOptimization,
    RandomSearch
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class ChunkedCSVOptimizer(CSVOnlyHeavyDBOptimizer):
    """Enhanced optimizer with chunked processing for large datasets"""
    
    def __init__(self):
        super().__init__()
        self.chunked_processor = ChunkedProcessor()
        self.large_dataset_threshold = 5000  # Use chunking above this
        
        # Algorithm instances
        self.algorithms = {
            'GA': GeneticAlgorithm(),
            'PSO': ParticleSwarmOptimization(),
            'SA': SimulatedAnnealing(),
            'DE': DifferentialEvolution(),
            'ACO': AntColonyOptimization(),
            'HC': HillClimbing(),
            'BO': BayesianOptimization(),
            'RS': RandomSearch()
        }
    
    def preprocess_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Preprocess data with intelligent chunking for large datasets
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with processed data
        """
        # Identify strategy columns
        date_cols = ['Date', 'Day']
        strategy_cols = [col for col in df.columns if col not in date_cols]
        n_strategies = len(strategy_cols)
        
        self.logger.info(f"📊 Preprocessing {n_strategies} strategies")
        
        # Decide whether to use chunking
        if n_strategies > self.large_dataset_threshold and HEAVYDB_AVAILABLE:
            self.logger.info(f"📦 Large dataset detected - using chunked processing")
            return self._preprocess_large_dataset_chunked(df)
        else:
            # Use standard processing for smaller datasets
            return super().preprocess_data(df)
    
    def _preprocess_large_dataset_chunked(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Process large dataset using chunked approach"""
        try:
            # Use chunked processor
            result = self.chunked_processor.process_large_dataset(
                df,
                base_table_name=f"strategies_{int(time.time())}",
                portfolio_size=35
            )
            
            if result['success']:
                # Prepare data for optimization algorithms
                numeric_columns = df.select_dtypes(include=[np.number]).columns
                
                return {
                    'matrix': df[numeric_columns].values,
                    'columns': list(numeric_columns),
                    'dates': df['Date'].tolist() if 'Date' in df else None,
                    'correlation_matrix': None,  # Too large for full correlation
                    'gpu_processed': True,
                    'chunked': True,
                    'chunk_info': result
                }
            else:
                raise Exception(f"Chunked processing failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            self.logger.error(f"❌ Chunked preprocessing failed: {e}")
            
            # Fall back to memory-efficient CPU processing
            self.logger.info("📊 Using memory-efficient CPU processing")
            return self._preprocess_cpu_efficient(df)
    
    def _preprocess_cpu_efficient(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Memory-efficient CPU preprocessing for large datasets"""
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        # Process in chunks to avoid memory issues
        n_strategies = len(numeric_columns)
        chunk_size = 1000
        
        # Calculate statistics in chunks
        means = []
        stds = []
        
        for i in range(0, n_strategies, chunk_size):
            chunk_cols = numeric_columns[i:i+chunk_size]
            chunk_data = df[chunk_cols]
            
            means.extend(chunk_data.mean().values)
            stds.extend(chunk_data.std().values)
            
            self.logger.info(f"   Processed {min(i+chunk_size, n_strategies)}/{n_strategies} strategies")
        
        return {
            'matrix': df[numeric_columns].values,
            'columns': list(numeric_columns),
            'dates': df['Date'].tolist() if 'Date' in df else None,
            'correlation_matrix': None,  # Skip for large datasets
            'means': np.array(means),
            'stds': np.array(stds),
            'gpu_processed': False,
            'chunked': True
        }
    
    def run_chunked_optimization(self, processed_data: Dict[str, Any], 
                               portfolio_size: int = 35) -> Dict[str, Any]:
        """
        Run optimization on chunked data
        
        Args:
            processed_data: Preprocessed data
            portfolio_size: Target portfolio size
            
        Returns:
            Optimization results
        """
        matrix = processed_data['matrix']
        n_strategies = matrix.shape[1]
        
        self.logger.info(f"\n🧬 Running optimization on {n_strategies} strategies")
        
        # If dataset is very large, optimize in stages
        if n_strategies > 10000:
            return self._run_staged_optimization(matrix, portfolio_size)
        else:
            return self._run_standard_optimization(matrix, portfolio_size)
    
    def _run_staged_optimization(self, matrix: np.ndarray, 
                                portfolio_size: int) -> Dict[str, Any]:
        """Run optimization in stages for very large datasets"""
        n_strategies = matrix.shape[1]
        stage_size = 5000
        
        self.logger.info(f"🎯 Running staged optimization ({n_strategies} strategies)")
        
        # Stage 1: Find best strategies in each chunk
        stage1_candidates = []
        
        for i in range(0, n_strategies, stage_size):
            chunk_end = min(i + stage_size, n_strategies)
            chunk_matrix = matrix[:, i:chunk_end]
            
            self.logger.info(f"\n📦 Stage 1 - Chunk {i//stage_size + 1}: strategies {i}-{chunk_end}")
            
            # Run fast algorithm (RS or HC) to find top candidates
            chunk_result = self._run_single_algorithm(
                'RS',  # Random Search is fast
                chunk_matrix,
                min(portfolio_size * 2, chunk_matrix.shape[1])
            )
            
            if chunk_result['best_portfolio']:
                # Map back to original indices
                chunk_candidates = [i + idx for idx in chunk_result['best_portfolio']]
                stage1_candidates.extend(chunk_candidates)
        
        self.logger.info(f"\n✅ Stage 1 complete: {len(stage1_candidates)} candidates")
        
        # Stage 2: Optimize among candidates
        if len(stage1_candidates) > portfolio_size:
            candidate_matrix = matrix[:, stage1_candidates]
            
            self.logger.info(f"\n🎯 Stage 2: Optimizing {len(stage1_candidates)} candidates")
            
            # Run better algorithms on reduced set
            best_result = None
            best_fitness = -float('inf')
            
            for algo_name in ['SA', 'GA', 'PSO']:
                self.logger.info(f"\n🧬 Running {algo_name} on candidates...")
                
                result = self._run_single_algorithm(
                    algo_name,
                    candidate_matrix,
                    portfolio_size
                )
                
                if result['best_fitness'] > best_fitness:
                    best_fitness = result['best_fitness']
                    best_result = result
                    # Map back to original indices
                    best_result['best_portfolio'] = [
                        stage1_candidates[idx] for idx in result['best_portfolio']
                    ]
            
            return best_result
        else:
            # Use all candidates
            return {
                'best_portfolio': stage1_candidates[:portfolio_size],
                'best_fitness': 0.0,
                'algorithm': 'staged_optimization',
                'n_strategies': n_strategies
            }
    
    def _run_standard_optimization(self, matrix: np.ndarray, 
                                 portfolio_size: int) -> Dict[str, Any]:
        """Run standard optimization for moderate datasets"""
        # Use configured algorithms
        results = {}
        best_overall = None
        best_fitness = -float('inf')
        
        # Run each algorithm
        for algo_name in ['SA', 'GA', 'PSO', 'DE', 'ACO', 'HC', 'BO', 'RS']:
            if algo_name in self.algorithms:
                self.logger.info(f"\n🧬 Running {algo_name}...")
                
                result = self._run_single_algorithm(algo_name, matrix, portfolio_size)
                results[algo_name] = result
                
                if result['best_fitness'] > best_fitness:
                    best_fitness = result['best_fitness']
                    best_overall = result
        
        return best_overall or {'best_portfolio': [], 'best_fitness': 0.0}
    
    def _run_single_algorithm(self, algo_name: str, matrix: np.ndarray, 
                            portfolio_size: int) -> Dict[str, Any]:
        """Run a single optimization algorithm"""
        try:
            algorithm = self.algorithms.get(algo_name)
            if not algorithm:
                return {'best_portfolio': [], 'best_fitness': 0.0}
            
            # Define fitness function
            def fitness_function(daily_matrix, portfolio):
                selected = daily_matrix[:, portfolio]
                avg_returns = np.mean(selected, axis=1)
                
                total_return = np.sum(avg_returns)
                cumulative = np.cumsum(avg_returns)
                peak = np.maximum.accumulate(cumulative)
                drawdown = peak - cumulative
                max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0
                
                if max_drawdown > 0:
                    return total_return / max_drawdown
                else:
                    return total_return * 100 if total_return > 0 else total_return
            
            # Run optimization
            result = algorithm.optimize(matrix, portfolio_size, fitness_function)
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Algorithm {algo_name} failed: {e}")
            return {'best_portfolio': [], 'best_fitness': 0.0}
    
    def run_optimization(self, input_file: str, portfolio_size: int = 35,
                        algorithms: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run full optimization with chunked processing support
        
        Args:
            input_file: Input CSV file path
            portfolio_size: Target portfolio size
            algorithms: List of algorithms to use
            
        Returns:
            Optimization results
        """
        try:
            # Load data
            self.logger.info(f"📥 Loading data from: {input_file}")
            df = pd.read_csv(input_file)
            
            # Check dataset size
            n_strategies = len([col for col in df.columns if col not in ['Date', 'Day']])
            
            self.logger.info("="*80)
            self.logger.info("🚀 CHUNKED CSV OPTIMIZATION")
            self.logger.info(f"📊 Dataset: {n_strategies} strategies x {len(df)} days")
            self.logger.info(f"🎯 Portfolio size: {portfolio_size}")
            self.logger.info("="*80)
            
            # Preprocess (will use chunking if needed)
            processed_data = self.preprocess_data(df)
            
            # Run optimization
            if processed_data.get('chunked', False):
                result = self.run_chunked_optimization(processed_data, portfolio_size)
            else:
                result = self._run_standard_optimization(
                    processed_data['matrix'], 
                    portfolio_size
                )
            
            # Generate output
            self._generate_output(df, result, processed_data)
            
            self.logger.info("\n✅ Optimization completed successfully")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Optimization failed: {e}")
            return {'error': str(e), 'success': False}
    
    def _generate_output(self, df: pd.DataFrame, result: Dict[str, Any], 
                        processed_data: Dict[str, Any]):
        """Generate output files"""
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"/mnt/optimizer_share/output/chunked_run_{timestamp}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save summary
        summary_file = output_dir / "optimization_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("CHUNKED OPTIMIZATION SUMMARY\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Dataset Size: {processed_data['matrix'].shape}\n")
            f.write(f"Chunked Processing: {processed_data.get('chunked', False)}\n")
            f.write(f"GPU Processed: {processed_data.get('gpu_processed', False)}\n")
            
            if 'chunk_info' in processed_data:
                info = processed_data['chunk_info']
                f.write(f"\nChunk Processing Info:\n")
                f.write(f"  - Total Strategies: {info.get('total_strategies', 'N/A')}\n")
                f.write(f"  - Chunks Processed: {info.get('chunks_processed', 'N/A')}\n")
                f.write(f"  - Processing Time: {info.get('total_processing_time', 0):.1f}s\n")
            
            f.write(f"\nBest Portfolio Size: {len(result.get('best_portfolio', []))}\n")
            f.write(f"Best Fitness: {result.get('best_fitness', 0):.6f}\n")
            f.write(f"Algorithm Used: {result.get('algorithm', 'Unknown')}\n")
        
        self.logger.info(f"📁 Results saved to: {output_dir}")


def main():
    """Test chunked CSV workflow"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Chunked CSV Optimizer')
    parser.add_argument('--input', '-i', required=True, help='Input CSV file')
    parser.add_argument('--portfolio-size', '-p', type=int, default=35, help='Portfolio size')
    parser.add_argument('--test', action='store_true', help='Run with test data')
    
    args = parser.parse_args()
    
    # Create optimizer
    optimizer = ChunkedCSVOptimizer()
    
    # Run optimization
    result = optimizer.run_optimization(args.input, args.portfolio_size)
    
    return 0 if result.get('success', True) else 1


if __name__ == "__main__":
    sys.exit(main())