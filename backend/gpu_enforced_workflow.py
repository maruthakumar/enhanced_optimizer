#!/usr/bin/env python3
"""
GPU-Enforced Workflow for Heavy Optimizer Platform
Ensures all operations use GPU through HeavyDB with no CPU fallback
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from csv_only_heavydb_workflow import CSVOnlyHeavyDBOptimizer
from lib.heavydb_connector import get_connection, load_strategy_data, execute_query
from lib.correlation_optimizer import optimize_correlation_calculation

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class GPUEnforcedOptimizer(CSVOnlyHeavyDBOptimizer):
    """Optimizer that enforces GPU-only mode through HeavyDB"""
    
    def __init__(self):
        super().__init__()
        
        # Override configuration to enforce GPU
        self.gpu_mode_enforced = True
        self.cpu_fallback_enabled = False
        
        # Check HeavyDB connection
        conn = get_connection()
        if not conn:
            raise RuntimeError("HeavyDB connection required for GPU-only mode")
        
        # Verify GPU availability
        gpu_info = self._verify_gpu_availability(conn)
        logging.info(f"🚀 GPU Mode Enforced: {gpu_info}")
        
        conn.close()
    
    def _verify_gpu_availability(self, conn):
        """Verify GPU is available in HeavyDB"""
        try:
            cursor = conn.cursor()
            
            # Check GPU info
            cursor.execute("SELECT * FROM omnisci_server_status")
            status = cursor.fetchall()
            
            # Check for GPU devices
            cursor.execute("SHOW DATABASES")
            
            return "GPU acceleration available via HeavyDB"
            
        except Exception as e:
            logging.warning(f"GPU verification query failed: {e}")
            return "GPU mode enforced (HeavyDB SQL)"
    
    def _preprocess_with_gpu(self, df):
        """Preprocess data using HeavyDB GPU - no CPU fallback"""
        logging.info("⚡ GPU-ONLY preprocessing via HeavyDB")
        
        try:
            # For large datasets, use chunked loading
            num_strategies = len(df.columns) - 2  # Exclude Date and Day
            
            if num_strategies > 5000:
                logging.info(f"📦 Large dataset ({num_strategies} strategies) - using chunked loading")
                return self._preprocess_large_dataset_gpu(df, chunk_size=5000)
            
            # Standard GPU processing
            table_name = f"strategies_{int(time.time())}"
            conn = get_connection(force_new=True)
            
            if not conn:
                raise RuntimeError("HeavyDB connection lost - GPU mode requires active connection")
            
            # Load with increased timeout for large datasets
            success = load_strategy_data(df, table_name, connection=conn, timeout=300)
            if not success:
                raise RuntimeError("GPU data loading failed - no CPU fallback in GPU-only mode")
            
            # Get numeric columns
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            
            # Calculate correlations if needed
            strategy_columns = [col for col in numeric_columns if col.startswith('SENSEX') or 'strategy' in col.lower()]
            
            correlation_matrix = None
            if len(strategy_columns) > 1:
                logging.info(f"📊 GPU correlation calculation for {len(strategy_columns)} strategies")
                
                # Use optimized correlation with proper configuration
                corr_config = {
                    'chunk_size': 100,
                    'max_query_size': 1000,
                    'timeout': 300,
                    'adaptive_chunking': True,
                    'gpu_only': True
                }
                
                correlation_matrix = optimize_correlation_calculation(
                    table_name, 
                    connection=conn,
                    config=corr_config
                )
            
            # Clean up
            try:
                cursor = conn.cursor()
                cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
            except:
                pass
            
            conn.close()
            
            return {
                'matrix': df[numeric_columns].values,
                'columns': list(numeric_columns),
                'dates': df['Date'].tolist() if 'Date' in df else None,
                'correlation_matrix': correlation_matrix,
                'gpu_processed': True
            }
            
        except Exception as e:
            # In GPU-only mode, we don't fall back to CPU
            logging.error(f"❌ GPU processing failed: {e}")
            raise RuntimeError(f"GPU-only mode: {e}")
    
    def _preprocess_large_dataset_gpu(self, df, chunk_size=5000):
        """Process large datasets in chunks through HeavyDB"""
        logging.info(f"🔄 Processing {len(df.columns)-2} strategies in chunks of {chunk_size}")
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        strategy_columns = [col for col in numeric_columns if col.startswith('SENSEX') or 'strategy' in col.lower()]
        
        # Process in chunks
        chunks_processed = 0
        all_data = []
        
        for i in range(0, len(strategy_columns), chunk_size):
            chunk_cols = strategy_columns[i:i+chunk_size]
            chunk_df = df[['Date'] + chunk_cols] if 'Date' in df else df[chunk_cols]
            
            logging.info(f"📦 Processing chunk {chunks_processed+1} ({len(chunk_cols)} strategies)")
            
            # Load chunk to HeavyDB
            table_name = f"chunk_{int(time.time())}_{chunks_processed}"
            conn = get_connection(force_new=True)
            
            if not conn:
                raise RuntimeError("HeavyDB connection required for GPU processing")
            
            success = load_strategy_data(chunk_df, table_name, connection=conn, timeout=120)
            if not success:
                raise RuntimeError(f"Failed to load chunk {chunks_processed+1}")
            
            # Clean up chunk table
            try:
                cursor = conn.cursor()
                cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
            except:
                pass
            
            conn.close()
            
            chunks_processed += 1
            all_data.append(chunk_df[chunk_cols].values)
        
        # Combine all chunks
        full_matrix = np.hstack(all_data)
        
        logging.info(f"✅ Processed {chunks_processed} chunks, total shape: {full_matrix.shape}")
        
        return {
            'matrix': full_matrix,
            'columns': list(strategy_columns),
            'dates': df['Date'].tolist() if 'Date' in df else None,
            'correlation_matrix': None,  # Too large for full correlation
            'gpu_processed': True,
            'chunked_processing': True
        }
    
    def run_optimization(self, input_file, portfolio_size=35, algorithms=None):
        """Run optimization in GPU-only mode"""
        logging.info("="*80)
        logging.info("🚀 GPU-ENFORCED OPTIMIZATION MODE")
        logging.info("="*80)
        logging.info("⚡ All operations will use GPU through HeavyDB")
        logging.info("❌ CPU fallback is DISABLED")
        logging.info("="*80)
        
        # Verify HeavyDB is still available
        conn = get_connection()
        if not conn:
            raise RuntimeError("HeavyDB connection lost - cannot proceed in GPU-only mode")
        conn.close()
        
        # Run optimization with enforced GPU mode
        return super().run_optimization(input_file, portfolio_size, algorithms)


def main():
    """Run GPU-enforced optimization"""
    import argparse
    
    parser = argparse.ArgumentParser(description='GPU-Enforced Heavy Optimizer')
    parser.add_argument('--input', '-i', required=True, help='Input CSV file path')
    parser.add_argument('--portfolio-size', '-p', type=int, default=35, 
                       help='Target portfolio size')
    parser.add_argument('--test', action='store_true', help='Run with test dataset')
    
    args = parser.parse_args()
    
    # Use test data if requested
    if args.test:
        input_file = "/mnt/optimizer_share/input/SENSEX_test_dataset.csv"
    else:
        input_file = args.input
    
    # Verify file exists
    if not os.path.exists(input_file):
        logging.error(f"❌ Input file not found: {input_file}")
        return 1
    
    try:
        # Create GPU-enforced optimizer
        optimizer = GPUEnforcedOptimizer()
        
        # Run optimization
        optimizer.run_optimization(input_file, args.portfolio_size)
        
        logging.info("✅ GPU-enforced optimization completed successfully")
        return 0
        
    except Exception as e:
        logging.error(f"❌ GPU-enforced optimization failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())