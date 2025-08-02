#!/usr/bin/env python3
"""
GPU-Enhanced Workflow - Ensures GPU mode is primary with optional CPU
"""

import os
import sys
import time
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from csv_only_heavydb_workflow import CSVOnlyHeavyDBOptimizer
from lib.heavydb_connector import get_connection, HEAVYDB_AVAILABLE
from lib.config_reader import get_gpu_config, is_gpu_required
from lib.algorithm_monitor import get_algorithm_monitor

class GPUEnhancedOptimizer(CSVOnlyHeavyDBOptimizer):
    """Enhanced optimizer that prioritizes GPU mode"""
    
    def __init__(self):
        """Initialize with GPU priority"""
        super().__init__()
        
        # Override to ensure GPU priority
        self.gpu_config = get_gpu_config()
        self.gpu_required = is_gpu_required()
        
        # Check GPU availability immediately
        self._check_gpu_availability()
    
    def _check_gpu_availability(self):
        """Check and ensure GPU is available if required"""
        print("\n🎮 GPU Mode Check")
        print("=" * 50)
        
        # Check configuration
        print(f"📋 Configuration:")
        print(f"  - GPU Required: {self.gpu_required}")
        print(f"  - GPU Acceleration: {self.gpu_config['acceleration']}")
        print(f"  - CPU Fallback: {self.gpu_config['cpu_fallback_allowed']}")
        
        # Test HeavyDB connection
        if self.gpu_required or self.gpu_config['enabled']:
            print("\n🔗 Testing HeavyDB connection...")
            
            # Try multiple times if required
            max_retries = self.gpu_config.get('max_connection_retries', 3)
            
            for attempt in range(max_retries):
                conn = get_connection(force_new=True)
                
                if conn:
                    print("✅ HeavyDB connection successful!")
                    
                    # Verify GPU mode
                    from lib.heavydb_connector import get_execution_mode
                    mode = get_execution_mode()
                    print(f"🎯 Execution mode: {mode.upper()}")
                    
                    if mode == 'gpu':
                        print("✅ GPU mode confirmed!")
                        self.heavydb_enabled = True
                        try:
                            conn.close()
                        except:
                            pass
                        return True
                    else:
                        print("⚠️ HeavyDB connected but GPU mode not active")
                    
                    try:
                        conn.close()
                    except:
                        pass
                else:
                    print(f"❌ Connection attempt {attempt + 1}/{max_retries} failed")
                    
                if attempt < max_retries - 1:
                    print("⏳ Retrying in 2 seconds...")
                    time.sleep(2)
            
            # If we get here, connection failed
            if self.gpu_required and not self.gpu_config['cpu_fallback_allowed']:
                raise Exception(
                    "GPU mode is required but HeavyDB connection failed!\n"
                    "Please ensure:\n"
                    "  1. HeavyDB server is running\n"
                    "  2. Connection parameters are correct\n"
                    "  3. GPU drivers are installed"
                )
            else:
                print("⚠️ GPU not available, using CPU mode")
                self.heavydb_enabled = False
        else:
            print("ℹ️ GPU not required, proceeding with available mode")
    
    def preprocess_data(self, loaded_data):
        """Enhanced preprocessing with GPU priority"""
        print("\n🔄 Preprocessing data...")
        
        # Extract DataFrame
        if isinstance(loaded_data, tuple):
            df = loaded_data[0]['data']
        elif isinstance(loaded_data, dict):
            df = loaded_data['data']
        else:
            df = loaded_data
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        n_strategies = len(numeric_columns)
        
        print(f"📊 Dataset: {n_strategies:,} strategies")
        
        # Determine processing mode
        use_gpu = False
        
        if self.heavydb_enabled and HEAVYDB_AVAILABLE:
            # Check if we should use GPU based on data size
            min_strategies = self.gpu_config.get('min_strategies_for_gpu', 10)
            
            if n_strategies >= min_strategies or self.gpu_config.get('force_gpu_mode', False):
                use_gpu = True
                print("⚡ Using GPU acceleration")
            else:
                print(f"ℹ️ Dataset too small for GPU ({n_strategies} < {min_strategies})")
        
        # Process based on mode
        if use_gpu:
            try:
                return self._preprocess_with_gpu(df)
            except Exception as e:
                print(f"⚠️ GPU preprocessing failed: {e}")
                
                if self.gpu_config['cpu_fallback_allowed']:
                    print("🔄 Falling back to CPU mode")
                    return self._preprocess_with_cpu(df)
                else:
                    raise Exception("GPU processing failed and CPU fallback is disabled")
        else:
            if self.gpu_required:
                raise Exception(
                    f"GPU processing required but not available!\n"
                    f"Dataset has {n_strategies} strategies (min required: {self.gpu_config.get('min_strategies_for_gpu', 10)})"
                )
            
            print("🖥️ Using CPU mode")
            return self._preprocess_with_cpu(df)
    
    def execute_algorithms_with_heavydb(self, processed_data, portfolio_size):
        """Enhanced algorithm execution with iteration monitoring"""
        print("\n🧬 Executing optimization algorithms...")
        
        # Get algorithm monitor
        monitor = get_algorithm_monitor()
        
        # Run parent implementation
        result, exec_time = super().execute_algorithms_with_heavydb(processed_data, portfolio_size)
        
        # Log summary
        print("\n📊 Algorithm Iteration Summary:")
        summary = monitor.get_summary()
        
        for algo, count in summary['iteration_counts'].items():
            evaluations = summary['fitness_evaluations'].get(algo, 0)
            print(f"  - {algo}: {count} iterations, {evaluations} evaluations")
        
        return result, exec_time

def main():
    """Run GPU-enhanced workflow"""
    import argparse
    
    parser = argparse.ArgumentParser(description='GPU-Enhanced Portfolio Optimizer')
    parser.add_argument('--input', required=True, help='Input CSV file')
    parser.add_argument('--portfolio-size', type=int, required=True, help='Portfolio size')
    parser.add_argument('--force-gpu', action='store_true', help='Force GPU mode')
    parser.add_argument('--allow-cpu', action='store_true', help='Allow CPU fallback')
    
    args = parser.parse_args()
    
    # Override configuration if specified
    if args.force_gpu:
        os.environ['FORCE_GPU_MODE'] = 'true'
    
    if args.allow_cpu:
        os.environ['GPU_FALLBACK_ALLOWED'] = 'true'
    
    print("🚀 GPU-Enhanced Portfolio Optimizer")
    print("=" * 60)
    
    # Create optimizer
    optimizer = GPUEnhancedOptimizer()
    
    # Run optimization
    success = optimizer.run_optimization(args.input, args.portfolio_size)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())