#!/usr/bin/env python3
"""
Configuration-Driven Workflow for Heavy Optimizer Platform

Demonstrates the complete configuration management implementation with:
- All algorithms reading from configuration
- Zone optimization with configuration
- ULTA with configuration
- Job-specific overrides
"""

import os
import sys
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import json

# Add backend to path
sys.path.append(str(Path(__file__).parent))

from config.config_manager import get_config_manager
from algorithms import AlgorithmFactory
from algorithms.workflow_adapter import WorkflowAlgorithmAdapter
from ulta_calculator import ULTACalculator
from zone_optimizer import ZoneOptimizer
from dal.csv_dal import CSVDAL


class ConfigDrivenWorkflow:
    """Workflow that fully utilizes configuration management"""
    
    def __init__(self, base_config_path: str, job_config_path: Optional[str] = None):
        """
        Initialize workflow with configuration
        
        Args:
            base_config_path: Path to base configuration file
            job_config_path: Optional job-specific configuration override
        """
        self.logger = logging.getLogger(__name__)
        
        # Initialize configuration manager
        self.config_manager = get_config_manager(base_config_path)
        
        # Apply job-specific overrides if provided
        if job_config_path and os.path.exists(job_config_path):
            self.config_manager.load_job_config(job_config_path)
            self.logger.info(f"Applied job-specific configuration from {job_config_path}")
        
        # Validate configuration
        validation_result = self.config_manager.validate_config()
        if not validation_result.is_valid:
            self.logger.error("Configuration validation failed!")
            self.logger.error(f"Errors: {validation_result.errors}")
            raise ValueError("Invalid configuration")
        
        if validation_result.warnings:
            self.logger.warning(f"Configuration warnings: {validation_result.warnings}")
        
        # Initialize components with configuration
        self.algorithm_factory = AlgorithmFactory(base_config_path)
        self.workflow_adapter = WorkflowAlgorithmAdapter(base_config_path)
        self.ulta_calculator = ULTACalculator(config_path=base_config_path)
        self.zone_optimizer = ZoneOptimizer(config_path=base_config_path)
        
        # Initialize DAL
        self.dal = CSVDAL()
        self.dal.connect()
        
        self.logger.info("Configuration-driven workflow initialized successfully")
    
    def run(self, input_file: str):
        """
        Run the complete workflow
        
        Args:
            input_file: Path to input CSV file
        """
        print("="*80)
        print("🚀 Configuration-Driven Heavy Optimizer Workflow")
        print("="*80)
        print(f"Configuration: {self.config_manager.base_config_path}")
        print(f"Input File: {input_file}")
        print("="*80)
        
        # Display active configuration
        self._display_configuration()
        
        try:
            # Step 1: Load data
            print("\n📥 Step 1: Loading Data...")
            self._load_data(input_file)
            
            # Step 2: Apply ULTA if enabled
            print("\n🔄 Step 2: Applying ULTA Transformation...")
            self._apply_ulta()
            
            # Step 3: Run optimization
            print("\n🧬 Step 3: Running Optimization Algorithms...")
            if self.config_manager.getboolean('ZONE', 'enabled'):
                results = self._run_zone_optimization()
            else:
                results = self._run_standard_optimization()
            
            # Step 4: Display results
            print("\n📊 Step 4: Results Summary")
            self._display_results(results)
            
            print("\n✅ Workflow completed successfully!")
            
        except Exception as e:
            self.logger.error(f"Workflow failed: {e}")
            print(f"\n❌ Workflow failed: {e}")
            raise
        
        finally:
            self.dal.disconnect()
    
    def _display_configuration(self):
        """Display active configuration settings"""
        print("\n📋 Active Configuration:")
        print("-" * 40)
        
        # Algorithm settings
        enabled_algos = self.config_manager.getlist('ALGORITHMS', 'enabled_algorithms')
        print(f"Enabled Algorithms: {', '.join(enabled_algos)}")
        
        # Portfolio settings
        print(f"Portfolio Size Range: {self.config_manager.getint('PORTFOLIO_OPTIMIZATION', 'min_portfolio_size')}"
              f" - {self.config_manager.getint('PORTFOLIO_OPTIMIZATION', 'max_portfolio_size')}")
        
        # Zone settings
        zone_enabled = self.config_manager.getboolean('ZONE', 'enabled')
        print(f"Zone Optimization: {'Enabled' if zone_enabled else 'Disabled'}")
        if zone_enabled:
            print(f"  - Zone Count: {self.config_manager.getint('ZONE', 'zone_count')}")
            print(f"  - Zone Weights: {self.config_manager.get('ZONE', 'zone_weights')}")
        
        # ULTA settings
        ulta_enabled = self.config_manager.getboolean('ULTA', 'enabled')
        print(f"ULTA: {'Enabled' if ulta_enabled else 'Disabled'}")
        if ulta_enabled:
            print(f"  - ROI Threshold: {self.config_manager.getfloat('ULTA', 'roi_threshold')}")
        
        print("-" * 40)
    
    def _load_data(self, input_file: str):
        """Load data from CSV file"""
        success = self.dal.load_csv_to_heavydb(input_file, 'trading_data')
        if not success:
            raise Exception("Failed to load data")
        
        table_info = self.dal.get_table_info('trading_data')
        print(f"  ✅ Loaded {table_info['rows']} rows, {table_info['columns']} columns")
    
    def _apply_ulta(self):
        """Apply ULTA transformation if enabled"""
        if not self.config_manager.getboolean('ULTA', 'enabled'):
            print("  ⏭️  ULTA disabled in configuration")
            return
        
        success = self.dal.apply_ulta_transformation('trading_data')
        if not success:
            raise Exception("ULTA transformation failed")
        
        # Get ULTA statistics
        inverted_count = len(self.ulta_calculator.inverted_strategies)
        print(f"  ✅ ULTA applied: {inverted_count} strategies inverted")
    
    def _run_standard_optimization(self) -> Dict:
        """Run standard optimization without zones"""
        # Get data
        df = self.dal.tables.get('trading_data_ulta', self.dal.tables.get('trading_data'))
        daily_matrix = df.values
        
        # Get portfolio size
        portfolio_size = self.config_manager.getint('PORTFOLIO_OPTIMIZATION', 'default_portfolio_size')
        
        # Run all enabled algorithms
        enabled_algos = self.config_manager.getlist('ALGORITHMS', 'enabled_algorithms')
        results = {}
        
        for algo_name in enabled_algos:
            if algo_name in ['ga', 'pso', 'sa', 'de', 'aco', 'hc', 'bo', 'rs']:
                print(f"\n  🔄 Running {algo_name.upper()}...")
                result = self.workflow_adapter._run_algorithm(algo_name, daily_matrix, portfolio_size)
                results[algo_name] = result
                print(f"     Fitness: {result['best_fitness']:.6f}, Time: {result['execution_time']:.3f}s")
        
        return results
    
    def _run_zone_optimization(self) -> Dict:
        """Run zone-based optimization"""
        # Get data
        df = self.dal.tables.get('trading_data_ulta', self.dal.tables.get('trading_data'))
        
        # Run zone optimization
        zone_results = self.zone_optimizer.optimize_zones(
            data=df,
            dal=self.dal,
            algorithms=['ga', 'pso', 'sa']  # Use subset for demo
        )
        
        # Convert to standard results format
        results = {}
        for zone_name, zone_result in zone_results.zone_results.items():
            results[f"zone_{zone_name}"] = {
                'best_fitness': zone_result.best_fitness,
                'best_portfolio': zone_result.best_portfolio,
                'execution_time': zone_result.execution_time
            }
        
        # Add combined result
        results['combined'] = {
            'best_fitness': zone_results.combined_fitness,
            'best_portfolio': zone_results.combined_portfolio,
            'execution_time': zone_results.total_execution_time
        }
        
        return results
    
    def _display_results(self, results: Dict):
        """Display optimization results"""
        print("-" * 60)
        print(f"{'Algorithm':<20} {'Fitness':>15} {'Portfolio Size':>15} {'Time (s)':>10}")
        print("-" * 60)
        
        best_algo = None
        best_fitness = -float('inf')
        
        for algo_name, result in results.items():
            fitness = result.get('best_fitness', -float('inf'))
            portfolio = result.get('best_portfolio', [])
            exec_time = result.get('execution_time', 0)
            
            print(f"{algo_name:<20} {fitness:>15.6f} {len(portfolio):>15} {exec_time:>10.3f}")
            
            if fitness > best_fitness:
                best_fitness = fitness
                best_algo = algo_name
        
        print("-" * 60)
        print(f"\n🏆 Best Algorithm: {best_algo} (Fitness: {best_fitness:.6f})")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Configuration-Driven Heavy Optimizer Workflow'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='/mnt/optimizer_share/config/production_config.ini',
        help='Base configuration file path'
    )
    parser.add_argument(
        '--job-config',
        type=str,
        help='Job-specific configuration override file'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input CSV file path'
    )
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate configuration without running workflow'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Validate configuration if requested
    if args.validate_only:
        config_manager = get_config_manager(args.config)
        if args.job_config:
            config_manager.load_job_config(args.job_config)
        
        print(config_manager.get_validation_report())
        return
    
    # Run workflow
    workflow = ConfigDrivenWorkflow(args.config, args.job_config)
    workflow.run(args.input)


if __name__ == '__main__':
    main()