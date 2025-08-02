#!/usr/bin/env python3
"""
Heavy Optimizer Platform - Pipeline Orchestrator
Manages the end-to-end data flow of the optimization pipeline with
configuration-driven execution and comprehensive monitoring.
"""

import os
import sys
import time
import logging
import configparser
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import existing modules
from backend.dal.dal_factory import DALFactory
from backend.ulta_calculator import ULTACalculator
from backend.zone_optimizer_dal import ZoneOptimizerDAL as ZoneOptimizer
from backend.lib.correlation.correlation_matrix_calculator import CorrelationMatrixCalculator as CorrelationCalculator
from backend.output_generation_engine import OutputGenerationEngine


class PipelineOrchestrator:
    """
    Orchestrates the Heavy Optimizer Platform pipeline execution.
    
    Key responsibilities:
    - Enforce strict execution order of pipeline steps
    - Read configuration from .ini files
    - Support multiple execution modes (full, zone-wise, dry-run)
    - Provide comprehensive monitoring and logging
    - Track performance metrics for each component
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize the Pipeline Orchestrator.
        
        Args:
            config_path: Path to configuration file. Defaults to production_config.ini
        """
        self.start_time = time.time()
        self.config_path = config_path or "/mnt/optimizer_share/config/production_config.ini"
        
        # Load configuration
        self.config = self._load_configuration()
        
        # Setup logging
        self._setup_logging()
        
        # Initialize components
        self.dal = None
        self.ulta_calculator = ULTACalculator()
        self.zone_optimizer = None
        self.correlation_calculator = CorrelationCalculator()
        self.output_engine = OutputGenerationEngine()
        
        # Performance tracking
        self.step_timings = {}
        self.step_results = {}
        
        # Define pipeline steps in strict order
        self.pipeline_steps = [
            ("load_csv_to_database", self._step_load_csv),
            ("validate_data", self._step_validate_data),
            ("apply_ulta_transformation", self._step_apply_ulta),
            ("compute_correlation_matrix", self._step_compute_correlation),
            ("run_optimization_algorithms", self._step_run_algorithms),
            ("select_final_portfolio", self._step_select_portfolio),
            ("calculate_final_metrics", self._step_calculate_metrics),
            ("generate_output_files", self._step_generate_outputs)
        ]
        
        self.logger.info(f"Pipeline Orchestrator initialized with config: {self.config_path}")
        
    def _load_configuration(self) -> configparser.ConfigParser:
        """Load configuration from .ini file."""
        # Use RawConfigParser to avoid interpolation issues with % characters
        config = configparser.RawConfigParser()
        
        if not Path(self.config_path).exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
            
        config.read(self.config_path)
        return config
        
    def _setup_logging(self):
        """Setup logging based on configuration."""
        log_config = self.config['LOGGING']
        
        # Create log directory if needed
        log_dir = Path(log_config.get('log_directory', '/mnt/optimizer_share/logs'))
        
        # Use fallback if primary log directory isn't writable
        if not os.access(log_dir.parent, os.W_OK):
            log_dir = Path('/mnt/optimizer_share/logs')
            
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logger
        log_level = getattr(logging, log_config.get('log_level', 'INFO'))
        log_file = log_dir / f"pipeline_orchestrator_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger('PipelineOrchestrator')
        self.logger.info(f"Logging initialized. Log file: {log_file}")
        
    def execute_pipeline(self, input_csv: str, portfolio_size: int = 35, 
                        execution_mode: str = 'full', output_dir: str = None) -> Dict[str, Any]:
        """
        Execute the complete optimization pipeline.
        
        Args:
            input_csv: Path to input CSV file
            portfolio_size: Target portfolio size
            execution_mode: Execution mode (full, zone-wise, dry-run)
            output_dir: Override output directory (optional)
            
        Returns:
            Dictionary containing execution results and metrics
        """
        self.logger.info(f"Starting pipeline execution")
        self.logger.info(f"Input CSV: {input_csv}")
        self.logger.info(f"Portfolio Size: {portfolio_size}")
        self.logger.info(f"Execution Mode: {execution_mode}")
        
        # Validate execution mode
        if execution_mode not in ['full', 'zone-wise', 'dry-run']:
            raise ValueError(f"Invalid execution mode: {execution_mode}")
            
        # Setup execution context
        context = {
            'input_csv': input_csv,
            'portfolio_size': portfolio_size,
            'execution_mode': execution_mode,
            'output_dir': output_dir or self._create_output_directory(),
            'start_time': time.time()
        }
        
        # Initialize DAL based on configuration
        self._initialize_dal(context)
        
        # Execute pipeline steps in strict order
        for step_name, step_func in self.pipeline_steps:
            self.logger.info(f"Executing step: {step_name}")
            step_start = time.time()
            
            try:
                # Skip actual execution in dry-run mode
                if execution_mode == 'dry-run':
                    self.logger.info(f"[DRY-RUN] Would execute: {step_name}")
                    self.step_results[step_name] = {'status': 'dry-run'}
                else:
                    # Execute the step
                    result = step_func(context)
                    self.step_results[step_name] = result
                    
                # Track timing
                step_time = time.time() - step_start
                self.step_timings[step_name] = step_time
                self.logger.info(f"Step '{step_name}' completed in {step_time:.3f}s")
                
            except Exception as e:
                self.logger.error(f"Error in step '{step_name}': {str(e)}")
                self._handle_step_error(step_name, e, context)
                raise
                
        # Calculate total execution time
        total_time = time.time() - context['start_time']
        
        # Prepare final results
        results = {
            'status': 'completed',
            'execution_mode': execution_mode,
            'input_file': input_csv,
            'output_directory': context['output_dir'],
            'total_execution_time': total_time,
            'step_timings': self.step_timings,
            'step_results': self.step_results,
            'portfolio_size': portfolio_size
        }
        
        self._log_execution_summary(results)
        
        return results
        
    def _initialize_dal(self, context: Dict[str, Any]):
        """Initialize Data Access Layer based on configuration."""
        # For now, use CSV DAL as HeavyDB is not yet integrated
        backend = 'csv'  # Will be configurable later
        
        # DAL Factory expects a config path, not the config object
        self.dal = DALFactory.create_dal(backend, self.config_path)
        self.zone_optimizer = ZoneOptimizer(self.dal)
        
        self.logger.info(f"DAL initialized with backend: {backend}")
        
    def _create_output_directory(self) -> str:
        """Create timestamped output directory."""
        paths_config = self.config['PATHS']
        output_base = Path(paths_config['output_base_directory'])
        
        timestamp = datetime.now().strftime(self.config['PATHS']['timestamp_format'])
        output_dir = output_base / f"run_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Created output directory: {output_dir}")
        return str(output_dir)
        
    def _step_load_csv(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Step 1: Load CSV to database/memory."""
        self.logger.info("Loading CSV data...")
        
        # Load data using DAL
        data = self.dal.load_data(context['input_csv'])
        
        # Store in context for next steps
        context['data'] = data
        context['row_count'] = len(data)
        context['column_count'] = len(data.columns) if hasattr(data, 'columns') else 0
        
        return {
            'status': 'success',
            'rows_loaded': context['row_count'],
            'columns_loaded': context['column_count']
        }
        
    def _step_validate_data(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Step 2: Validate loaded data."""
        self.logger.info("Validating data...")
        
        validation_config = self.config['INPUT_PROCESSING']
        
        # Validation checks
        issues = []
        
        # Check minimum rows
        min_rows = int(validation_config.get('min_rows', 50))
        if context['row_count'] < min_rows:
            issues.append(f"Row count {context['row_count']} below minimum {min_rows}")
            
        # Check minimum columns  
        min_cols = int(validation_config.get('min_columns', 10))
        if context['column_count'] < min_cols:
            issues.append(f"Column count {context['column_count']} below minimum {min_cols}")
            
        # Additional validation can be added here
        
        return {
            'status': 'success' if not issues else 'warning',
            'validation_issues': issues,
            'is_valid': len(issues) == 0
        }
        
    def _step_apply_ulta(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Step 3: Apply ULTA transformation."""
        self.logger.info("Applying ULTA transformation...")
        
        # Apply ULTA using the calculator
        ulta_results = self.ulta_calculator.calculate_ulta_scores(context['data'])
        
        # Store results in context
        context['ulta_results'] = ulta_results
        
        return {
            'status': 'success',
            'strategies_processed': len(ulta_results) if isinstance(ulta_results, list) else 1
        }
        
    def _step_compute_correlation(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Step 4: Compute correlation matrix."""
        self.logger.info("Computing correlation matrix...")
        
        correlation_config = self.config['CORRELATION']
        
        # Compute correlations
        correlation_matrix = self.correlation_calculator.compute_correlation_matrix(
            context['data'],
            gpu_acceleration=correlation_config.getboolean('gpu_acceleration', True)
        )
        
        # Store in context
        context['correlation_matrix'] = correlation_matrix
        
        return {
            'status': 'success',
            'matrix_shape': correlation_matrix.shape if hasattr(correlation_matrix, 'shape') else None
        }
        
    def _step_run_algorithms(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Step 5: Run optimization algorithms."""
        self.logger.info("Running optimization algorithms...")
        
        algorithms_config = self.config['ALGORITHMS']
        
        # Determine which algorithms to run
        enabled_algorithms = []
        for algo in ['sa', 'ga', 'pso', 'de', 'aco', 'bo', 'rs']:
            if algorithms_config.getboolean(f'{algo}_enabled', True):
                enabled_algorithms.append(algo.upper())
                
        self.logger.info(f"Running algorithms: {enabled_algorithms}")
        
        # Run algorithms (placeholder - actual implementation would call algorithm modules)
        algorithm_results = {}
        for algo in enabled_algorithms:
            # Simulate algorithm execution
            algorithm_results[algo] = {
                'fitness': 0.85 + (hash(algo) % 15) / 100,  # Placeholder
                'portfolio': list(range(context['portfolio_size'])),  # Placeholder
                'execution_time': float(algorithms_config.get(f'{algo.lower()}_execution_time', 0.1))
            }
            
        context['algorithm_results'] = algorithm_results
        
        return {
            'status': 'success',
            'algorithms_run': enabled_algorithms,
            'algorithm_count': len(enabled_algorithms)
        }
        
    def _step_select_portfolio(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Step 6: Select final portfolio."""
        self.logger.info("Selecting final portfolio...")
        
        algorithms_config = self.config['ALGORITHMS']
        
        # Select best algorithm based on fitness
        if algorithms_config.getboolean('auto_select_best', True):
            best_algo = max(context['algorithm_results'].items(), 
                          key=lambda x: x[1]['fitness'])[0]
        else:
            best_algo = algorithms_config.get('default_best_algorithm', 'SA')
            
        # Get best portfolio
        best_portfolio = context['algorithm_results'][best_algo]['portfolio']
        
        context['best_algorithm'] = best_algo
        context['final_portfolio'] = best_portfolio
        
        return {
            'status': 'success',
            'selected_algorithm': best_algo,
            'portfolio_size': len(best_portfolio)
        }
        
    def _step_calculate_metrics(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Step 7: Calculate final metrics."""
        self.logger.info("Calculating final metrics...")
        
        # Calculate comprehensive metrics (placeholder)
        metrics = {
            'total_roi': 150.5,
            'max_drawdown': -45000,
            'win_rate': 0.65,
            'profit_factor': 1.35,
            'sharpe_ratio': 1.2,
            'roi_drawdown_ratio': 3.34
        }
        
        context['final_metrics'] = metrics
        
        return {
            'status': 'success',
            'metrics_calculated': list(metrics.keys())
        }
        
    def _step_generate_outputs(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Step 8: Generate output files."""
        self.logger.info("Generating output files...")
        
        output_config = self.config['OUTPUT_GENERATION']
        
        # Generate outputs using output engine
        generated_files = []
        
        # Generate summary report
        if output_config.getboolean('generate_summary_report', True):
            summary_file = self._generate_summary_report(context)
            generated_files.append(summary_file)
            
        # Generate other outputs (placeholder)
        # This would call the actual output generation engine
        
        return {
            'status': 'success',
            'files_generated': generated_files,
            'output_directory': context['output_dir']
        }
        
    def _generate_summary_report(self, context: Dict[str, Any]) -> str:
        """Generate summary report file."""
        output_dir = Path(context['output_dir'])
        timestamp = datetime.now().strftime(self.config['PATHS']['timestamp_format'])
        summary_file = output_dir / f"optimization_summary_{timestamp}.txt"
        
        with open(summary_file, 'w') as f:
            f.write("Heavy Optimizer Platform - Optimization Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Execution Mode: {context['execution_mode']}\n")
            f.write(f"Input File: {context['input_csv']}\n")
            f.write(f"Portfolio Size: {context['portfolio_size']}\n")
            f.write(f"Best Algorithm: {context.get('best_algorithm', 'N/A')}\n")
            f.write(f"\nExecution Time: {time.time() - context['start_time']:.2f}s\n")
            
        return str(summary_file)
        
    def _handle_step_error(self, step_name: str, error: Exception, context: Dict[str, Any]):
        """Handle errors during step execution."""
        error_info = {
            'step': step_name,
            'error_type': type(error).__name__,
            'error_message': str(error),
            'timestamp': datetime.now().isoformat()
        }
        
        # Log error details
        self.logger.error(f"Step error details: {error_info}")
        
        # Save error information if output directory exists
        if 'output_dir' in context:
            error_file = Path(context['output_dir']) / 'error_log.txt'
            with open(error_file, 'a') as f:
                f.write(f"\n{datetime.now()}: Error in {step_name}: {str(error)}\n")
                
    def _log_execution_summary(self, results: Dict[str, Any]):
        """Log execution summary."""
        self.logger.info("=" * 60)
        self.logger.info("PIPELINE EXECUTION SUMMARY")
        self.logger.info("=" * 60)
        self.logger.info(f"Status: {results['status']}")
        self.logger.info(f"Total Execution Time: {results['total_execution_time']:.2f}s")
        self.logger.info(f"Output Directory: {results['output_directory']}")
        
        self.logger.info("\nStep Timings:")
        for step, timing in results['step_timings'].items():
            self.logger.info(f"  {step}: {timing:.3f}s")
            
    def get_supported_execution_modes(self) -> List[str]:
        """Get list of supported execution modes."""
        return ['full', 'zone-wise', 'dry-run']
        
    def validate_configuration(self) -> Dict[str, Any]:
        """Validate the loaded configuration."""
        issues = []
        warnings = []
        
        # Check required sections
        required_sections = ['SYSTEM', 'PATHS', 'ALGORITHMS', 'OUTPUT_GENERATION']
        for section in required_sections:
            if section not in self.config:
                issues.append(f"Missing required section: {section}")
                
        # Check critical paths
        if 'PATHS' in self.config:
            for key in ['base_directory', 'input_directory', 'output_base_directory']:
                if key in self.config['PATHS']:
                    path = Path(self.config['PATHS'][key])
                    if not path.exists():
                        warnings.append(f"Path does not exist: {key} = {path}")
                        
        return {
            'is_valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings
        }


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Heavy Optimizer Platform - Pipeline Orchestrator")
    parser.add_argument('--input', required=True, help='Input CSV file path')
    parser.add_argument('--portfolio-size', type=int, default=35, help='Target portfolio size')
    parser.add_argument('--mode', choices=['full', 'zone-wise', 'dry-run'], default='full',
                       help='Execution mode')
    parser.add_argument('--config', help='Configuration file path')
    parser.add_argument('--output-dir', help='Override output directory')
    
    args = parser.parse_args()
    
    try:
        # Create orchestrator
        orchestrator = PipelineOrchestrator(config_path=args.config)
        
        # Validate configuration
        validation = orchestrator.validate_configuration()
        if not validation['is_valid']:
            print(f"Configuration validation failed: {validation['issues']}")
            sys.exit(1)
            
        # Execute pipeline
        results = orchestrator.execute_pipeline(
            input_csv=args.input,
            portfolio_size=args.portfolio_size,
            execution_mode=args.mode,
            output_dir=args.output_dir
        )
        
        print(f"\nPipeline execution completed successfully!")
        print(f"Output directory: {results['output_directory']}")
        print(f"Total time: {results['total_execution_time']:.2f}s")
        
    except Exception as e:
        print(f"Pipeline execution failed: {str(e)}")
        sys.exit(1)