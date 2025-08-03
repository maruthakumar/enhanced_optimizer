#!/usr/bin/env python3
"""
Legacy System Wrapper for Portfolio Optimizer
Executes the legacy Optimizer_New_patched.py and captures results
"""

import os
import sys
import subprocess
import logging
import time
import json
import shutil
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LegacySystemWrapper:
    """
    Wrapper for executing legacy optimizer system and capturing results
    """
    
    def __init__(self, legacy_base_path: str = "/mnt/optimizer_share/zone_optimization_25_06_25"):
        """
        Initialize wrapper with legacy system path
        
        Args:
            legacy_base_path: Path to legacy optimizer directory
        """
        self.legacy_base_path = Path(legacy_base_path)
        self.optimizer_script = self.legacy_base_path / "Optimizer_New_patched.py"
        self.config_zone = self.legacy_base_path / "config_zone.ini"
        self.config_consol = self.legacy_base_path / "config_consol.ini"
        self.output_dir = self.legacy_base_path / "Output"
        
        # Verify paths exist
        if not self.optimizer_script.exists():
            raise FileNotFoundError(f"Legacy optimizer script not found: {self.optimizer_script}")
        
        logger.info(f"Legacy system wrapper initialized with base path: {self.legacy_base_path}")
        
    def execute_legacy_system(self, 
                            input_csv: str,
                            portfolio_sizes: List[int],
                            timeout_minutes: int = 30,
                            use_zone_mode: bool = False) -> Dict[str, Any]:
        """
        Execute legacy optimizer system
        
        Args:
            input_csv: Path to input CSV file
            portfolio_sizes: List of portfolio sizes to optimize
            timeout_minutes: Maximum execution time in minutes
            use_zone_mode: Whether to use zone-based mode
            
        Returns:
            Dictionary containing execution results and output paths
        """
        logger.info(f"Executing legacy system with input: {input_csv}")
        logger.info(f"Portfolio sizes: {portfolio_sizes}")
        
        # Create unique run ID
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Prepare input file in legacy format
        input_path = self._prepare_input_file(input_csv)
        
        # Configure portfolio sizes
        self._configure_portfolio_sizes(portfolio_sizes)
        
        # Execute optimizer
        start_time = time.time()
        result = self._run_optimizer(timeout_minutes, use_zone_mode)
        execution_time = time.time() - start_time
        
        # Find output directory
        output_run_dir = self._find_latest_output_dir()
        
        # Collect results
        results = {
            'run_id': run_id,
            'execution_time': execution_time,
            'success': result['success'],
            'output_dir': str(output_run_dir) if output_run_dir else None,
            'error_message': result.get('error'),
            'stdout': result.get('stdout', ''),
            'stderr': result.get('stderr', ''),
            'portfolio_sizes': portfolio_sizes
        }
        
        logger.info(f"Legacy execution completed in {execution_time:.2f} seconds")
        
        return results
        
    def _prepare_input_file(self, input_csv: str) -> Path:
        """
        Copy input file to legacy input directory
        
        Args:
            input_csv: Path to input CSV file
            
        Returns:
            Path to copied file in legacy system
        """
        # Determine target directory based on file type
        input_dir = self.legacy_base_path / "Input" / "Python_Multi_Files"
        input_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy file to legacy input directory
        source_path = Path(input_csv)
        target_path = input_dir / source_path.name
        
        logger.info(f"Copying input file to: {target_path}")
        shutil.copy2(source_path, target_path)
        
        return target_path
        
    def _configure_portfolio_sizes(self, portfolio_sizes: List[int]):
        """
        Update config file with portfolio sizes
        
        Args:
            portfolio_sizes: List of portfolio sizes
        """
        import configparser
        
        # Read existing config
        config = configparser.ConfigParser()
        config.read(self.config_zone)
        
        # Update portfolio size range
        if 'portfolio' not in config:
            config['portfolio'] = {}
            
        config['portfolio']['min_size'] = str(min(portfolio_sizes))
        config['portfolio']['max_size'] = str(max(portfolio_sizes))
        
        # Write updated config
        with open(self.config_zone, 'w') as f:
            config.write(f)
            
        logger.info(f"Updated config with portfolio sizes: {min(portfolio_sizes)}-{max(portfolio_sizes)}")
        
    def _run_optimizer(self, timeout_minutes: int, use_zone_mode: bool) -> Dict[str, Any]:
        """
        Execute the optimizer script
        
        Args:
            timeout_minutes: Maximum execution time
            use_zone_mode: Whether to use zone mode
            
        Returns:
            Execution result dictionary
        """
        # Build command
        cmd = [
            sys.executable,
            str(self.optimizer_script)
        ]
        
        if use_zone_mode:
            cmd.append("--zone-mode")
            
        logger.info(f"Executing command: {' '.join(cmd)}")
        
        try:
            # Run with timeout
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(self.legacy_base_path),
                text=True
            )
            
            stdout, stderr = process.communicate(timeout=timeout_minutes * 60)
            
            if process.returncode == 0:
                return {
                    'success': True,
                    'stdout': stdout,
                    'stderr': stderr
                }
            else:
                return {
                    'success': False,
                    'error': f"Process exited with code {process.returncode}",
                    'stdout': stdout,
                    'stderr': stderr
                }
                
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()
            return {
                'success': False,
                'error': f"Execution timed out after {timeout_minutes} minutes",
                'stdout': stdout if stdout else '',
                'stderr': stderr if stderr else ''
            }
        except Exception as e:
            return {
                'success': False,
                'error': f"Execution failed: {str(e)}"
            }
            
    def _find_latest_output_dir(self) -> Optional[Path]:
        """
        Find the most recently created output directory
        
        Returns:
            Path to latest output directory or None
        """
        try:
            # List all run directories
            run_dirs = [d for d in self.output_dir.iterdir() 
                       if d.is_dir() and d.name.startswith('run_')]
            
            if not run_dirs:
                return None
                
            # Sort by modification time
            latest = max(run_dirs, key=lambda d: d.stat().st_mtime)
            
            logger.info(f"Found latest output directory: {latest}")
            return latest
            
        except Exception as e:
            logger.error(f"Error finding output directory: {e}")
            return None
            
    def get_legacy_results(self, output_dir: str) -> Dict[str, Any]:
        """
        Parse results from legacy output directory
        
        Args:
            output_dir: Path to output directory
            
        Returns:
            Dictionary containing parsed results
        """
        output_path = Path(output_dir)
        results = {
            'portfolio_results': {},
            'optimization_summary': None,
            'strategy_metrics': None
        }
        
        # Parse optimization summary
        summary_files = list(output_path.glob("optimization_summary_*.txt"))
        if summary_files:
            results['optimization_summary'] = self._parse_optimization_summary(summary_files[0])
            
        # Parse best portfolio files
        for portfolio_file in output_path.glob("best_portfolio_size*_*.txt"):
            size = self._extract_portfolio_size(portfolio_file.name)
            if size:
                results['portfolio_results'][size] = self._parse_portfolio_file(portfolio_file)
                
        # Parse strategy metrics
        metrics_file = output_path / "strategy_metrics.csv"
        if metrics_file.exists():
            results['strategy_metrics'] = pd.read_csv(metrics_file).to_dict('records')
            
        return results
        
    def _parse_optimization_summary(self, file_path: Path) -> Dict[str, Any]:
        """Parse optimization summary file"""
        summary = {
            'run_id': None,
            'date': None,
            'parameters': {},
            'best_portfolio': {}
        }
        
        with open(file_path, 'r') as f:
            lines = f.readlines()
            
        # Parse the structured data
        for i, line in enumerate(lines):
            line = line.strip()
            
            if line.startswith("Run ID:"):
                summary['run_id'] = line.split(":", 1)[1].strip()
            elif line.startswith("Date:"):
                summary['date'] = line.split(":", 1)[1].strip()
            elif line.startswith("- Size:"):
                summary['best_portfolio']['size'] = int(line.split(":", 1)[1].strip())
            elif line.startswith("- Method:"):
                summary['best_portfolio']['method'] = line.split(":", 1)[1].strip()
            elif line.startswith("- Fitness:"):
                summary['best_portfolio']['fitness'] = float(line.split(":", 1)[1].strip())
                
        return summary
        
    def _parse_portfolio_file(self, file_path: Path) -> Dict[str, Any]:
        """Parse individual portfolio result file"""
        portfolio = {
            'method': None,
            'fitness': None,
            'metrics': {},
            'strategies': []
        }
        
        with open(file_path, 'r') as f:
            lines = f.readlines()
            
        parsing_metrics = False
        parsing_strategies = False
        
        for line in lines:
            line = line.strip()
            
            if line.startswith("Best Portfolio"):
                # Extract method from first line
                if "Method:" in line:
                    portfolio['method'] = line.split("Method:", 1)[1].strip()
            elif line.startswith("Fitness:"):
                portfolio['fitness'] = float(line.split(":", 1)[1].strip())
            elif line == "Performance Metrics:":
                parsing_metrics = True
                parsing_strategies = False
            elif line == "Selected Strategies:":
                parsing_metrics = False
                parsing_strategies = True
            elif parsing_metrics and ":" in line:
                key, value = line.split(":", 1)
                try:
                    portfolio['metrics'][key.strip()] = float(value.strip())
                except ValueError:
                    portfolio['metrics'][key.strip()] = value.strip()
            elif parsing_strategies and line and line[0].isdigit():
                # Parse strategy line
                parts = line.split(".", 1)
                if len(parts) > 1:
                    portfolio['strategies'].append(parts[1].strip())
                    
        return portfolio
        
    def _extract_portfolio_size(self, filename: str) -> Optional[int]:
        """Extract portfolio size from filename"""
        import re
        match = re.search(r'size(\d+)_', filename)
        if match:
            return int(match.group(1))
        return None


def main():
    """Test the legacy wrapper"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Legacy System Wrapper')
    parser.add_argument('--input', required=True, help='Input CSV file')
    parser.add_argument('--sizes', nargs='+', type=int, default=[35, 37, 50, 60],
                       help='Portfolio sizes to test')
    parser.add_argument('--timeout', type=int, default=30,
                       help='Timeout in minutes')
    
    args = parser.parse_args()
    
    # Create wrapper
    wrapper = LegacySystemWrapper()
    
    # Execute legacy system
    results = wrapper.execute_legacy_system(
        input_csv=args.input,
        portfolio_sizes=args.sizes,
        timeout_minutes=args.timeout
    )
    
    print(f"Execution results: {json.dumps(results, indent=2)}")
    
    # Parse results if successful
    if results['success'] and results['output_dir']:
        parsed = wrapper.get_legacy_results(results['output_dir'])
        print(f"\nParsed results: {json.dumps(parsed, indent=2, default=str)}")


if __name__ == "__main__":
    main()