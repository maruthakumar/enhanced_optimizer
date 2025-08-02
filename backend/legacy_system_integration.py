#!/usr/bin/env python3
"""
Legacy System Integration for Heavy Optimizer Platform
Executes the legacy Optimizer_New_patched.py and validates results against new implementation
"""

import os
import sys
import subprocess
import time
import json
import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class LegacySystemExecutor:
    """Executes the legacy optimizer system and captures results"""
    
    def __init__(self, legacy_base_path: str = "/mnt/optimizer_share/zone_optimization_25_06_25"):
        self.legacy_base_path = Path(legacy_base_path)
        self.legacy_script = self.legacy_base_path / "Optimizer_New_patched.py"
        self.logger = logging.getLogger(self.__class__.__name__)
        
        if not self.legacy_script.exists():
            raise FileNotFoundError(f"Legacy script not found: {self.legacy_script}")
    
    def execute_legacy_optimizer(self, 
                               input_file: str, 
                               portfolio_size: int,
                               min_size: Optional[int] = None,
                               max_size: Optional[int] = None,
                               config_file: Optional[str] = None,
                               timeout: int = 3600) -> Tuple[str, int]:
        """
        Execute the legacy optimizer with given parameters
        
        Args:
            input_file: Path to input CSV file
            portfolio_size: Target portfolio size (used if min/max not specified)
            min_size: Minimum portfolio size
            max_size: Maximum portfolio size
            timeout: Execution timeout in seconds
            
        Returns:
            Tuple of (output_directory, return_code)
        """
        self.logger.info(f"Executing legacy optimizer for portfolio size {portfolio_size}")
        
        # Prepare execution environment
        env = os.environ.copy()
        env['PYTHONPATH'] = str(self.legacy_base_path)
        
        # Change to legacy directory for execution
        original_cwd = os.getcwd()
        os.chdir(self.legacy_base_path)
        
        try:
            # Use default config if not specified
            if config_file is None:
                config_file = str(self.legacy_base_path / "config_zone.ini")
            
            # Build command with config file
            cmd = [sys.executable, str(self.legacy_script), "--config", config_file]
            
            # Create temporary config file if we need to override portfolio size
            if min_size is not None and max_size is not None:
                config_file = self._create_temp_config(config_file, min_size, max_size)
                cmd = [sys.executable, str(self.legacy_script), "--config", config_file]
            elif portfolio_size:
                config_file = self._create_temp_config(config_file, portfolio_size, portfolio_size)
                cmd = [sys.executable, str(self.legacy_script), "--config", config_file]
            
            # Execute the optimizer
            start_time = time.time()
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                text=True
            )
            
            try:
                stdout, stderr = process.communicate(timeout=timeout)
                return_code = process.returncode
                execution_time = time.time() - start_time
                
                self.logger.info(f"Legacy optimizer completed in {execution_time:.2f}s with code {return_code}")
                
                if return_code != 0:
                    self.logger.error(f"Legacy optimizer failed: {stderr}")
                    return None, return_code
                
                # Find the output directory
                output_dir = self._find_latest_output_dir()
                
                return str(output_dir), return_code
                
            except subprocess.TimeoutExpired:
                process.kill()
                self.logger.error(f"Legacy optimizer timed out after {timeout}s")
                return None, -1
                
        finally:
            # Restore original directory
            os.chdir(original_cwd)
    
    def _find_latest_output_dir(self) -> Path:
        """Find the most recently created output directory"""
        output_base = self.legacy_base_path / "Output"
        
        # Find all run directories
        run_dirs = [d for d in output_base.iterdir() if d.is_dir() and d.name.startswith("run_")]
        
        if not run_dirs:
            raise RuntimeError("No output directories found")
        
        # Return the most recent
        return max(run_dirs, key=lambda d: d.stat().st_mtime)
    
    def _create_temp_config(self, base_config: str, min_size: int, max_size: int) -> str:
        """Create a temporary config file with specified portfolio sizes"""
        import configparser
        import tempfile
        
        config = configparser.ConfigParser()
        config.read(base_config)
        
        # Update portfolio sizes
        if 'PORTFOLIO' not in config:
            config['PORTFOLIO'] = {}
        config['PORTFOLIO']['min_size'] = str(min_size)
        config['PORTFOLIO']['max_size'] = str(max_size)
        
        # Create temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False, 
                                        dir=self.legacy_base_path) as f:
            config.write(f)
            return f.name


class LegacyOutputParser:
    """Parses output from the legacy optimizer system"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def parse_legacy_results(self, output_directory: str) -> Dict[str, Any]:
        """
        Parse results from legacy optimizer output directory
        
        Args:
            output_directory: Path to the output directory
            
        Returns:
            Dictionary containing parsed results
        """
        output_path = Path(output_directory)
        
        if not output_path.exists():
            raise ValueError(f"Output directory not found: {output_directory}")
        
        results = {
            'output_directory': str(output_path),
            'timestamp': output_path.name.split('_')[-1] if '_' in output_path.name else None,
            'portfolio_results': {},
            'best_overall': None
        }
        
        # Parse optimization summary
        summary_files = list(output_path.glob("optimization_summary_*.txt"))
        if summary_files:
            summary_data = self._parse_summary_file(summary_files[0])
            results.update(summary_data)
        
        # Parse individual portfolio results
        portfolio_files = list(output_path.glob("best_portfolio_size*.txt"))
        for pf in portfolio_files:
            size = self._extract_portfolio_size(pf.name)
            if size:
                portfolio_data = self._parse_portfolio_file(pf)
                results['portfolio_results'][size] = portfolio_data
        
        # Parse strategy metrics if available
        metrics_file = output_path / "strategy_metrics.csv"
        if metrics_file.exists():
            results['strategy_metrics'] = pd.read_csv(metrics_file).to_dict('records')
        
        return results
    
    def _parse_summary_file(self, summary_file: Path) -> Dict[str, Any]:
        """Parse the optimization summary file"""
        data = {
            'parameters': {},
            'best_overall': {}
        }
        
        with open(summary_file, 'r') as f:
            content = f.read()
        
        # Extract parameters
        param_section = re.search(r'Optimization Parameters:(.*?)Best Overall Portfolio:', 
                                 content, re.DOTALL)
        if param_section:
            param_text = param_section.group(1)
            for line in param_text.strip().split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip('- ').strip()
                    value = value.strip()
                    data['parameters'][key] = value
        
        # Extract best overall portfolio
        best_section = re.search(r'Best Overall Portfolio:(.*?)Selected Strategies:', 
                                content, re.DOTALL)
        if best_section:
            best_text = best_section.group(1)
            for line in best_text.strip().split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip('- ').strip()
                    value = value.strip()
                    
                    # Convert numeric values
                    if key in ['Size', 'Fitness']:
                        try:
                            value = float(value) if '.' in value else int(value)
                        except ValueError:
                            pass
                    
                    data['best_overall'][key.lower()] = value
        
        # Extract selected strategies
        strategies_section = re.search(r'Selected Strategies:(.*?)(?:Performance Metrics:|$)', 
                                     content, re.DOTALL)
        if strategies_section:
            strategies = []
            for line in strategies_section.group(1).strip().split('\n'):
                if re.match(r'^\d+\.', line):
                    strategy = line.split('.', 1)[1].strip()
                    strategies.append(strategy)
            data['best_overall']['strategies'] = strategies
        
        return data
    
    def _parse_portfolio_file(self, portfolio_file: Path) -> Dict[str, Any]:
        """Parse individual portfolio result file"""
        data = {}
        
        with open(portfolio_file, 'r') as f:
            content = f.read()
        
        # Extract key metrics
        for pattern, key in [
            (r'Method:\s*(\w+)', 'method'),
            (r'Fitness:\s*([\d.]+)', 'fitness'),
            (r'Total return:\s*([\d.]+)%', 'total_return'),
            (r'Max drawdown:\s*([\d.]+)%', 'max_drawdown'),
            (r'Win rate:\s*([\d.]+)%', 'win_rate'),
            (r'Profit factor:\s*([\d.]+)', 'profit_factor')
        ]:
            match = re.search(pattern, content)
            if match:
                value = match.group(1)
                # Convert numeric values
                if key in ['fitness', 'total_return', 'max_drawdown', 'win_rate', 'profit_factor']:
                    try:
                        value = float(value)
                    except ValueError:
                        pass
                data[key] = value
        
        # Extract strategies
        strategies = []
        for line in content.split('\n'):
            if line.strip().startswith('- '):
                strategies.append(line.strip()[2:])
        data['strategies'] = strategies
        
        return data
    
    def _extract_portfolio_size(self, filename: str) -> Optional[int]:
        """Extract portfolio size from filename"""
        match = re.search(r'size(\d+)', filename)
        if match:
            return int(match.group(1))
        return None


class FitnessCalculationValidator:
    """Validates fitness calculations between legacy and new systems"""
    
    def __init__(self, tolerance: float = 0.0001):
        self.tolerance = tolerance
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def validate_fitness_parity(self, 
                              legacy_fitness: float, 
                              new_fitness: float,
                              tolerance: Optional[float] = None) -> bool:
        """
        Validate that fitness values match within tolerance
        
        Args:
            legacy_fitness: Fitness value from legacy system
            new_fitness: Fitness value from new system
            tolerance: Optional override for tolerance (default 0.01%)
            
        Returns:
            True if values match within tolerance
        """
        if tolerance is None:
            tolerance = self.tolerance
        
        if legacy_fitness == 0:
            return new_fitness == 0
        
        relative_diff = abs(legacy_fitness - new_fitness) / abs(legacy_fitness)
        
        return relative_diff <= tolerance
    
    def generate_comparison_report(self, 
                                 legacy_results: Dict[str, Any],
                                 new_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a detailed comparison report between legacy and new results
        
        Args:
            legacy_results: Results from legacy system
            new_results: Results from new system
            
        Returns:
            Dictionary containing comparison data
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {},
            'portfolio_comparisons': {},
            'deviations': []
        }
        
        # Compare best overall results
        if 'best_overall' in legacy_results and 'best_algorithm' in new_results:
            legacy_best = legacy_results['best_overall']
            
            report['summary'] = {
                'legacy_fitness': legacy_best.get('fitness', 0),
                'new_fitness': new_results.get('best_fitness', 0),
                'legacy_algorithm': legacy_best.get('method', ''),
                'new_algorithm': new_results.get('best_algorithm', ''),
                'fitness_parity': False
            }
            
            # Check fitness parity
            if 'fitness' in legacy_best and 'best_fitness' in new_results:
                parity = self.validate_fitness_parity(
                    legacy_best['fitness'],
                    new_results['best_fitness']
                )
                report['summary']['fitness_parity'] = parity
                report['summary']['relative_difference'] = abs(
                    legacy_best['fitness'] - new_results['best_fitness']
                ) / legacy_best['fitness'] if legacy_best['fitness'] != 0 else 0
                
                if not parity:
                    report['deviations'].append({
                        'type': 'fitness_mismatch',
                        'legacy_value': legacy_best['fitness'],
                        'new_value': new_results['best_fitness'],
                        'difference': abs(legacy_best['fitness'] - new_results['best_fitness'])
                    })
        
        # Compare individual portfolio results
        if 'portfolio_results' in legacy_results:
            for size, legacy_portfolio in legacy_results['portfolio_results'].items():
                comparison = {
                    'size': size,
                    'legacy': legacy_portfolio,
                    'new': None,
                    'matches': False
                }
                
                # Find corresponding new result
                # This will need to be implemented based on how new results are structured
                
                report['portfolio_comparisons'][size] = comparison
        
        # Flag significant deviations
        if report['deviations']:
            self.logger.warning(f"Found {len(report['deviations'])} deviations between systems")
        
        return report
    
    def save_comparison_report(self, report: Dict[str, Any], output_path: str):
        """Save comparison report to file"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"Comparison report saved to {output_file}")


class LegacySystemIntegration:
    """Main integration class for legacy system"""
    
    def __init__(self, legacy_base_path: Optional[str] = None):
        self.executor = LegacySystemExecutor(legacy_base_path) if legacy_base_path else LegacySystemExecutor()
        self.parser = LegacyOutputParser()
        self.validator = FitnessCalculationValidator()
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def run_legacy_comparison(self, 
                            input_file: str,
                            portfolio_size: int,
                            new_results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run legacy system and compare with new results
        
        Args:
            input_file: Path to input CSV file
            portfolio_size: Target portfolio size
            new_results: Optional results from new system for comparison
            
        Returns:
            Dictionary containing legacy results and comparison report
        """
        self.logger.info(f"Starting legacy system comparison for portfolio size {portfolio_size}")
        
        # Execute legacy system
        output_dir, return_code = self.executor.execute_legacy_optimizer(
            input_file, portfolio_size
        )
        
        if return_code != 0:
            raise RuntimeError(f"Legacy system execution failed with code {return_code}")
        
        # Parse results
        legacy_results = self.parser.parse_legacy_results(output_dir)
        
        response = {
            'legacy_results': legacy_results,
            'execution_status': 'success'
        }
        
        # Compare with new results if provided
        if new_results:
            comparison_report = self.validator.generate_comparison_report(
                legacy_results, new_results
            )
            response['comparison_report'] = comparison_report
            
            # Save comparison report
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = f"/mnt/optimizer_share/output/legacy_comparison/comparison_{timestamp}.json"
            self.validator.save_comparison_report(comparison_report, report_path)
        
        return response


def main():
    """Test the legacy system integration"""
    integration = LegacySystemIntegration()
    
    # Test with sample data
    input_file = "/mnt/optimizer_share/input/Python_Multi_Consolidated_20250726_161921.csv"
    portfolio_size = 37
    
    try:
        results = integration.run_legacy_comparison(input_file, portfolio_size)
        
        print("Legacy System Results:")
        print(f"Best Overall: {results['legacy_results'].get('best_overall', {})}")
        
    except Exception as e:
        logging.error(f"Test failed: {e}")
        raise


if __name__ == "__main__":
    main()