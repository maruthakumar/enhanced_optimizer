#!/usr/bin/env python3
"""
Main Benchmark Runner Script
Orchestrates benchmark execution, report generation, and CI integration
"""

import os
import sys
import json
import argparse
import logging
import subprocess
from datetime import datetime
from pathlib import Path

# Add scripts directory to path
scripts_dir = Path(__file__).parent / 'scripts'
sys.path.append(str(scripts_dir))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_benchmark_suite(scenario=None, fail_on_sla=False, generate_reports=True, 
                       verbose=False, output_dir=None):
    """
    Run the complete benchmark suite with optional report generation.
    
    Args:
        scenario: Specific scenario name to run (None for all)
        fail_on_sla: Exit with error code on SLA violations
        generate_reports: Generate HTML/CSV reports after benchmarking
        verbose: Enable verbose logging
        output_dir: Custom output directory for results
    
    Returns:
        Dictionary with execution results
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Setup paths
    base_dir = Path(__file__).parent
    scripts_dir = base_dir / 'scripts'
    reports_dir = base_dir / 'reports'
    
    if output_dir:
        reports_dir = Path(output_dir)
    
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'start_time': datetime.now().isoformat(),
        'scenario_filter': scenario,
        'fail_on_sla': fail_on_sla,
        'benchmark_results': None,
        'reports_generated': [],
        'status': 'running'
    }
    
    try:
        # Step 1: Generate test data if needed
        logger.info("🔄 Checking test data availability...")
        data_dir = base_dir / 'data'
        
        # Check if we have recent test data
        test_data_needed = True
        if data_dir.exists():
            recent_files = [
                f for f in data_dir.glob('*.csv') 
                if (datetime.now() - datetime.fromtimestamp(f.stat().st_mtime)).days < 7
            ]
            if len(recent_files) >= 4:  # We need at least 4 scenario datasets
                test_data_needed = False
                logger.info("✅ Using existing test data")
        
        if test_data_needed:
            logger.info("🔄 Generating fresh test datasets...")
            cmd = [sys.executable, str(scripts_dir / 'generate_test_data.py'), '--all-scenarios']
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=scripts_dir)
            
            if result.returncode != 0:
                raise RuntimeError(f"Test data generation failed: {result.stderr}")
            
            logger.info("✅ Test datasets generated successfully")
        
        # Step 2: Run benchmarks
        logger.info("🚀 Running benchmark suite...")
        
        benchmark_cmd = [sys.executable, str(scripts_dir / 'parquet_pipeline_benchmark.py')]
        
        if scenario:
            benchmark_cmd.extend(['--scenario', scenario])
        
        if fail_on_sla:
            benchmark_cmd.append('--fail-on-sla')
        
        if verbose:
            benchmark_cmd.append('--verbose')
        
        # Set output file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        benchmark_output = reports_dir / f"benchmark_results_{timestamp}.json"
        benchmark_cmd.extend(['--output', str(benchmark_output)])
        
        logger.info(f"Running: {' '.join(benchmark_cmd)}")
        
        benchmark_result = subprocess.run(
            benchmark_cmd, 
            capture_output=True, 
            text=True, 
            cwd=scripts_dir,
            timeout=1800  # 30 minute timeout
        )
        
        if benchmark_result.returncode != 0:
            logger.error(f"Benchmark execution failed: {benchmark_result.stderr}")
            results['status'] = 'failed'
            results['error'] = benchmark_result.stderr
            results['stdout'] = benchmark_result.stdout
            
            if fail_on_sla and "SLA violations detected" in benchmark_result.stderr:
                logger.error("SLA violations detected - failing as requested")
                results['sla_violations'] = True
                return results
            
            # Even if benchmark fails, try to generate reports if we have partial results
            if not benchmark_output.exists():
                return results
        
        logger.info("✅ Benchmarks completed successfully")
        results['benchmark_results'] = str(benchmark_output)
        
        # Step 3: Generate reports
        if generate_reports and benchmark_output.exists():
            logger.info("📊 Generating comprehensive reports...")
            
            report_cmd = [
                sys.executable, 
                str(scripts_dir / 'generate_report.py'),
                '--results', str(benchmark_output),
                '--output-dir', str(reports_dir),
                '--format', 'all'
            ]
            
            report_result = subprocess.run(
                report_cmd,
                capture_output=True,
                text=True,
                cwd=scripts_dir
            )
            
            if report_result.returncode == 0:
                logger.info("✅ Reports generated successfully")
                
                # Find generated report files
                for report_file in reports_dir.glob(f"*{timestamp}*"):
                    if report_file.suffix in ['.html', '.csv']:
                        results['reports_generated'].append(str(report_file))
                
            else:
                logger.warning(f"Report generation failed: {report_result.stderr}")
                results['report_error'] = report_result.stderr
        
        # Step 4: Print summary
        if benchmark_output.exists():
            with open(benchmark_output, 'r') as f:
                benchmark_data = json.load(f)
            
            scenarios = benchmark_data.get('scenarios', [])
            total_scenarios = len(scenarios)
            compliant_scenarios = sum(1 for s in scenarios if s.get('validation', {}).get('sla_compliance', False))
            
            print(f"\n{'='*60}")
            print(f"🎯 BENCHMARK EXECUTION SUMMARY")
            print(f"{'='*60}")
            print(f"📊 Scenarios executed: {total_scenarios}")
            print(f"✅ SLA compliant: {compliant_scenarios}/{total_scenarios} ({compliant_scenarios/total_scenarios*100:.0f}%)" if total_scenarios > 0 else "No scenarios executed")
            print(f"📁 Results: {benchmark_output}")
            
            if results['reports_generated']:
                print(f"📋 Reports generated:")
                for report in results['reports_generated']:
                    print(f"   - {report}")
            
            # Show scenario details
            if scenarios:
                print(f"\n📈 Scenario Details:")
                for scenario in scenarios:
                    dataset = scenario.get('dataset_info', {})
                    validation = scenario.get('validation', {})
                    
                    status_icon = '✅' if validation.get('sla_compliance', False) else '❌'
                    actual_time = validation.get('actual_time_ms', 0)
                    target_time = validation.get('target_time_ms', 0)
                    speedup = validation.get('speedup_factor', 1.0)
                    
                    print(f"   {status_icon} {dataset.get('name', 'Unknown')}: "
                          f"{actual_time:.0f}ms/{target_time:.0f}ms ({speedup:.1f}x speedup)")
            
            print(f"{'='*60}")
        
        results['status'] = 'success'
        results['end_time'] = datetime.now().isoformat()
        
    except subprocess.TimeoutExpired:
        logger.error("❌ Benchmark execution timed out")
        results['status'] = 'timeout'
        results['error'] = 'Benchmark execution exceeded 30-minute timeout'
        
    except Exception as e:
        logger.error(f"❌ Benchmark suite failed: {e}")
        results['status'] = 'error'
        results['error'] = str(e)
    
    return results

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Heavy Optimizer Parquet Pipeline Benchmark Suite',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all benchmark scenarios
  python run_benchmark.py
  
  # Run specific scenario
  python run_benchmark.py --scenario production_baseline
  
  # Run with SLA failure enforcement for CI
  python run_benchmark.py --fail-on-sla --ci-mode
  
  # Quick run without reports
  python run_benchmark.py --no-reports --scenario micro_dataset
        """
    )
    
    parser.add_argument('--scenario', type=str, 
                       help='Run specific scenario only')
    parser.add_argument('--fail-on-sla', action='store_true', 
                       help='Exit with error code if SLA violations detected')
    parser.add_argument('--no-reports', action='store_true',
                       help='Skip report generation (faster execution)')
    parser.add_argument('--output-dir', type=str,
                       help='Custom output directory for results and reports')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--ci-mode', action='store_true',
                       help='Run in CI mode (fail-on-sla + minimal output)')
    
    args = parser.parse_args()
    
    # Set CI mode defaults
    if args.ci_mode:
        args.fail_on_sla = True
    
    # Run benchmark suite
    results = run_benchmark_suite(
        scenario=args.scenario,
        fail_on_sla=args.fail_on_sla,
        generate_reports=not args.no_reports,
        verbose=args.verbose,
        output_dir=args.output_dir
    )
    
    # Exit with appropriate code
    if results['status'] == 'success':
        sys.exit(0)
    elif results['status'] == 'failed' and results.get('sla_violations'):
        logger.error("Exiting with error code due to SLA violations")
        sys.exit(2)  # Special exit code for SLA violations
    else:
        logger.error(f"Benchmark suite failed: {results.get('error', 'Unknown error')}")
        sys.exit(1)

if __name__ == "__main__":
    main()