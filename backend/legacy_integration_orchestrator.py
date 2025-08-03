#!/usr/bin/env python3
"""
Legacy Integration Orchestrator
Main entry point for comparing legacy and new systems
"""

import os
import sys
import json
import logging
import argparse
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add backend to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our modules
from legacy_system_wrapper import LegacySystemWrapper
from legacy_comparison import LegacyComparison
from legacy_report_generator import LegacyReportGenerator
from parquet_cudf_workflow import ParquetCuDFWorkflow

class LegacyIntegrationOrchestrator:
    """
    Orchestrates the complete legacy system comparison workflow
    """
    
    def __init__(self, 
                 legacy_base_path: str = "/mnt/optimizer_share/zone_optimization_25_06_25",
                 output_dir: str = "/mnt/optimizer_share/output/legacy_comparison"):
        """
        Initialize orchestrator
        
        Args:
            legacy_base_path: Path to legacy optimizer system
            output_dir: Directory for comparison outputs
        """
        self.legacy_wrapper = LegacySystemWrapper(legacy_base_path)
        self.comparison_engine = LegacyComparison(tolerance=0.0001)  # 0.01% tolerance
        self.report_generator = LegacyReportGenerator(output_dir)
        self.new_workflow = ParquetCuDFWorkflow()
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def run_complete_comparison(self,
                              input_csv: str,
                              portfolio_sizes: List[int],
                              timeout_minutes: int = 30) -> Dict[str, Any]:
        """
        Run complete comparison between legacy and new systems
        
        Args:
            input_csv: Path to input CSV file
            portfolio_sizes: List of portfolio sizes to test
            timeout_minutes: Timeout for legacy system execution
            
        Returns:
            Comprehensive comparison results
        """
        logger.info("🚀 Starting Legacy System Comparison")
        logger.info(f"Input: {input_csv}")
        logger.info(f"Portfolio sizes: {portfolio_sizes}")
        
        start_time = time.time()
        
        # Step 1: Execute Legacy System
        logger.info("📊 Step 1: Executing Legacy System")
        legacy_execution = self._execute_legacy_system(input_csv, portfolio_sizes, timeout_minutes)
        
        if not legacy_execution['success']:
            logger.error("❌ Legacy system execution failed")
            return {
                'status': 'FAILED',
                'error': 'Legacy system execution failed',
                'legacy_execution': legacy_execution
            }
        
        # Step 2: Parse Legacy Results
        logger.info("📋 Step 2: Parsing Legacy Results")
        legacy_results = self._parse_legacy_results(legacy_execution['output_dir'])
        
        # Step 3: Execute New System
        logger.info("⚡ Step 3: Executing New Parquet/cuDF System")
        new_results = self._execute_new_system(input_csv, portfolio_sizes)
        
        # Step 4: Compare Results
        logger.info("🔍 Step 4: Comparing Results")
        comparison_results = self._compare_systems(legacy_results, new_results, portfolio_sizes)
        
        # Step 5: Generate Reports
        logger.info("📄 Step 5: Generating Reports")
        report_paths = self._generate_reports(legacy_results, new_results, comparison_results)
        
        total_time = time.time() - start_time
        
        # Compile final results
        final_results = {
            'status': 'COMPLETED',
            'execution_time': total_time,
            'timestamp': datetime.now().isoformat(),
            'input_csv': input_csv,
            'portfolio_sizes': portfolio_sizes,
            'legacy_execution': legacy_execution,
            'legacy_results': legacy_results,
            'new_results': new_results,
            'comparison_results': comparison_results,
            'report_paths': report_paths,
            'summary': self._create_executive_summary(comparison_results)
        }
        
        # Save complete results
        results_file = self.output_dir / f"complete_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(final_results, f, indent=2, default=str)
            
        logger.info(f"✅ Comparison completed in {total_time:.2f} seconds")
        logger.info(f"📁 Results saved to: {results_file}")
        
        return final_results
        
    def _execute_legacy_system(self, input_csv: str, portfolio_sizes: List[int], timeout_minutes: int) -> Dict[str, Any]:
        """Execute legacy system and capture results"""
        try:
            return self.legacy_wrapper.execute_legacy_system(
                input_csv=input_csv,
                portfolio_sizes=portfolio_sizes,
                timeout_minutes=timeout_minutes
            )
        except Exception as e:
            logger.error(f"Legacy system execution failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
            
    def _parse_legacy_results(self, output_dir: str) -> Dict[str, Any]:
        """Parse results from legacy system output"""
        try:
            return self.legacy_wrapper.get_legacy_results(output_dir)
        except Exception as e:
            logger.error(f"Failed to parse legacy results: {e}")
            return {
                'error': str(e),
                'portfolio_results': {},
                'optimization_summary': None
            }
            
    def _execute_new_system(self, input_csv: str, portfolio_sizes: List[int]) -> Dict[str, Any]:
        """Execute new Parquet/cuDF system"""
        try:
            new_results = {}
            
            for size in portfolio_sizes:
                logger.info(f"  Running new system for portfolio size {size}")
                
                # Execute new workflow
                result = self.new_workflow.run_optimization(
                    input_csv=input_csv,
                    portfolio_size=size,
                    use_gpu=True
                )
                
                new_results[size] = {
                    'fitness': result.get('best_fitness', 0),
                    'algorithm': result.get('best_algorithm', 'Unknown'),
                    'strategies': result.get('best_portfolio', []),
                    'metrics': result.get('metrics', {}),
                    'execution_time': result.get('execution_time', 0)
                }
                
            return {
                'success': True,
                'portfolio_results': new_results,
                'total_execution_time': sum(r['execution_time'] for r in new_results.values())
            }
            
        except Exception as e:
            logger.error(f"New system execution failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'portfolio_results': {}
            }
            
    def _compare_systems(self, 
                        legacy_results: Dict[str, Any], 
                        new_results: Dict[str, Any],
                        portfolio_sizes: List[int]) -> Dict[str, Any]:
        """Compare results between systems"""
        
        comparison_results = []
        
        for size in portfolio_sizes:
            # Get results for this portfolio size
            legacy_portfolio = legacy_results.get('portfolio_results', {}).get(size)
            new_portfolio = new_results.get('portfolio_results', {}).get(size)
            
            if not legacy_portfolio or not new_portfolio:
                logger.warning(f"Missing results for portfolio size {size}")
                continue
                
            # Compare fitness values
            fitness_comparison = self.comparison_engine.compare_fitness_values(
                legacy_fitness=legacy_portfolio.get('fitness', 0),
                new_fitness=new_portfolio.get('fitness', 0),
                portfolio_size=size,
                algorithm=legacy_portfolio.get('method', 'Unknown')
            )
            
            # Compare portfolios
            portfolio_comparison = self.comparison_engine.compare_portfolios(
                legacy_strategies=legacy_portfolio.get('strategies', []),
                new_strategies=new_portfolio.get('strategies', []),
                portfolio_size=size
            )
            
            # Compare metrics
            metrics_comparison = self.comparison_engine.compare_metrics(
                legacy_metrics=legacy_portfolio.get('metrics', {}),
                new_metrics=new_portfolio.get('metrics', {}),
                portfolio_size=size
            )
            
            comparison_results.append({
                'portfolio_size': size,
                'fitness_comparison': fitness_comparison,
                'portfolio_comparison': portfolio_comparison,
                'metrics_comparison': metrics_comparison
            })
        
        # Generate overall summary
        summary = self.comparison_engine.generate_comparison_summary()
        
        return {
            'detailed_comparisons': comparison_results,
            'summary': summary
        }
        
    def _generate_reports(self,
                         legacy_results: Dict[str, Any],
                         new_results: Dict[str, Any],
                         comparison_results: Dict[str, Any]) -> Dict[str, str]:
        """Generate all reports and visualizations"""
        
        report_paths = {}
        
        try:
            # Generate comprehensive HTML report
            html_report = self.report_generator.generate_comprehensive_report(
                legacy_results=legacy_results,
                new_results=new_results,
                comparison_summary=comparison_results['summary']
            )
            report_paths['html_report'] = html_report
            
            # Generate dashboard
            dashboard = self.report_generator.create_summary_dashboard(
                comparison_summary=comparison_results['summary']
            )
            report_paths['dashboard'] = dashboard
            
            # Save comparison data
            self.comparison_engine.save_comparison_report(
                str(self.output_dir / "comparison_data.json")
            )
            report_paths['comparison_data'] = str(self.output_dir / "comparison_data.json")
            
            logger.info("📊 All reports generated successfully")
            
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            report_paths['error'] = str(e)
            
        return report_paths
        
    def _create_executive_summary(self, comparison_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create executive summary of comparison"""
        
        summary = comparison_results.get('summary', {})
        
        return {
            'overall_status': 'PASS' if summary.get('all_within_tolerance', False) else 'FAIL',
            'fitness_match_rate': summary.get('fitness_match_rate', 0),
            'total_comparisons': summary.get('total_comparisons', 0),
            'verdict': summary.get('verdict', 'Unknown'),
            'recommendation': self._get_recommendation(summary),
            'key_findings': self._extract_key_findings(comparison_results)
        }
        
    def _get_recommendation(self, summary: Dict[str, Any]) -> str:
        """Generate recommendation based on results"""
        
        match_rate = summary.get('fitness_match_rate', 0)
        
        if match_rate >= 100:
            return "✅ APPROVED: New system matches legacy perfectly. Ready for production migration."
        elif match_rate >= 95:
            return "✅ APPROVED: New system matches legacy with minor acceptable differences. Ready for production migration."
        elif match_rate >= 90:
            return "⚠️ CONDITIONAL: New system mostly matches legacy. Review differences before migration."
        elif match_rate >= 80:
            return "❌ NOT APPROVED: Significant differences found. Further investigation required."
        else:
            return "❌ NOT APPROVED: Major differences found. New system requires fixes before migration."
            
    def _extract_key_findings(self, comparison_results: Dict[str, Any]) -> List[str]:
        """Extract key findings from comparison"""
        
        findings = []
        summary = comparison_results.get('summary', {})
        
        # Fitness accuracy
        match_rate = summary.get('fitness_match_rate', 0)
        findings.append(f"Fitness accuracy: {match_rate:.1f}% of tests passed")
        
        # Average difference
        avg_diff = summary.get('average_percentage_difference', 0)
        findings.append(f"Average fitness difference: {avg_diff:.4f}%")
        
        # Algorithm performance
        detailed = comparison_results.get('detailed_comparisons', [])
        if detailed:
            algorithms = set(d['fitness_comparison']['algorithm'] for d in detailed)
            findings.append(f"Algorithms tested: {', '.join(algorithms)}")
            
        return findings


def main():
    """Main entry point for legacy comparison"""
    parser = argparse.ArgumentParser(description='Legacy System Integration Comparison')
    parser.add_argument('--input', required=True, help='Input CSV file path')
    parser.add_argument('--sizes', nargs='+', type=int, default=[35, 37, 50, 60],
                       help='Portfolio sizes to compare')
    parser.add_argument('--timeout', type=int, default=30,
                       help='Timeout for legacy system in minutes')
    parser.add_argument('--output-dir', default='/mnt/optimizer_share/output/legacy_comparison',
                       help='Output directory for results')
    parser.add_argument('--legacy-path', default='/mnt/optimizer_share/zone_optimization_25_06_25',
                       help='Path to legacy system')
    
    args = parser.parse_args()
    
    # Validate input file
    if not Path(args.input).exists():
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)
        
    # Create orchestrator
    orchestrator = LegacyIntegrationOrchestrator(
        legacy_base_path=args.legacy_path,
        output_dir=args.output_dir
    )
    
    # Run comparison
    try:
        results = orchestrator.run_complete_comparison(
            input_csv=args.input,
            portfolio_sizes=args.sizes,
            timeout_minutes=args.timeout
        )
        
        # Print summary
        print("\n" + "="*50)
        print("LEGACY SYSTEM COMPARISON SUMMARY")
        print("="*50)
        
        summary = results.get('summary', {})
        print(f"Status: {summary.get('overall_status', 'Unknown')}")
        print(f"Fitness Match Rate: {summary.get('fitness_match_rate', 0):.1f}%")
        print(f"Total Comparisons: {summary.get('total_comparisons', 0)}")
        print(f"Verdict: {summary.get('verdict', 'Unknown')}")
        print(f"\nRecommendation: {summary.get('recommendation', 'None')}")
        
        print(f"\nDetailed results saved to: {args.output_dir}")
        
        # Exit with appropriate code
        if summary.get('overall_status') == 'PASS':
            sys.exit(0)
        else:
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Comparison failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()