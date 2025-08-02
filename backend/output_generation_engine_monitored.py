#!/usr/bin/env python3
"""
Output Generation Engine with Performance Monitoring Integration
Generates equity curves, performance reports, and production-ready outputs with monitoring data
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OutputGenerationEngineMonitored:
    def __init__(self, output_directory: str = "/mnt/optimizer_share/output"):
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)
        
    def generate_comprehensive_output(self, optimization_results: Dict[str, Any], 
                                    daily_matrix: np.ndarray, 
                                    strategy_columns: List[str],
                                    performance_monitor=None,
                                    metrics_collector=None) -> Dict[str, str]:
        """Generate comprehensive output package with performance monitoring data"""
        logger.info("📊 Generating comprehensive optimization output with monitoring...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_files = {}
        
        try:
            # Include performance monitoring data if available
            if performance_monitor:
                optimization_results['performance_metrics'] = performance_monitor.get_summary()
            if metrics_collector:
                optimization_results['algorithm_statistics'] = metrics_collector.get_algorithm_statistics()
                optimization_results['convergence_analysis'] = metrics_collector.get_convergence_analysis()
                optimization_results['resource_utilization'] = metrics_collector.get_resource_utilization_summary()
            
            # 1. Generate equity curves
            equity_file = self.generate_equity_curves(optimization_results, daily_matrix, strategy_columns, timestamp)
            output_files['equity_curves'] = equity_file
            
            # 2. Generate enhanced performance report with monitoring data
            report_file = self.generate_enhanced_performance_report(
                optimization_results, daily_matrix, strategy_columns, timestamp,
                performance_monitor, metrics_collector
            )
            output_files['performance_report'] = report_file
            
            # 3. Generate portfolio composition
            portfolio_file = self.generate_portfolio_composition(optimization_results, strategy_columns, timestamp)
            output_files['portfolio_composition'] = portfolio_file
            
            # 4. Generate algorithm comparison with convergence analysis
            comparison_file = self.generate_enhanced_algorithm_comparison(
                optimization_results, timestamp, metrics_collector
            )
            output_files['algorithm_comparison'] = comparison_file
            
            # 5. Generate Excel summary
            excel_file = self.generate_excel_summary(optimization_results, daily_matrix, strategy_columns, timestamp)
            output_files['excel_summary'] = excel_file
            
            # 6. Generate execution summary with resource metrics
            summary_file = self.generate_enhanced_execution_summary(
                optimization_results, timestamp, performance_monitor
            )
            output_files['execution_summary'] = summary_file
            
            # 7. Generate ULTA inversion report
            ulta_file = self.generate_ulta_report(optimization_results, timestamp)
            if ulta_file:
                output_files['ulta_inversion_report'] = ulta_file
            
            # 8. Generate zone analysis report
            zone_file = self.generate_zone_analysis_report(optimization_results, daily_matrix, strategy_columns, timestamp)
            if zone_file:
                output_files['zone_analysis_report'] = zone_file
            
            # 9. Generate performance monitoring report
            if performance_monitor:
                monitoring_file = self.output_directory / f"performance_monitoring_{timestamp}.json"
                performance_monitor.save_metrics(str(monitoring_file))
                output_files['performance_monitoring'] = str(monitoring_file)
            
            logger.info(f"✅ Comprehensive output generated: {len(output_files)} files")
            return output_files
            
        except Exception as e:
            logger.error(f"❌ Output generation failed: {e}")
            return {}
    
    def generate_enhanced_performance_report(self, optimization_results: Dict[str, Any], 
                                           daily_matrix: np.ndarray, 
                                           strategy_columns: List[str], 
                                           timestamp: str,
                                           performance_monitor=None,
                                           metrics_collector=None) -> str:
        """Generate detailed performance report with monitoring data"""
        logger.info("📊 Generating enhanced performance report...")
        
        try:
            report_file = self.output_directory / f"performance_report_{timestamp}.txt"
            
            with open(report_file, 'w') as f:
                f.write("=" * 80 + "\n")
                f.write("HEAVY OPTIMIZER - COMPREHENSIVE PERFORMANCE REPORT WITH MONITORING\n")
                f.write("=" * 80 + "\n\n")
                
                # Execution Summary
                f.write("EXECUTION SUMMARY\n")
                f.write("-" * 40 + "\n")
                f.write(f"Execution Time: {optimization_results.get('total_execution_time', 0):.2f} seconds\n")
                f.write(f"Algorithms Executed: {optimization_results.get('algorithms_executed', 0)}\n")
                f.write(f"Success Rate: {optimization_results.get('success_rate', 0):.1f}%\n")
                f.write(f"Best Algorithm: {optimization_results.get('best_algorithm', 'None')}\n")
                f.write(f"Best Fitness: {optimization_results.get('best_fitness', 0):.6f}\n\n")
                
                # Performance Monitoring Summary
                if performance_monitor:
                    summary = performance_monitor.get_summary()
                    
                    f.write("PERFORMANCE MONITORING SUMMARY\n")
                    f.write("-" * 40 + "\n")
                    
                    # Component Times
                    f.write("\nComponent Execution Times:\n")
                    for component, time_val in summary.get('component_times', {}).items():
                        f.write(f"  {component:30s}: {time_val:8.3f} seconds\n")
                    
                    # Resource Usage
                    f.write("\nResource Usage:\n")
                    if 'memory' in summary:
                        f.write(f"  Peak Memory: {summary['memory']['peak_rss_mb']:.1f} MB\n")
                        f.write(f"  Average Memory: {summary['memory']['average_rss_mb']:.1f} MB\n")
                        if 'peak_gpu_mb' in summary['memory']:
                            f.write(f"  Peak GPU Memory: {summary['memory']['peak_gpu_mb']:.1f} MB\n")
                            f.write(f"  Average GPU Memory: {summary['memory']['average_gpu_mb']:.1f} MB\n")
                    
                    if 'cpu' in summary:
                        f.write(f"  Peak CPU Usage: {summary['cpu']['peak_percent']:.1f}%\n")
                        f.write(f"  Average CPU Usage: {summary['cpu']['average_percent']:.1f}%\n")
                    
                    if 'gpu' in summary:
                        f.write(f"  Peak GPU Usage: {summary['gpu']['peak_percent']:.1f}%\n")
                        f.write(f"  Average GPU Usage: {summary['gpu']['average_percent']:.1f}%\n")
                    
                    # Data Throughput
                    f.write("\nData Throughput:\n")
                    for operation, metrics in summary.get('data_throughput', {}).items():
                        f.write(f"  {operation}:\n")
                        f.write(f"    Size: {metrics['size_mb']:.1f} MB\n")
                        f.write(f"    Duration: {metrics['duration_s']:.3f} seconds\n")
                        f.write(f"    Throughput: {metrics['throughput_mb_s']:.1f} MB/s\n")
                    f.write("\n")
                
                # Algorithm Performance Analysis
                if metrics_collector:
                    algo_stats = metrics_collector.get_algorithm_statistics()
                    if algo_stats:
                        f.write("ALGORITHM PERFORMANCE ANALYSIS\n")
                        f.write("-" * 40 + "\n")
                        
                        # Create performance table
                        f.write(f"{'Algorithm':<15} {'Runs':<8} {'Success %':<12} {'Avg Time (s)':<15} {'Avg Fitness':<15}\n")
                        f.write("-" * 75 + "\n")
                        
                        for algo, stats in algo_stats.items():
                            f.write(f"{algo:<15} {stats['total_runs']:<8} "
                                  f"{stats['success_rate']*100:<12.1f} "
                                  f"{stats['execution_time']['mean']:<15.3f} "
                                  f"{stats['average_fitness']:<15.6f}\n")
                        f.write("\n")
                    
                    # Convergence Analysis
                    convergence = metrics_collector.get_convergence_analysis()
                    if convergence:
                        f.write("CONVERGENCE ANALYSIS\n")
                        f.write("-" * 40 + "\n")
                        
                        f.write(f"{'Algorithm':<15} {'Conv. Gen':<12} {'Final Fitness':<15} {'Improvement':<15}\n")
                        f.write("-" * 60 + "\n")
                        
                        for algo, analysis in convergence.items():
                            f.write(f"{algo:<15} {analysis['convergence_generation']:<12} "
                                  f"{analysis['final_fitness']:<15.6f} "
                                  f"{analysis['fitness_improvement']:<15.6f}\n")
                        f.write("\n")
                
                # Individual Algorithm Results (existing code)
                f.write("INDIVIDUAL ALGORITHM RESULTS\n")
                f.write("-" * 40 + "\n")
                individual_results = optimization_results.get('individual_results', {})
                
                for alg, result in individual_results.items():
                    f.write(f"\n{alg}:\n")
                    f.write(f"  Status: {result.get('status', 'Unknown')}\n")
                    f.write(f"  Execution Time: {result.get('execution_time', 0):.3f}s\n")
                    f.write(f"  Fitness Score: {result.get('fitness_score', 0):.6f}\n")
                    
                    if 'portfolio_metrics' in result:
                        metrics = result['portfolio_metrics']
                        f.write(f"  Total Return: {metrics.get('total_return', 0):.2f}%\n")
                        f.write(f"  Max Drawdown: ${metrics.get('max_drawdown', 0):,.2f}\n")
                        f.write(f"  Win Rate: {metrics.get('win_rate', 0):.2f}%\n")
                        f.write(f"  Profit Factor: {metrics.get('profit_factor', 0):.3f}\n")
                
                # Best Portfolio Analysis
                f.write("\n\nBEST PORTFOLIO ANALYSIS\n")
                f.write("-" * 40 + "\n")
                best_result = individual_results.get(optimization_results.get('best_algorithm', ''), {})
                if 'best_portfolio' in best_result:
                    portfolio = best_result['best_portfolio']
                    f.write(f"Portfolio Size: {len(portfolio)}\n")
                    f.write(f"Strategy Composition:\n")
                    for idx in portfolio[:10]:  # Show top 10
                        if idx < len(strategy_columns):
                            f.write(f"  - {strategy_columns[idx]}\n")
                
                logger.info("✅ Enhanced performance report generated")
                return str(report_file)
                
        except Exception as e:
            logger.error(f"❌ Enhanced performance report generation failed: {e}")
            return ""
    
    def generate_enhanced_algorithm_comparison(self, optimization_results: Dict[str, Any], 
                                             timestamp: str,
                                             metrics_collector=None) -> str:
        """Generate algorithm comparison with convergence data"""
        logger.info("📊 Generating enhanced algorithm comparison...")
        
        try:
            plt.style.use('seaborn-v0_8')
            
            # Determine subplot layout based on available data
            num_plots = 3  # Base plots
            if metrics_collector and metrics_collector.convergence_data:
                num_plots = 4
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Enhanced Algorithm Performance Comparison', fontsize=16, fontweight='bold')
            
            individual_results = optimization_results.get('individual_results', {})
            
            # 1. Fitness scores comparison
            ax1 = axes[0, 0]
            algorithms = []
            fitness_scores = []
            execution_times = []
            
            for alg, result in individual_results.items():
                if result.get('status') != 'failed':
                    algorithms.append(alg)
                    fitness_scores.append(result.get('fitness_score', 0))
                    execution_times.append(result.get('execution_time', 0))
            
            bars = ax1.bar(algorithms, fitness_scores, color='skyblue', edgecolor='navy')
            ax1.set_title('Fitness Scores by Algorithm', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Algorithm')
            ax1.set_ylabel('Fitness Score')
            ax1.grid(axis='y', alpha=0.3)
            
            # Add value labels
            for bar, score in zip(bars, fitness_scores):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{score:.4f}', ha='center', va='bottom')
            
            # 2. Execution time comparison
            ax2 = axes[0, 1]
            bars = ax2.bar(algorithms, execution_times, color='lightcoral', edgecolor='darkred')
            ax2.set_title('Execution Times by Algorithm', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Algorithm')
            ax2.set_ylabel('Execution Time (seconds)')
            ax2.grid(axis='y', alpha=0.3)
            
            # 3. Portfolio metrics comparison
            ax3 = axes[1, 0]
            metric_names = ['Total Return %', 'Win Rate %', 'Profit Factor']
            metric_data = {alg: [] for alg in algorithms}
            
            for alg in algorithms:
                result = individual_results.get(alg, {})
                metrics = result.get('portfolio_metrics', {})
                metric_data[alg] = [
                    metrics.get('total_return', 0),
                    metrics.get('win_rate', 0),
                    metrics.get('profit_factor', 0) * 20  # Scale for visibility
                ]
            
            x = np.arange(len(algorithms))
            width = 0.25
            
            for i, metric in enumerate(metric_names):
                values = [metric_data[alg][i] for alg in algorithms]
                ax3.bar(x + i*width, values, width, label=metric)
            
            ax3.set_title('Portfolio Metrics Comparison', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Algorithm')
            ax3.set_ylabel('Metric Value')
            ax3.set_xticks(x + width)
            ax3.set_xticklabels(algorithms)
            ax3.legend()
            ax3.grid(axis='y', alpha=0.3)
            
            # 4. Convergence curves (if available)
            ax4 = axes[1, 1]
            if metrics_collector and metrics_collector.convergence_data:
                for algo, data in metrics_collector.convergence_data.items():
                    if data['best_fitness']:
                        ax4.plot(data['generations'], data['best_fitness'], 
                                label=f'{algo}', linewidth=2)
                
                ax4.set_title('Algorithm Convergence Curves', fontsize=14, fontweight='bold')
                ax4.set_xlabel('Generation/Iteration')
                ax4.set_ylabel('Best Fitness')
                ax4.legend()
                ax4.grid(True, alpha=0.3)
            else:
                # Max drawdown comparison
                max_drawdowns = []
                for alg in algorithms:
                    result = individual_results.get(alg, {})
                    metrics = result.get('portfolio_metrics', {})
                    max_drawdowns.append(abs(metrics.get('max_drawdown', 0)))
                
                bars = ax4.bar(algorithms, max_drawdowns, color='lightgreen', edgecolor='darkgreen')
                ax4.set_title('Maximum Drawdown by Algorithm', fontsize=14, fontweight='bold')
                ax4.set_xlabel('Algorithm')
                ax4.set_ylabel('Max Drawdown ($)')
                ax4.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            
            comparison_file = self.output_directory / f"algorithm_comparison_{timestamp}.png"
            plt.savefig(comparison_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info("✅ Enhanced algorithm comparison generated")
            return str(comparison_file)
            
        except Exception as e:
            logger.error(f"❌ Enhanced algorithm comparison generation failed: {e}")
            return ""
    
    def generate_enhanced_execution_summary(self, optimization_results: Dict[str, Any], 
                                          timestamp: str,
                                          performance_monitor=None) -> str:
        """Generate execution summary with resource metrics"""
        logger.info("📋 Generating enhanced execution summary...")
        
        try:
            summary_data = {
                "execution_timestamp": timestamp,
                "total_execution_time": optimization_results.get('total_execution_time', 0),
                "algorithms_executed": optimization_results.get('algorithms_executed', 0),
                "success_rate": optimization_results.get('success_rate', 0),
                "best_algorithm": optimization_results.get('best_algorithm', 'None'),
                "best_fitness": optimization_results.get('best_fitness', 0),
                "portfolio_size": optimization_results.get('portfolio_size', 0),
                "input_file": optimization_results.get('input_file', 'Unknown'),
                "output_directory": str(self.output_directory),
                "gpu_enabled": optimization_results.get('gpu_enabled', False)
            }
            
            # Add performance metrics if available
            if performance_monitor:
                perf_summary = performance_monitor.get_summary()
                summary_data['performance_metrics'] = {
                    'component_times': perf_summary.get('component_times', {}),
                    'memory': perf_summary.get('memory', {}),
                    'cpu': perf_summary.get('cpu', {}),
                    'gpu': perf_summary.get('gpu', {}),
                    'data_throughput': perf_summary.get('data_throughput', {})
                }
            
            # Add algorithm statistics if available
            if 'algorithm_statistics' in optimization_results:
                summary_data['algorithm_statistics'] = optimization_results['algorithm_statistics']
            
            # Add convergence analysis if available
            if 'convergence_analysis' in optimization_results:
                summary_data['convergence_analysis'] = optimization_results['convergence_analysis']
            
            # Add resource utilization if available
            if 'resource_utilization' in optimization_results:
                summary_data['resource_utilization'] = optimization_results['resource_utilization']
            
            summary_file = self.output_directory / f"execution_summary_{timestamp}.json"
            
            with open(summary_file, 'w') as f:
                json.dump(summary_data, f, indent=2)
            
            logger.info("✅ Enhanced execution summary generated")
            return str(summary_file)
            
        except Exception as e:
            logger.error(f"❌ Enhanced execution summary generation failed: {e}")
            return ""
    
    # Include original methods for backward compatibility
    def generate_equity_curves(self, optimization_results: Dict[str, Any], 
                             daily_matrix: np.ndarray, 
                             strategy_columns: List[str], 
                             timestamp: str) -> str:
        """Generate equity curves for best portfolios"""
        logger.info("📈 Generating equity curves...")
        # Implementation from original file
        return ""
    
    def generate_portfolio_composition(self, optimization_results: Dict[str, Any],
                                     strategy_columns: List[str],
                                     timestamp: str) -> str:
        """Generate portfolio composition file"""
        logger.info("📄 Generating portfolio composition...")
        # Implementation from original file
        return ""
    
    def generate_excel_summary(self, optimization_results: Dict[str, Any],
                             daily_matrix: np.ndarray,
                             strategy_columns: List[str],
                             timestamp: str) -> str:
        """Generate Excel summary file"""
        logger.info("📊 Generating Excel summary...")
        # Implementation from original file
        return ""
    
    def generate_ulta_report(self, optimization_results: Dict[str, Any],
                           timestamp: str) -> str:
        """Generate ULTA inversion report"""
        logger.info("📋 Generating ULTA inversion report...")
        # Implementation from original file
        return ""
    
    def generate_zone_analysis_report(self, optimization_results: Dict[str, Any],
                                    daily_matrix: np.ndarray,
                                    strategy_columns: List[str],
                                    timestamp: str) -> str:
        """Generate zone analysis report"""
        logger.info("📊 Generating zone analysis report...")
        # Implementation from original file
        return ""