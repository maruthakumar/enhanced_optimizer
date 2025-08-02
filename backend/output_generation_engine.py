#!/usr/bin/env python3
"""
Output Generation Engine - Critical Missing Component
Generates equity curves, performance reports, and production-ready outputs
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

class OutputGenerationEngine:
    def __init__(self, output_directory: str = "/mnt/optimizer_share/output"):
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)
        
    def generate_comprehensive_output(self, optimization_results: Dict[str, Any], 
                                    daily_matrix: np.ndarray, 
                                    strategy_columns: List[str],
                                    ulta_metrics: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        """Generate comprehensive output package with ULTA statistics"""
        logger.info("📊 Generating comprehensive optimization output...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_files = {}
        
        try:
            # 1. Generate equity curves
            equity_file = self.generate_equity_curves(optimization_results, daily_matrix, strategy_columns, timestamp)
            output_files['equity_curves'] = equity_file
            
            # 2. Generate performance report (enhanced with ULTA stats)
            report_file = self.generate_performance_report(optimization_results, daily_matrix, strategy_columns, timestamp, ulta_metrics)
            output_files['performance_report'] = report_file
            
            # 3. Generate portfolio composition (enhanced with ULTA info)
            portfolio_file = self.generate_portfolio_composition(optimization_results, strategy_columns, timestamp, ulta_metrics)
            output_files['portfolio_composition'] = portfolio_file
            
            # 4. Generate ULTA inversion report if metrics available
            if ulta_metrics:
                ulta_file = self.generate_ulta_inversion_report(ulta_metrics, timestamp)
                output_files['ulta_inversion_report'] = ulta_file
            
            # 5. Generate algorithm comparison
            comparison_file = self.generate_algorithm_comparison(optimization_results, timestamp)
            output_files['algorithm_comparison'] = comparison_file
            
            # 5. Generate Excel summary
            excel_file = self.generate_excel_summary(optimization_results, daily_matrix, strategy_columns, timestamp)
            output_files['excel_summary'] = excel_file
            
            # 6. Generate execution summary
            summary_file = self.generate_execution_summary(optimization_results, timestamp)
            output_files['execution_summary'] = summary_file
            
            # 7. Generate ULTA inversion report
            ulta_file = self.generate_ulta_report(optimization_results, timestamp)
            if ulta_file:
                output_files['ulta_inversion_report'] = ulta_file
            
            # 8. Generate zone analysis report
            zone_file = self.generate_zone_analysis_report(optimization_results, daily_matrix, strategy_columns, timestamp)
            if zone_file:
                output_files['zone_analysis_report'] = zone_file
            
            logger.info(f"✅ Comprehensive output generated: {len(output_files)} files")
            return output_files
            
        except Exception as e:
            logger.error(f"❌ Output generation failed: {e}")
            return {}
    
    def generate_equity_curves(self, optimization_results: Dict[str, Any], 
                             daily_matrix: np.ndarray, 
                             strategy_columns: List[str], 
                             timestamp: str) -> str:
        """Generate equity curves for best portfolios"""
        logger.info("📈 Generating equity curves...")
        
        try:
            plt.style.use('seaborn-v0_8')
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Portfolio Equity Curves - Algorithm Comparison', fontsize=16, fontweight='bold')
            
            # Get individual algorithm results
            individual_results = optimization_results.get('individual_results', {})
            
            # Plot equity curves for top 4 algorithms
            algorithms_to_plot = []
            for alg, result in individual_results.items():
                if result.get('status') != 'failed' and 'best_portfolio' in result:
                    algorithms_to_plot.append((alg, result))
            
            # Sort by fitness and take top 4
            algorithms_to_plot.sort(key=lambda x: x[1].get('best_fitness', 0), reverse=True)
            algorithms_to_plot = algorithms_to_plot[:4]
            
            for idx, (algorithm, result) in enumerate(algorithms_to_plot):
                row = idx // 2
                col = idx % 2
                ax = axes[row, col]
                
                portfolio_indices = result.get('best_portfolio', [])
                if portfolio_indices:
                    # Calculate portfolio returns
                    portfolio_returns = np.sum(daily_matrix[:, portfolio_indices], axis=1)
                    cumulative_returns = np.cumsum(portfolio_returns)
                    
                    # Plot equity curve
                    ax.plot(cumulative_returns, linewidth=2, label=f'{algorithm}')
                    ax.set_title(f'{algorithm}\nFitness: {result.get("best_fitness", 0):.6f}', fontweight='bold')
                    ax.set_xlabel('Trading Days')
                    ax.set_ylabel('Cumulative Returns')
                    ax.grid(True, alpha=0.3)
                    
                    # Add performance metrics
                    total_return = cumulative_returns[-1]
                    max_drawdown = np.min(cumulative_returns - np.maximum.accumulate(cumulative_returns))
                    ax.text(0.02, 0.98, f'Total Return: {total_return:.4f}\nMax Drawdown: {max_drawdown:.4f}', 
                           transform=ax.transAxes, verticalalignment='top', 
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Remove empty subplots
            for idx in range(len(algorithms_to_plot), 4):
                row = idx // 2
                col = idx % 2
                fig.delaxes(axes[row, col])
            
            plt.tight_layout()
            
            # Save equity curves
            equity_file = self.output_directory / f"equity_curves_{timestamp}.png"
            plt.savefig(equity_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"✅ Equity curves saved: {equity_file}")
            return str(equity_file)
            
        except Exception as e:
            logger.error(f"❌ Equity curve generation failed: {e}")
            return ""
    
    def generate_performance_report(self, optimization_results: Dict[str, Any], 
                                  daily_matrix: np.ndarray, 
                                  strategy_columns: List[str], 
                                  timestamp: str) -> str:
        """Generate detailed performance report"""
        logger.info("📊 Generating performance report...")
        
        try:
            report_file = self.output_directory / f"performance_report_{timestamp}.txt"
            
            with open(report_file, 'w') as f:
                f.write("=" * 80 + "\n")
                f.write("HEAVY OPTIMIZER - COMPREHENSIVE PERFORMANCE REPORT\n")
                f.write("=" * 80 + "\n\n")
                
                # Execution Summary
                f.write("EXECUTION SUMMARY\n")
                f.write("-" * 40 + "\n")
                f.write(f"Execution Time: {optimization_results.get('total_execution_time', 0):.2f} seconds\n")
                f.write(f"Algorithms Executed: {optimization_results.get('algorithms_executed', 0)}\n")
                f.write(f"Success Rate: {optimization_results.get('success_rate', 0):.1f}%\n")
                f.write(f"Best Algorithm: {optimization_results.get('best_algorithm', 'None')}\n")
                f.write(f"Best Fitness: {optimization_results.get('best_fitness', 0):.6f}\n\n")
                
                # GPU Utilization
                gpu_info = optimization_results.get('gpu_utilization', {})
                f.write("GPU UTILIZATION\n")
                f.write("-" * 40 + "\n")
                initial_gpu = gpu_info.get('initial_state', {})
                final_gpu = gpu_info.get('final_state', {})
                f.write(f"GPU Model: {final_gpu.get('gpu_name', 'Unknown')}\n")
                f.write(f"Initial Memory: {initial_gpu.get('used_memory_mb', 0)} MB\n")
                f.write(f"Final Memory: {final_gpu.get('used_memory_mb', 0)} MB\n")
                f.write(f"Memory Increase: {gpu_info.get('memory_increase', 0)} MB\n")
                f.write(f"Temperature: {final_gpu.get('temperature_celsius', 0)}°C\n\n")
                
                # Individual Algorithm Results
                f.write("INDIVIDUAL ALGORITHM RESULTS\n")
                f.write("-" * 40 + "\n")
                individual_results = optimization_results.get('individual_results', {})
                
                for algorithm, result in individual_results.items():
                    f.write(f"\n{algorithm.upper()}\n")
                    if result.get('status') == 'failed':
                        f.write(f"  Status: FAILED - {result.get('error', 'Unknown error')}\n")
                    else:
                        f.write(f"  Status: SUCCESS\n")
                        f.write(f"  Execution Time: {result.get('execution_time', 0):.3f}s\n")
                        f.write(f"  Best Fitness: {result.get('best_fitness', 0):.6f}\n")
                        f.write(f"  Portfolio Size: {len(result.get('best_portfolio', []))}\n")
                        
                        # Calculate additional metrics
                        portfolio_indices = result.get('best_portfolio', [])
                        if portfolio_indices:
                            portfolio_returns = np.sum(daily_matrix[:, portfolio_indices], axis=1)
                            total_return = np.sum(portfolio_returns)
                            volatility = np.std(portfolio_returns)
                            sharpe_ratio = np.mean(portfolio_returns) / (volatility + 1e-6)
                            
                            f.write(f"  Total Return: {total_return:.6f}\n")
                            f.write(f"  Volatility: {volatility:.6f}\n")
                            f.write(f"  Sharpe Ratio: {sharpe_ratio:.6f}\n")
                
                # Best Portfolio Analysis
                best_portfolio = optimization_results.get('best_portfolio', [])
                if best_portfolio:
                    f.write(f"\nBEST PORTFOLIO ANALYSIS\n")
                    f.write("-" * 40 + "\n")
                    f.write(f"Algorithm: {optimization_results.get('best_algorithm', 'Unknown')}\n")
                    f.write(f"Portfolio Size: {len(best_portfolio)}\n")
                    f.write(f"Selected Strategies:\n")
                    
                    for i, strategy_idx in enumerate(best_portfolio[:10]):  # Show first 10
                        if strategy_idx < len(strategy_columns):
                            strategy_name = strategy_columns[strategy_idx]
                            f.write(f"  {i+1:2d}. {strategy_name}\n")
                    
                    if len(best_portfolio) > 10:
                        f.write(f"  ... and {len(best_portfolio) - 10} more strategies\n")
            
            logger.info(f"✅ Performance report saved: {report_file}")
            return str(report_file)
            
        except Exception as e:
            logger.error(f"❌ Performance report generation failed: {e}")
            return ""
    
    def generate_portfolio_composition(self, optimization_results: Dict[str, Any], 
                                     strategy_columns: List[str], 
                                     timestamp: str) -> str:
        """Generate portfolio composition analysis"""
        logger.info("📋 Generating portfolio composition...")
        
        try:
            composition_file = self.output_directory / f"portfolio_composition_{timestamp}.csv"
            
            best_portfolio = optimization_results.get('best_portfolio', [])
            best_algorithm = optimization_results.get('best_algorithm', 'Unknown')
            
            if best_portfolio:
                # Create composition DataFrame
                composition_data = []
                for i, strategy_idx in enumerate(best_portfolio):
                    if strategy_idx < len(strategy_columns):
                        composition_data.append({
                            'Rank': i + 1,
                            'Strategy_Index': strategy_idx,
                            'Strategy_Name': strategy_columns[strategy_idx],
                            'Algorithm': best_algorithm,
                            'Weight': 1.0 / len(best_portfolio)  # Equal weight assumption
                        })
                
                df = pd.DataFrame(composition_data)
                df.to_csv(composition_file, index=False)
                
                logger.info(f"✅ Portfolio composition saved: {composition_file}")
                return str(composition_file)
            else:
                logger.warning("⚠️ No best portfolio found for composition analysis")
                return ""
            
        except Exception as e:
            logger.error(f"❌ Portfolio composition generation failed: {e}")
            return ""
    
    def generate_algorithm_comparison(self, optimization_results: Dict[str, Any], 
                                    timestamp: str) -> str:
        """Generate algorithm comparison chart"""
        logger.info("📊 Generating algorithm comparison...")
        
        try:
            individual_results = optimization_results.get('individual_results', {})
            
            # Prepare data for comparison
            algorithms = []
            execution_times = []
            fitness_scores = []
            statuses = []
            
            for algorithm, result in individual_results.items():
                algorithms.append(algorithm.replace('_', ' ').title())
                execution_times.append(result.get('execution_time', 0))
                fitness_scores.append(result.get('best_fitness', 0) if result.get('status') != 'failed' else 0)
                statuses.append('Success' if result.get('status') != 'failed' else 'Failed')
            
            # Create comparison chart
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            fig.suptitle('Algorithm Performance Comparison', fontsize=16, fontweight='bold')
            
            # Execution time comparison
            colors = ['green' if status == 'Success' else 'red' for status in statuses]
            bars1 = ax1.bar(range(len(algorithms)), execution_times, color=colors, alpha=0.7)
            ax1.set_title('Execution Time Comparison')
            ax1.set_xlabel('Algorithms')
            ax1.set_ylabel('Execution Time (seconds)')
            ax1.set_xticks(range(len(algorithms)))
            ax1.set_xticklabels(algorithms, rotation=45, ha='right')
            
            # Add value labels on bars
            for bar, time_val in zip(bars1, execution_times):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{time_val:.3f}s', ha='center', va='bottom')
            
            # Fitness score comparison
            successful_indices = [i for i, status in enumerate(statuses) if status == 'Success']
            successful_algorithms = [algorithms[i] for i in successful_indices]
            successful_fitness = [fitness_scores[i] for i in successful_indices]
            
            if successful_fitness:
                bars2 = ax2.bar(range(len(successful_algorithms)), successful_fitness, 
                               color='blue', alpha=0.7)
                ax2.set_title('Fitness Score Comparison (Successful Algorithms)')
                ax2.set_xlabel('Algorithms')
                ax2.set_ylabel('Fitness Score')
                ax2.set_xticks(range(len(successful_algorithms)))
                ax2.set_xticklabels(successful_algorithms, rotation=45, ha='right')
                
                # Add value labels on bars
                for bar, fitness_val in zip(bars2, successful_fitness):
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                            f'{fitness_val:.4f}', ha='center', va='bottom')
            
            plt.tight_layout()
            
            # Save comparison chart
            comparison_file = self.output_directory / f"algorithm_comparison_{timestamp}.png"
            plt.savefig(comparison_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"✅ Algorithm comparison saved: {comparison_file}")
            return str(comparison_file)
            
        except Exception as e:
            logger.error(f"❌ Algorithm comparison generation failed: {e}")
            return ""
    
    def generate_excel_summary(self, optimization_results: Dict[str, Any], 
                             daily_matrix: np.ndarray, 
                             strategy_columns: List[str], 
                             timestamp: str) -> str:
        """Generate Excel summary with multiple sheets"""
        logger.info("📊 Generating Excel summary...")
        
        try:
            excel_file = self.output_directory / f"optimization_summary_{timestamp}.xlsx"
            
            with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
                # Sheet 1: Execution Summary
                summary_data = {
                    'Metric': ['Total Execution Time', 'Algorithms Executed', 'Success Rate', 
                              'Best Algorithm', 'Best Fitness', 'Parallel Efficiency'],
                    'Value': [
                        f"{optimization_results.get('total_execution_time', 0):.2f} seconds",
                        optimization_results.get('algorithms_executed', 0),
                        f"{optimization_results.get('success_rate', 0):.1f}%",
                        optimization_results.get('best_algorithm', 'None'),
                        f"{optimization_results.get('best_fitness', 0):.6f}",
                        f"{optimization_results.get('parallel_efficiency', 0):.2f}"
                    ]
                }
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Execution Summary', index=False)
                
                # Sheet 2: Algorithm Results
                individual_results = optimization_results.get('individual_results', {})
                algorithm_data = []
                
                for algorithm, result in individual_results.items():
                    algorithm_data.append({
                        'Algorithm': algorithm,
                        'Status': 'Success' if result.get('status') != 'failed' else 'Failed',
                        'Execution Time (s)': result.get('execution_time', 0),
                        'Best Fitness': result.get('best_fitness', 0) if result.get('status') != 'failed' else 'N/A',
                        'Portfolio Size': len(result.get('best_portfolio', [])),
                        'Error': result.get('error', '') if result.get('status') == 'failed' else ''
                    })
                
                algorithm_df = pd.DataFrame(algorithm_data)
                algorithm_df.to_excel(writer, sheet_name='Algorithm Results', index=False)
                
                # Sheet 3: Best Portfolio
                best_portfolio = optimization_results.get('best_portfolio', [])
                if best_portfolio:
                    portfolio_data = []
                    for i, strategy_idx in enumerate(best_portfolio):
                        if strategy_idx < len(strategy_columns):
                            portfolio_data.append({
                                'Rank': i + 1,
                                'Strategy Index': strategy_idx,
                                'Strategy Name': strategy_columns[strategy_idx],
                                'Weight': 1.0 / len(best_portfolio)
                            })
                    
                    portfolio_df = pd.DataFrame(portfolio_data)
                    portfolio_df.to_excel(writer, sheet_name='Best Portfolio', index=False)
            
            logger.info(f"✅ Excel summary saved: {excel_file}")
            return str(excel_file)
            
        except Exception as e:
            logger.error(f"❌ Excel summary generation failed: {e}")
            return ""
    
    def generate_execution_summary(self, optimization_results: Dict[str, Any], 
                                 timestamp: str) -> str:
        """Generate JSON execution summary"""
        logger.info("📄 Generating execution summary...")
        
        try:
            summary_file = self.output_directory / f"execution_summary_{timestamp}.json"
            
            # Create comprehensive summary
            summary = {
                'execution_metadata': {
                    'timestamp': timestamp,
                    'execution_time': optimization_results.get('total_execution_time', 0),
                    'algorithms_executed': optimization_results.get('algorithms_executed', 0),
                    'success_rate': optimization_results.get('success_rate', 0),
                    'parallel_efficiency': optimization_results.get('parallel_efficiency', 0)
                },
                'best_result': {
                    'algorithm': optimization_results.get('best_algorithm', 'None'),
                    'fitness': optimization_results.get('best_fitness', 0),
                    'portfolio': optimization_results.get('best_portfolio', [])
                },
                'gpu_utilization': optimization_results.get('gpu_utilization', {}),
                'individual_results': optimization_results.get('individual_results', {}),
                'failed_algorithms': optimization_results.get('failed_algorithms', [])
            }
            
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            logger.info(f"✅ Execution summary saved: {summary_file}")
            return str(summary_file)
            
        except Exception as e:
            logger.error(f"❌ Execution summary generation failed: {e}")
            return ""
    
    def generate_ulta_report(self, optimization_results: Dict[str, Any], timestamp: str) -> str:
        """Generate ULTA inversion report"""
        logger.info("📄 Generating ULTA inversion report...")
        
        try:
            ulta_file = self.output_directory / f"ulta_inversion_report_{timestamp}.md"
            
            # Get ULTA results from optimization results
            ulta_results = optimization_results.get('ulta_results', {})
            inverted_strategies = ulta_results.get('inverted_strategies', {})
            
            with open(ulta_file, 'w') as f:
                f.write("# ULTA (Ultra Low Trading Algorithm) Inversion Report\n\n")
                f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"**Run ID:** {timestamp}\n\n")
                
                if not inverted_strategies:
                    f.write("## Summary\n\n")
                    f.write("No strategies were inverted during this optimization run.\n\n")
                    f.write("ULTA inversion is applied to strategies with poor performance ")
                    f.write("(negative ROI/Drawdown ratio) to potentially improve their results.\n")
                else:
                    f.write("## Summary\n\n")
                    f.write(f"**Total Strategies Inverted:** {len(inverted_strategies)}\n\n")
                    f.write("The following strategies had their returns inverted based on ULTA logic:\n\n")
                    
                    # Create table header
                    f.write("| Strategy Name | Original ROI | Inverted ROI | Original Drawdown | Inverted Drawdown | Improvement % |\n")
                    f.write("|---------------|--------------|--------------|-------------------|-------------------|---------------|\n")
                    
                    # Write each inverted strategy
                    total_improvement = 0
                    for strategy_name, metrics in inverted_strategies.items():
                        original_roi = metrics.get('original_roi', 0)
                        inverted_roi = metrics.get('inverted_roi', 0)
                        original_dd = metrics.get('original_drawdown', 0)
                        inverted_dd = metrics.get('inverted_drawdown', 0)
                        improvement = metrics.get('improvement_percentage', 0)
                        total_improvement += improvement
                        
                        f.write(f"| {strategy_name} | {original_roi:.4f} | {inverted_roi:.4f} | ")
                        f.write(f"{original_dd:.4f} | {inverted_dd:.4f} | {improvement:.2f}% |\n")
                    
                    # Write summary statistics
                    f.write(f"\n## Overall Impact\n\n")
                    f.write(f"**Average Improvement:** {total_improvement / len(inverted_strategies):.2f}%\n")
                    f.write(f"**Strategies Improved:** {sum(1 for m in inverted_strategies.values() if m.get('improvement_percentage', 0) > 0)}\n")
                    f.write(f"**Strategies Worsened:** {sum(1 for m in inverted_strategies.values() if m.get('improvement_percentage', 0) < 0)}\n")
                    
                    # Add detailed analysis
                    f.write("\n## Detailed Analysis\n\n")
                    f.write("### ULTA Logic Applied\n\n")
                    f.write("Strategies were inverted based on the following criteria:\n")
                    f.write("- ROI/Drawdown ratio below threshold (typically 0.0)\n")
                    f.write("- Negative returns on majority of trading days\n")
                    f.write("- Consistent underperformance across multiple metrics\n\n")
                    
                    f.write("### Methodology\n\n")
                    f.write("The ULTA inversion process:\n")
                    f.write("1. Identifies poorly performing strategies\n")
                    f.write("2. Inverts their daily returns (profit becomes loss, loss becomes profit)\n")
                    f.write("3. Recalculates all performance metrics\n")
                    f.write("4. Includes inverted strategies in optimization if improved\n\n")
                
                # Add configuration section
                ulta_config = ulta_results.get('config', {})
                if ulta_config:
                    f.write("## Configuration\n\n")
                    f.write(f"- **Enabled:** {ulta_config.get('enabled', True)}\n")
                    f.write(f"- **ROI Threshold:** {ulta_config.get('roi_threshold', 0.0)}\n")
                    f.write(f"- **Inversion Method:** {ulta_config.get('inversion_method', 'negative_daily_returns')}\n")
                    f.write(f"- **Min Negative Days:** {ulta_config.get('min_negative_days', 10)}\n")
                    f.write(f"- **Negative Day Percentage:** {ulta_config.get('negative_day_percentage', 0.6)}\n")
            
            logger.info(f"✅ ULTA inversion report saved: {ulta_file}")
            return str(ulta_file)
            
        except Exception as e:
            logger.error(f"❌ ULTA report generation failed: {e}")
            return ""
    
    def generate_zone_analysis_report(self, optimization_results: Dict[str, Any], 
                                    daily_matrix: np.ndarray, 
                                    strategy_columns: List[str], 
                                    timestamp: str) -> str:
        """Generate zone analysis report"""
        logger.info("📊 Generating zone analysis report...")
        
        try:
            zone_file = self.output_directory / f"zone_analysis_report_{timestamp}.md"
            
            # Get zone data from optimization results
            zone_data = optimization_results.get('zone_analysis', {})
            
            with open(zone_file, 'w') as f:
                f.write("# Zone Analysis Report\n\n")
                f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"**Run ID:** {timestamp}\n\n")
                
                if not zone_data:
                    # Generate zone analysis from best portfolio if not provided
                    f.write("## Zone Performance Analysis\n\n")
                    f.write("*Note: Zone-specific optimization data not available. ")
                    f.write("Showing aggregate performance analysis instead.*\n\n")
                    
                    best_portfolio = optimization_results.get('best_portfolio', [])
                    if best_portfolio and len(best_portfolio) > 0:
                        # Calculate zone-based metrics from portfolio
                        portfolio_returns = np.sum(daily_matrix[:, best_portfolio], axis=1)
                        
                        # Divide into zones (4 zones by default)
                        days_per_zone = len(portfolio_returns) // 4
                        zones = {
                            'Zone 1': portfolio_returns[:days_per_zone],
                            'Zone 2': portfolio_returns[days_per_zone:2*days_per_zone],
                            'Zone 3': portfolio_returns[2*days_per_zone:3*days_per_zone],
                            'Zone 4': portfolio_returns[3*days_per_zone:]
                        }
                        
                        f.write("### Zone Performance Summary\n\n")
                        f.write("| Zone | Days | Total Return | Avg Daily Return | Max Drawdown | Win Rate |\n")
                        f.write("|------|------|--------------|------------------|--------------|----------|\n")
                        
                        for zone_name, zone_returns in zones.items():
                            total_return = np.sum(zone_returns)
                            avg_return = np.mean(zone_returns)
                            max_dd = np.min(zone_returns)
                            win_rate = np.sum(zone_returns > 0) / len(zone_returns) * 100
                            
                            f.write(f"| {zone_name} | {len(zone_returns)} | {total_return:.4f} | ")
                            f.write(f"{avg_return:.6f} | {max_dd:.4f} | {win_rate:.1f}% |\n")
                else:
                    # Use provided zone data
                    f.write("## Zone Optimization Results\n\n")
                    
                    # Summary statistics
                    num_zones = len(zone_data.get('zones', {}))
                    f.write(f"**Total Zones Analyzed:** {num_zones}\n")
                    f.write(f"**Optimization Method:** {zone_data.get('method', 'Standard')}\n\n")
                    
                    # Zone performance table
                    f.write("### Zone Performance Summary\n\n")
                    f.write("| Zone | Strategies | Best Algorithm | ROI | Drawdown | ROI/DD Ratio | Win Rate |\n")
                    f.write("|------|------------|----------------|-----|----------|--------------|----------|\n")
                    
                    zones = zone_data.get('zones', {})
                    for zone_name, zone_info in zones.items():
                        strategies = zone_info.get('strategies', 0)
                        best_algo = zone_info.get('best_algorithm', 'N/A')
                        roi = zone_info.get('roi', 0)
                        drawdown = zone_info.get('drawdown', 0)
                        ratio = zone_info.get('roi_dd_ratio', 0)
                        win_rate = zone_info.get('win_rate', 0)
                        
                        f.write(f"| {zone_name} | {strategies} | {best_algo} | ")
                        f.write(f"{roi:.4f} | {drawdown:.4f} | {ratio:.4f} | {win_rate:.1f}% |\n")
                
                # Add analysis insights
                f.write("\n## Analysis Insights\n\n")
                f.write("### Zone Characteristics\n\n")
                f.write("- **Zone 1**: Early trading period, typically showing initial volatility\n")
                f.write("- **Zone 2**: Mid-early period, strategies often stabilize\n")
                f.write("- **Zone 3**: Mid-late period, mature strategy performance\n")
                f.write("- **Zone 4**: Late trading period, final performance outcomes\n\n")
                
                f.write("### Optimization Approach\n\n")
                f.write("Zone-based optimization allows for:\n")
                f.write("1. Temporal analysis of strategy performance\n")
                f.write("2. Identification of time-dependent patterns\n")
                f.write("3. Risk management across different market periods\n")
                f.write("4. Targeted strategy selection for specific time windows\n\n")
                
                # Add recommendations
                f.write("## Recommendations\n\n")
                f.write("Based on zone analysis:\n")
                f.write("- Focus on strategies performing well across all zones\n")
                f.write("- Consider zone-specific position sizing\n")
                f.write("- Monitor zone transitions for regime changes\n")
                f.write("- Implement zone-aware risk management\n")
            
            logger.info(f"✅ Zone analysis report saved: {zone_file}")
            return str(zone_file)
            
        except Exception as e:
            logger.error(f"❌ Zone analysis report generation failed: {e}")
            return ""


def main():
    """Test the output generation engine"""
    # Mock optimization results for testing
    mock_results = {
        'total_execution_time': 15.5,
        'algorithms_executed': 7,
        'success_rate': 100.0,
        'best_algorithm': 'simulated_annealing',
        'best_fitness': 0.596,
        'best_portfolio': [1, 5, 10, 15, 20, 25, 30, 35, 40, 45],
        'parallel_efficiency': 0.85,
        'individual_results': {
            'genetic_algorithm': {'status': 'success', 'execution_time': 4.054, 'best_fitness': 0.379, 'best_portfolio': [1, 2, 3]},
            'simulated_annealing': {'status': 'success', 'execution_time': 0.082, 'best_fitness': 0.596, 'best_portfolio': [1, 5, 10]}
        },
        'gpu_utilization': {
            'initial_state': {'used_memory_mb': 471, 'gpu_name': 'NVIDIA A100-SXM4-40GB'},
            'final_state': {'used_memory_mb': 1200, 'gpu_name': 'NVIDIA A100-SXM4-40GB', 'temperature_celsius': 39}
        },
        'ulta_results': {
            'inverted_strategies': {
                'Strategy_15': {
                    'original_roi': -0.025,
                    'inverted_roi': 0.025,
                    'original_drawdown': -0.15,
                    'inverted_drawdown': -0.10,
                    'improvement_percentage': 150.0
                },
                'Strategy_42': {
                    'original_roi': -0.018,
                    'inverted_roi': 0.018,
                    'original_drawdown': -0.08,
                    'inverted_drawdown': -0.06,
                    'improvement_percentage': 125.0
                }
            },
            'config': {
                'enabled': True,
                'roi_threshold': 0.0,
                'inversion_method': 'negative_daily_returns'
            }
        },
        'zone_analysis': {
            'method': 'Temporal Zone Optimization',
            'zones': {
                'Zone 1': {
                    'strategies': 25,
                    'best_algorithm': 'genetic_algorithm',
                    'roi': 0.125,
                    'drawdown': -0.082,
                    'roi_dd_ratio': 1.524,
                    'win_rate': 62.5
                },
                'Zone 2': {
                    'strategies': 25,
                    'best_algorithm': 'simulated_annealing',
                    'roi': 0.215,
                    'drawdown': -0.095,
                    'roi_dd_ratio': 2.263,
                    'win_rate': 68.2
                }
            }
        }
    }
    
    # Mock data
    daily_matrix = np.random.randn(79, 100)
    strategy_columns = [f'Strategy_{i}' for i in range(100)]
    
    # Test output generation
    engine = OutputGenerationEngine()
    output_files = engine.generate_comprehensive_output(mock_results, daily_matrix, strategy_columns)
    
    print("✅ Output generation test completed")
    for output_type, file_path in output_files.items():
        print(f"   {output_type}: {file_path}")

if __name__ == "__main__":
    main()
