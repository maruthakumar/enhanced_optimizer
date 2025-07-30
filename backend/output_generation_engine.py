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
                                    strategy_columns: List[str]) -> Dict[str, str]:
        """Generate comprehensive output package"""
        logger.info("📊 Generating comprehensive optimization output...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_files = {}
        
        try:
            # 1. Generate equity curves
            equity_file = self.generate_equity_curves(optimization_results, daily_matrix, strategy_columns, timestamp)
            output_files['equity_curves'] = equity_file
            
            # 2. Generate performance report
            report_file = self.generate_performance_report(optimization_results, daily_matrix, strategy_columns, timestamp)
            output_files['performance_report'] = report_file
            
            # 3. Generate portfolio composition
            portfolio_file = self.generate_portfolio_composition(optimization_results, strategy_columns, timestamp)
            output_files['portfolio_composition'] = portfolio_file
            
            # 4. Generate algorithm comparison
            comparison_file = self.generate_algorithm_comparison(optimization_results, timestamp)
            output_files['algorithm_comparison'] = comparison_file
            
            # 5. Generate Excel summary
            excel_file = self.generate_excel_summary(optimization_results, daily_matrix, strategy_columns, timestamp)
            output_files['excel_summary'] = excel_file
            
            # 6. Generate execution summary
            summary_file = self.generate_execution_summary(optimization_results, timestamp)
            output_files['execution_summary'] = summary_file
            
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
