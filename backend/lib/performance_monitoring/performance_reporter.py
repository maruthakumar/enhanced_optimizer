"""
Performance Reporter for Heavy Optimizer Platform
Generates performance reports and visualizations
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional

# Make visualization libraries optional
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.backends.backend_pdf import PdfPages
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None
    mdates = None
    PdfPages = None

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

class PerformanceReporter:
    """Generates performance reports from collected metrics"""
    
    def __init__(self, metrics_collector=None, performance_monitor=None):
        """Initialize reporter with data sources"""
        self.metrics_collector = metrics_collector
        self.performance_monitor = performance_monitor
        self.report_data = {}
        
    def generate_text_report(self, output_file: str):
        """Generate a text-based performance report"""
        lines = []
        lines.append("=" * 80)
        lines.append("HEAVY OPTIMIZER PLATFORM - PERFORMANCE REPORT")
        lines.append("=" * 80)
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        # Get performance summary
        if self.performance_monitor:
            summary = self.performance_monitor.get_summary()
            
            # Execution Summary
            lines.append("EXECUTION SUMMARY")
            lines.append("-" * 40)
            lines.append(f"Total Execution Time: {summary['total_execution_time']:.2f} seconds")
            lines.append("")
            
            # Component Times
            lines.append("COMPONENT EXECUTION TIMES")
            lines.append("-" * 40)
            for component, time_val in summary['component_times'].items():
                lines.append(f"{component:30s}: {time_val:8.3f} seconds")
            lines.append("")
            
            # Resource Usage
            lines.append("RESOURCE USAGE SUMMARY")
            lines.append("-" * 40)
            if summary.get('memory'):
                lines.append(f"Peak Memory Usage: {summary['memory']['peak_rss_mb']:.1f} MB")
                lines.append(f"Average Memory Usage: {summary['memory']['average_rss_mb']:.1f} MB")
                if 'peak_gpu_mb' in summary['memory']:
                    lines.append(f"Peak GPU Memory: {summary['memory']['peak_gpu_mb']:.1f} MB")
            
            if summary.get('cpu'):
                lines.append(f"Peak CPU Usage: {summary['cpu']['peak_percent']:.1f}%")
                lines.append(f"Average CPU Usage: {summary['cpu']['average_percent']:.1f}%")
            
            if summary.get('gpu'):
                lines.append(f"Peak GPU Usage: {summary['gpu']['peak_percent']:.1f}%")
                lines.append(f"Average GPU Usage: {summary['gpu']['average_percent']:.1f}%")
            lines.append("")
            
            # Data Throughput
            if summary.get('data_throughput'):
                lines.append("DATA THROUGHPUT")
                lines.append("-" * 40)
                for operation, metrics in summary['data_throughput'].items():
                    lines.append(f"{operation}:")
                    lines.append(f"  Size: {metrics['size_mb']:.1f} MB")
                    lines.append(f"  Duration: {metrics['duration_s']:.3f} seconds")
                    lines.append(f"  Throughput: {metrics['throughput_mb_s']:.1f} MB/s")
                lines.append("")
        
        # Algorithm Performance
        if self.metrics_collector:
            algo_stats = self.metrics_collector.get_algorithm_statistics()
            if algo_stats:
                lines.append("ALGORITHM PERFORMANCE")
                lines.append("-" * 40)
                for algo, stats in algo_stats.items():
                    lines.append(f"\n{algo}:")
                    lines.append(f"  Total Runs: {stats['total_runs']}")
                    lines.append(f"  Success Rate: {stats['success_rate']*100:.1f}%")
                    lines.append(f"  Avg Execution Time: {stats['execution_time']['mean']:.3f}s (±{stats['execution_time']['std']:.3f}s)")
                    lines.append(f"  Avg Iterations: {stats['average_iterations']:.0f}")
                    lines.append(f"  Avg Final Fitness: {stats['average_fitness']:.6f}")
                lines.append("")
            
            # Convergence Analysis
            convergence = self.metrics_collector.get_convergence_analysis()
            if convergence:
                lines.append("CONVERGENCE ANALYSIS")
                lines.append("-" * 40)
                for algo, analysis in convergence.items():
                    lines.append(f"\n{algo}:")
                    lines.append(f"  Convergence at generation: {analysis['convergence_generation']}")
                    lines.append(f"  Final fitness: {analysis['final_fitness']:.6f}")
                    lines.append(f"  Total improvement: {analysis['fitness_improvement']:.6f}")
                    lines.append(f"  Avg improvement rate: {analysis['average_improvement_rate']*100:.2f}% per generation")
                lines.append("")
            
            # Resource Utilization
            resource_summary = self.metrics_collector.get_resource_utilization_summary()
            if resource_summary:
                lines.append("RESOURCE UTILIZATION EFFICIENCY")
                lines.append("-" * 40)
                if 'cpu' in resource_summary:
                    lines.append(f"CPU Utilization Efficiency: {resource_summary['cpu']['utilization_efficiency']*100:.1f}%")
                if 'gpu' in resource_summary:
                    lines.append(f"GPU Utilization Efficiency: {resource_summary['gpu']['utilization_efficiency']*100:.1f}%")
                if 'memory' in resource_summary:
                    lines.append(f"Memory Stability Score: {resource_summary['memory']['stability']:.2f}")
        
        lines.append("")
        lines.append("=" * 80)
        lines.append("END OF REPORT")
        lines.append("=" * 80)
        
        # Write report
        with open(output_file, 'w') as f:
            f.write('\n'.join(lines))
    
    def generate_json_report(self, output_file: str):
        """Generate a JSON performance report for API consumption"""
        report = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'platform': 'Heavy Optimizer Platform',
                'version': '1.0'
            }
        }
        
        # Add monitor summary
        if self.performance_monitor:
            report['performance_summary'] = self.performance_monitor.get_summary()
            report['detailed_metrics'] = self.performance_monitor.metrics
        
        # Add collector data
        if self.metrics_collector:
            report['algorithm_statistics'] = self.metrics_collector.get_algorithm_statistics()
            report['convergence_analysis'] = self.metrics_collector.get_convergence_analysis()
            report['resource_utilization'] = self.metrics_collector.get_resource_utilization_summary()
        
        # Write report
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
    
    def generate_performance_plots(self, output_dir: str):
        """Generate performance visualization plots"""
        if not self.performance_monitor:
            return
        
        if not MATPLOTLIB_AVAILABLE:
            print("Warning: Matplotlib not available. Skipping plot generation.")
            return
        
        # Create figure for multiple subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Heavy Optimizer Platform - Performance Metrics', fontsize=16)
        
        # Plot 1: Memory Usage Over Time
        ax1 = axes[0, 0]
        if self.performance_monitor.metrics['memory_usage']:
            timestamps = [m['timestamp'] for m in self.performance_monitor.metrics['memory_usage']]
            rss_values = [m['rss_mb'] for m in self.performance_monitor.metrics['memory_usage']]
            ax1.plot(timestamps, rss_values, 'b-', label='RSS Memory')
            
            # Add GPU memory if available
            gpu_values = [m.get('gpu_used_mb', 0) for m in self.performance_monitor.metrics['memory_usage']]
            if any(gpu_values):
                ax1.plot(timestamps, gpu_values, 'g-', label='GPU Memory')
            
            ax1.set_xlabel('Time (seconds)')
            ax1.set_ylabel('Memory (MB)')
            ax1.set_title('Memory Usage Over Time')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # Plot 2: CPU/GPU Usage Over Time
        ax2 = axes[0, 1]
        if self.performance_monitor.metrics['cpu_usage']:
            timestamps = [c['timestamp'] for c in self.performance_monitor.metrics['cpu_usage']]
            cpu_values = [c['percent'] for c in self.performance_monitor.metrics['cpu_usage']]
            ax2.plot(timestamps, cpu_values, 'r-', label='CPU Usage')
            
            # Add GPU usage if available
            if self.performance_monitor.metrics['gpu_usage']:
                gpu_timestamps = [g['timestamp'] for g in self.performance_monitor.metrics['gpu_usage']]
                gpu_values = [g['gpu_percent'] for g in self.performance_monitor.metrics['gpu_usage']]
                ax2.plot(gpu_timestamps, gpu_values, 'orange', label='GPU Usage')
            
            ax2.set_xlabel('Time (seconds)')
            ax2.set_ylabel('Usage (%)')
            ax2.set_title('CPU/GPU Utilization')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Plot 3: Algorithm Execution Times
        ax3 = axes[1, 0]
        if self.performance_monitor.metrics['component_times']:
            components = list(self.performance_monitor.metrics['component_times'].keys())
            times = list(self.performance_monitor.metrics['component_times'].values())
            
            bars = ax3.bar(range(len(components)), times)
            ax3.set_xticks(range(len(components)))
            ax3.set_xticklabels(components, rotation=45, ha='right')
            ax3.set_ylabel('Time (seconds)')
            ax3.set_title('Component Execution Times')
            ax3.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar, time_val in zip(bars, times):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{time_val:.2f}s', ha='center', va='bottom')
        
        # Plot 4: Algorithm Convergence (if available)
        ax4 = axes[1, 1]
        if self.metrics_collector and self.metrics_collector.convergence_data:
            for algo, data in self.metrics_collector.convergence_data.items():
                if data['best_fitness']:
                    ax4.plot(data['generations'], data['best_fitness'], 
                            label=f'{algo} (best)', linewidth=2)
                    if data['avg_fitness']:
                        ax4.plot(data['generations'], data['avg_fitness'], 
                                '--', label=f'{algo} (avg)', alpha=0.7)
            
            ax4.set_xlabel('Generation/Iteration')
            ax4.set_ylabel('Fitness Score')
            ax4.set_title('Algorithm Convergence')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_file = os.path.join(output_dir, 'performance_metrics.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Generate additional algorithm comparison plot if data available
        if self.metrics_collector:
            self._generate_algorithm_comparison_plot(output_dir)
    
    def _generate_algorithm_comparison_plot(self, output_dir: str):
        """Generate algorithm comparison plot"""
        if not MATPLOTLIB_AVAILABLE:
            return
            
        stats = self.metrics_collector.get_algorithm_statistics()
        if not stats:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Algorithm Performance Comparison', fontsize=16)
        
        algorithms = list(stats.keys())
        
        # Success rates
        ax1 = axes[0, 0]
        success_rates = [stats[algo]['success_rate'] * 100 for algo in algorithms]
        bars = ax1.bar(algorithms, success_rates)
        ax1.set_ylabel('Success Rate (%)')
        ax1.set_title('Algorithm Success Rates')
        ax1.set_ylim(0, 105)
        for bar, rate in zip(bars, success_rates):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                    f'{rate:.1f}%', ha='center', va='bottom')
        
        # Execution times
        ax2 = axes[0, 1]
        mean_times = [stats[algo]['execution_time']['mean'] for algo in algorithms]
        std_times = [stats[algo]['execution_time']['std'] for algo in algorithms]
        ax2.bar(algorithms, mean_times, yerr=std_times, capsize=5)
        ax2.set_ylabel('Execution Time (seconds)')
        ax2.set_title('Average Execution Times')
        
        # Average fitness
        ax3 = axes[1, 0]
        avg_fitness = [stats[algo]['average_fitness'] for algo in algorithms]
        bars = ax3.bar(algorithms, avg_fitness)
        ax3.set_ylabel('Average Fitness Score')
        ax3.set_title('Average Final Fitness')
        
        # Iterations
        ax4 = axes[1, 1]
        avg_iterations = [stats[algo]['average_iterations'] for algo in algorithms]
        ax4.bar(algorithms, avg_iterations)
        ax4.set_ylabel('Average Iterations')
        ax4.set_title('Average Iterations to Convergence')
        
        plt.tight_layout()
        plot_file = os.path.join(output_dir, 'algorithm_comparison.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_comprehensive_report(self, output_dir: str):
        """Generate all report types"""
        # Text report
        text_file = os.path.join(output_dir, 'performance_report.txt')
        self.generate_text_report(text_file)
        
        # JSON report
        json_file = os.path.join(output_dir, 'performance_metrics.json')
        self.generate_json_report(json_file)
        
        # Plots
        self.generate_performance_plots(output_dir)
        
        # Historical data export
        if self.metrics_collector:
            self.metrics_collector.export_metrics(output_dir)