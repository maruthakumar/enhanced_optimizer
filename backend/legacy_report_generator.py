#!/usr/bin/env python3
"""
Legacy Report Generator
Creates comprehensive comparison reports with visualizations
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import seaborn as sns

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LegacyReportGenerator:
    """
    Generate comprehensive comparison reports between legacy and new systems
    """
    
    def __init__(self, output_dir: str = "/mnt/optimizer_share/output/legacy_comparison"):
        """
        Initialize report generator
        
        Args:
            output_dir: Directory to save reports and visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "charts").mkdir(exist_ok=True)
        (self.output_dir / "raw_results").mkdir(exist_ok=True)
        (self.output_dir / "reports").mkdir(exist_ok=True)
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
    def generate_comprehensive_report(self,
                                    legacy_results: Dict[str, Any],
                                    new_results: Dict[str, Any],
                                    comparison_summary: Dict[str, Any]) -> str:
        """
        Generate comprehensive HTML report
        
        Args:
            legacy_results: Results from legacy system
            new_results: Results from new system
            comparison_summary: Comparison analysis results
            
        Returns:
            Path to generated HTML report
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.output_dir / "reports" / f"legacy_comparison_{timestamp}.html"
        
        # Generate visualizations
        chart_paths = self._create_all_visualizations(legacy_results, new_results, comparison_summary)
        
        # Create HTML report
        html_content = self._create_html_report(
            legacy_results, new_results, comparison_summary, chart_paths, timestamp
        )
        
        with open(report_path, 'w') as f:
            f.write(html_content)
            
        logger.info(f"Comprehensive report generated: {report_path}")
        return str(report_path)
        
    def _create_html_report(self,
                          legacy_results: Dict[str, Any],
                          new_results: Dict[str, Any],
                          comparison_summary: Dict[str, Any],
                          chart_paths: Dict[str, str],
                          timestamp: str) -> str:
        """Create HTML report content"""
        
        # Determine overall status
        status_color = "green" if comparison_summary.get('all_within_tolerance', False) else "red"
        status_text = "PASS" if comparison_summary.get('all_within_tolerance', False) else "FAIL"
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Legacy System Comparison Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .status-pass {{ color: green; font-weight: bold; }}
        .status-fail {{ color: red; font-weight: bold; }}
        .section {{ margin: 20px 0; }}
        .chart {{ text-align: center; margin: 20px 0; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .metric-table {{ margin: 10px 0; }}
        .summary-box {{ background-color: #f9f9f9; padding: 15px; border-radius: 5px; margin: 10px 0; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Legacy System Comparison Report</h1>
        <p><strong>Generated:</strong> {timestamp}</p>
        <p><strong>Status:</strong> <span class="status-{status_text.lower()}">{status_text}</span></p>
        <p><strong>Overall Verdict:</strong> {comparison_summary.get('verdict', 'Unknown')}</p>
    </div>
    
    <div class="section">
        <h2>Executive Summary</h2>
        <div class="summary-box">
            <p><strong>Total Comparisons:</strong> {comparison_summary.get('total_comparisons', 0)}</p>
            <p><strong>Fitness Match Rate:</strong> {comparison_summary.get('fitness_match_rate', 0):.2f}%</p>
            <p><strong>Average Difference:</strong> {comparison_summary.get('average_percentage_difference', 0):.4f}%</p>
            <p><strong>Maximum Difference:</strong> {comparison_summary.get('max_percentage_difference', 0):.4f}%</p>
        </div>
    </div>
"""
        
        # Add fitness comparison chart
        if 'fitness_comparison' in chart_paths:
            html += f"""
    <div class="section">
        <h2>Fitness Value Comparison</h2>
        <div class="chart">
            <img src="{chart_paths['fitness_comparison']}" alt="Fitness Comparison" style="max-width: 100%; height: auto;">
        </div>
    </div>
"""
        
        # Add algorithm performance chart
        if 'algorithm_performance' in chart_paths:
            html += f"""
    <div class="section">
        <h2>Algorithm Performance</h2>
        <div class="chart">
            <img src="{chart_paths['algorithm_performance']}" alt="Algorithm Performance" style="max-width: 100%; height: auto;">
        </div>
    </div>
"""
        
        # Add detailed comparison table
        html += """
    <div class="section">
        <h2>Detailed Fitness Comparisons</h2>
        <table>
            <tr>
                <th>Portfolio Size</th>
                <th>Algorithm</th>
                <th>Legacy Fitness</th>
                <th>New Fitness</th>
                <th>Difference (%)</th>
                <th>Status</th>
            </tr>
"""
        
        for result in comparison_summary.get('detailed_results', []):
            status_class = "status-pass" if result['within_tolerance'] else "status-fail"
            status_symbol = "✅" if result['within_tolerance'] else "❌"
            
            html += f"""
            <tr>
                <td>{result['portfolio_size']}</td>
                <td>{result['algorithm']}</td>
                <td>{result['legacy_fitness']:.6f}</td>
                <td>{result['new_fitness']:.6f}</td>
                <td>{result['percentage_difference']:.4f}%</td>
                <td class="{status_class}">{status_symbol}</td>
            </tr>
"""
        
        html += """
        </table>
    </div>
"""
        
        # Add performance improvement section
        html += """
    <div class="section">
        <h2>Performance Improvements</h2>
        <div class="summary-box">
            <p><strong>New System Benefits:</strong></p>
            <ul>
                <li>GPU-accelerated processing with cuDF</li>
                <li>Columnar data format (Parquet) for faster I/O</li>
                <li>Memory-efficient Arrow data structures</li>
                <li>Enhanced algorithm implementations</li>
            </ul>
        </div>
    </div>
    
    <div class="section">
        <h2>Technical Notes</h2>
        <div class="summary-box">
            <p><strong>Expected Differences:</strong></p>
            <ul>
                <li>Minor floating-point precision differences due to GPU calculations</li>
                <li>Algorithm implementation improvements may result in slightly different paths</li>
                <li>Memory allocation differences between CPU and GPU processing</li>
                <li>Tolerance threshold: ±0.01% for fitness values</li>
            </ul>
        </div>
    </div>
    
</body>
</html>
"""
        
        return html
        
    def _create_all_visualizations(self,
                                 legacy_results: Dict[str, Any],
                                 new_results: Dict[str, Any],
                                 comparison_summary: Dict[str, Any]) -> Dict[str, str]:
        """Create all visualization charts"""
        chart_paths = {}
        
        # Fitness comparison chart
        chart_paths['fitness_comparison'] = self._create_fitness_comparison_chart(comparison_summary)
        
        # Algorithm performance chart
        chart_paths['algorithm_performance'] = self._create_algorithm_performance_chart(comparison_summary)
        
        # Difference distribution chart
        chart_paths['difference_distribution'] = self._create_difference_distribution_chart(comparison_summary)
        
        return chart_paths
        
    def _create_fitness_comparison_chart(self, comparison_summary: Dict[str, Any]) -> str:
        """Create fitness comparison scatter plot"""
        detailed_results = comparison_summary.get('detailed_results', [])
        if not detailed_results:
            return ""
            
        # Extract data
        legacy_fitness = [r['legacy_fitness'] for r in detailed_results]
        new_fitness = [r['new_fitness'] for r in detailed_results]
        portfolio_sizes = [r['portfolio_size'] for r in detailed_results]
        algorithms = [r['algorithm'] for r in detailed_results]
        within_tolerance = [r['within_tolerance'] for r in detailed_results]
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Color points based on tolerance
        colors = ['green' if wt else 'red' for wt in within_tolerance]
        
        scatter = ax.scatter(legacy_fitness, new_fitness, c=colors, alpha=0.7, s=100)
        
        # Add perfect correlation line
        min_val = min(min(legacy_fitness), min(new_fitness))
        max_val = max(max(legacy_fitness), max(new_fitness))
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Perfect Match')
        
        # Annotate points with portfolio size and algorithm
        for i, (x, y, size, alg) in enumerate(zip(legacy_fitness, new_fitness, portfolio_sizes, algorithms)):
            ax.annotate(f'{size}({alg})', (x, y), xytext=(5, 5), 
                       textcoords='offset points', fontsize=8, alpha=0.7)
        
        ax.set_xlabel('Legacy System Fitness')
        ax.set_ylabel('New System Fitness')
        ax.set_title('Fitness Value Comparison: Legacy vs New System')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Save chart
        chart_path = self.output_dir / "charts" / "fitness_comparison.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(chart_path.relative_to(self.output_dir))
        
    def _create_algorithm_performance_chart(self, comparison_summary: Dict[str, Any]) -> str:
        """Create algorithm performance comparison chart"""
        detailed_results = comparison_summary.get('detailed_results', [])
        if not detailed_results:
            return ""
            
        # Group by algorithm
        algorithm_data = {}
        for result in detailed_results:
            alg = result['algorithm']
            if alg not in algorithm_data:
                algorithm_data[alg] = {'legacy': [], 'new': [], 'sizes': []}
            algorithm_data[alg]['legacy'].append(result['legacy_fitness'])
            algorithm_data[alg]['new'].append(result['new_fitness'])
            algorithm_data[alg]['sizes'].append(result['portfolio_size'])
        
        # Create grouped bar chart
        fig, ax = plt.subplots(figsize=(14, 8))
        
        algorithms = list(algorithm_data.keys())
        x = np.arange(len(algorithms))
        width = 0.35
        
        legacy_means = [np.mean(algorithm_data[alg]['legacy']) for alg in algorithms]
        new_means = [np.mean(algorithm_data[alg]['new']) for alg in algorithms]
        
        bars1 = ax.bar(x - width/2, legacy_means, width, label='Legacy System', alpha=0.7)
        bars2 = ax.bar(x + width/2, new_means, width, label='New System', alpha=0.7)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=9)
                   
        for bar in bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('Algorithm')
        ax.set_ylabel('Average Fitness')
        ax.set_title('Algorithm Performance Comparison (Average Fitness)')
        ax.set_xticks(x)
        ax.set_xticklabels(algorithms, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        chart_path = self.output_dir / "charts" / "algorithm_performance.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(chart_path.relative_to(self.output_dir))
        
    def _create_difference_distribution_chart(self, comparison_summary: Dict[str, Any]) -> str:
        """Create difference distribution histogram"""
        detailed_results = comparison_summary.get('detailed_results', [])
        if not detailed_results:
            return ""
            
        differences = [r['percentage_difference'] for r in detailed_results]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create histogram
        n, bins, patches = ax.hist(differences, bins=20, alpha=0.7, edgecolor='black')
        
        # Color bars based on tolerance
        tolerance_pct = 0.01  # 0.01% tolerance
        for i, patch in enumerate(patches):
            if bins[i] <= tolerance_pct:
                patch.set_facecolor('green')
            else:
                patch.set_facecolor('red')
        
        # Add tolerance line
        ax.axvline(tolerance_pct, color='red', linestyle='--', 
                  label=f'Tolerance Threshold ({tolerance_pct}%)')
        
        ax.set_xlabel('Percentage Difference (%)')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Fitness Differences')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        chart_path = self.output_dir / "charts" / "difference_distribution.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(chart_path.relative_to(self.output_dir))
        
    def create_summary_dashboard(self, comparison_summary: Dict[str, Any]) -> str:
        """Create a summary dashboard image"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Legacy System Comparison Dashboard', fontsize=16, fontweight='bold')
        
        detailed_results = comparison_summary.get('detailed_results', [])
        
        # 1. Pass/Fail Pie Chart
        pass_count = sum(1 for r in detailed_results if r['within_tolerance'])
        fail_count = len(detailed_results) - pass_count
        
        ax1.pie([pass_count, fail_count], labels=['Pass', 'Fail'], 
               colors=['green', 'red'], autopct='%1.1f%%')
        ax1.set_title('Fitness Validation Results')
        
        # 2. Difference by Portfolio Size
        sizes = [r['portfolio_size'] for r in detailed_results]
        diffs = [r['percentage_difference'] for r in detailed_results]
        
        ax2.scatter(sizes, diffs, alpha=0.7)
        ax2.axhline(0.01, color='red', linestyle='--', label='Tolerance')
        ax2.set_xlabel('Portfolio Size')
        ax2.set_ylabel('Difference (%)')
        ax2.set_title('Difference by Portfolio Size')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Algorithm Accuracy
        algorithm_accuracy = {}
        for result in detailed_results:
            alg = result['algorithm']
            if alg not in algorithm_accuracy:
                algorithm_accuracy[alg] = {'pass': 0, 'total': 0}
            algorithm_accuracy[alg]['total'] += 1
            if result['within_tolerance']:
                algorithm_accuracy[alg]['pass'] += 1
        
        algs = list(algorithm_accuracy.keys())
        accuracy = [algorithm_accuracy[alg]['pass'] / algorithm_accuracy[alg]['total'] * 100 
                   for alg in algs]
        
        bars = ax3.bar(algs, accuracy, color=['green' if acc == 100 else 'orange' for acc in accuracy])
        ax3.set_ylabel('Accuracy (%)')
        ax3.set_title('Algorithm Validation Accuracy')
        ax3.set_ylim(0, 105)
        
        # Add value labels
        for bar, acc in zip(bars, accuracy):
            ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                    f'{acc:.1f}%', ha='center', va='bottom')
        
        # 4. Summary Statistics
        ax4.axis('off')
        stats_text = f"""
Summary Statistics:

Total Comparisons: {comparison_summary.get('total_comparisons', 0)}
Fitness Matches: {comparison_summary.get('fitness_matches', 0)}
Match Rate: {comparison_summary.get('fitness_match_rate', 0):.2f}%

Average Difference: {comparison_summary.get('average_percentage_difference', 0):.4f}%
Maximum Difference: {comparison_summary.get('max_percentage_difference', 0):.4f}%

Status: {comparison_summary.get('verdict', 'Unknown')}
"""
        ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=12,
                verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        
        dashboard_path = self.output_dir / "charts" / "dashboard.png"
        plt.savefig(dashboard_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(dashboard_path)


def main():
    """Test the report generator"""
    # Example usage with dummy data
    generator = LegacyReportGenerator()
    
    # Create dummy comparison summary
    comparison_summary = {
        'total_comparisons': 10,
        'fitness_matches': 9,
        'fitness_match_rate': 90.0,
        'average_percentage_difference': 0.005,
        'max_percentage_difference': 0.02,
        'all_within_tolerance': False,
        'verdict': 'MOSTLY PASS - 1 test exceeded tolerance',
        'detailed_results': [
            {
                'portfolio_size': 35,
                'algorithm': 'GA',
                'legacy_fitness': 25.123,
                'new_fitness': 25.124,
                'percentage_difference': 0.004,
                'within_tolerance': True
            },
            {
                'portfolio_size': 37,
                'algorithm': 'SA', 
                'legacy_fitness': 30.458,
                'new_fitness': 30.463,
                'percentage_difference': 0.016,
                'within_tolerance': False
            }
        ]
    }
    
    # Generate dashboard
    dashboard_path = generator.create_summary_dashboard(comparison_summary)
    print(f"Dashboard created: {dashboard_path}")


if __name__ == "__main__":
    main()