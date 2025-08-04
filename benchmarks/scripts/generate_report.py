#!/usr/bin/env python3
"""
Benchmark Report Generation System
Creates comprehensive HTML dashboards and performance visualizations
"""

import os
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BenchmarkReportGenerator:
    """Generates comprehensive benchmark reports with visualizations."""
    
    def __init__(self, output_dir: str = None):
        """Initialize report generator."""
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(__file__), '..', 'reports')
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Template directory
        self.template_dir = Path(__file__).parent / 'templates'
        self.template_dir.mkdir(exist_ok=True)
        
        # Create HTML template if it doesn't exist
        self._create_html_template()
    
    def _create_html_template(self):
        """Create HTML template for reports."""
        template_path = self.template_dir / 'benchmark_report.html'
        
        if not template_path.exists():
            html_template = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Parquet Pipeline Benchmark Report</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        .header h1 {
            margin: 0;
            font-size: 2.5em;
        }
        .header .subtitle {
            margin-top: 10px;
            opacity: 0.9;
            font-size: 1.1em;
        }
        .content {
            padding: 30px;
        }
        .summary-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .summary-card {
            background: #f8f9fa;
            border-radius: 8px;
            padding: 20px;
            text-align: center;
            border-left: 4px solid #667eea;
        }
        .summary-card h3 {
            margin: 0 0 10px 0;
            color: #333;
        }
        .summary-card .value {
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
        }
        .summary-card .unit {
            font-size: 0.9em;
            color: #666;
        }
        .chart-container {
            margin: 30px 0;
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .chart-title {
            font-size: 1.5em;
            font-weight: bold;
            margin-bottom: 20px;
            color: #333;
        }
        .scenario-table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .scenario-table th,
        .scenario-table td {
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        .scenario-table th {
            background: #667eea;
            color: white;
            font-weight: bold;
        }
        .scenario-table tr:nth-child(even) {
            background-color: #f8f9fa;
        }
        .status-pass {
            color: #28a745;
            font-weight: bold;
        }
        .status-fail {
            color: #dc3545;
            font-weight: bold;
        }
        .footer {
            background: #f8f9fa;
            padding: 20px;
            text-align: center;
            color: #666;
            border-top: 1px solid #ddd;
        }
        .error-section {
            background: #f8d7da;
            border: 1px solid #f5c6cb;
            border-radius: 5px;
            padding: 15px;
            margin: 20px 0;
        }
        .error-section h4 {
            color: #721c24;
            margin-top: 0;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Parquet Pipeline Benchmark Report</h1>
            <div class="subtitle">Performance Analysis and SLA Validation</div>
            <div class="subtitle">Generated: {{TIMESTAMP}}</div>
        </div>
        
        <div class="content">
            <!-- Summary Cards -->
            <div class="summary-grid">
                <div class="summary-card">
                    <h3>Scenarios Run</h3>
                    <div class="value">{{SCENARIOS_COUNT}}</div>
                </div>
                <div class="summary-card">
                    <h3>SLA Compliance</h3>
                    <div class="value status-{{SLA_STATUS}}">{{SLA_RATE}}%</div>
                </div>
                <div class="summary-card">
                    <h3>Average Speedup</h3>
                    <div class="value">{{AVG_SPEEDUP}}x</div>
                </div>
                <div class="summary-card">
                    <h3>Fastest Scenario</h3>
                    <div class="value">{{FASTEST_TIME}}</div>
                    <div class="unit">milliseconds</div>
                </div>
            </div>
            
            <!-- Performance Chart -->
            <div class="chart-container">
                <div class="chart-title">Scenario Performance vs SLA Targets</div>
                <canvas id="performanceChart" width="400" height="200"></canvas>
            </div>
            
            <!-- Speedup Chart -->
            <div class="chart-container">
                <div class="chart-title">Performance Improvement vs Legacy System</div>
                <canvas id="speedupChart" width="400" height="200"></canvas>
            </div>
            
            <!-- Detailed Results Table -->
            <div class="chart-title">Detailed Scenario Results</div>
            <table class="scenario-table">
                <thead>
                    <tr>
                        <th>Scenario</th>
                        <th>Strategy Count</th>
                        <th>Trading Days</th>
                        <th>Actual Time (ms)</th>
                        <th>Target Time (ms)</th>
                        <th>SLA Status</th>
                        <th>Speedup Factor</th>
                        <th>Memory Usage (MB)</th>
                    </tr>
                </thead>
                <tbody>
                    {{SCENARIO_ROWS}}
                </tbody>
            </table>
            
            <!-- Pipeline Breakdown -->
            <div class="chart-container">
                <div class="chart-title">Pipeline Stage Breakdown</div>
                <canvas id="pipelineChart" width="400" height="200"></canvas>
            </div>
            
            <!-- System Information -->
            <div class="chart-container">
                <div class="chart-title">System Information</div>
                <table class="scenario-table">
                    <tr><td><strong>GPU Available:</strong></td><td>{{GPU_AVAILABLE}}</td></tr>
                    <tr><td><strong>Legacy Available:</strong></td><td>{{LEGACY_AVAILABLE}}</td></tr>
                    <tr><td><strong>System Memory:</strong></td><td>{{SYSTEM_MEMORY}} GB</td></tr>
                    <tr><td><strong>Python Version:</strong></td><td>{{PYTHON_VERSION}}</td></tr>
                </table>
            </div>
            
            <!-- Errors Section -->
            {{ERROR_SECTION}}
        </div>
        
        <div class="footer">
            <p>Generated by Heavy Optimizer Platform Benchmark Framework</p>
            <p>Report timestamp: {{FULL_TIMESTAMP}}</p>
        </div>
    </div>
    
    <script>
        // Performance Chart
        const performanceCtx = document.getElementById('performanceChart').getContext('2d');
        new Chart(performanceCtx, {
            type: 'bar',
            data: {
                labels: {{SCENARIO_LABELS}},
                datasets: [
                    {
                        label: 'Actual Time (ms)',
                        data: {{ACTUAL_TIMES}},
                        backgroundColor: 'rgba(102, 126, 234, 0.8)',
                        borderColor: 'rgba(102, 126, 234, 1)',
                        borderWidth: 1
                    },
                    {
                        label: 'Target Time (ms)',
                        data: {{TARGET_TIMES}},
                        backgroundColor: 'rgba(220, 53, 69, 0.8)',
                        borderColor: 'rgba(220, 53, 69, 1)',
                        borderWidth: 1
                    }
                ]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Time (milliseconds)'
                        }
                    }
                }
            }
        });
        
        // Speedup Chart
        const speedupCtx = document.getElementById('speedupChart').getContext('2d');
        new Chart(speedupCtx, {
            type: 'bar',
            data: {
                labels: {{SCENARIO_LABELS}},
                datasets: [{
                    label: 'Speedup Factor',
                    data: {{SPEEDUP_FACTORS}},
                    backgroundColor: 'rgba(40, 167, 69, 0.8)',
                    borderColor: 'rgba(40, 167, 69, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Speedup Factor (x)'
                        }
                    }
                }
            }
        });
        
        // Pipeline Breakdown Chart
        const pipelineCtx = document.getElementById('pipelineChart').getContext('2d');
        new Chart(pipelineCtx, {
            type: 'doughnut',
            data: {
                labels: ['CSV to Parquet', 'Parquet to cuDF', 'Algorithm Processing', 'Other'],
                datasets: [{
                    data: {{PIPELINE_BREAKDOWN}},
                    backgroundColor: [
                        'rgba(102, 126, 234, 0.8)',
                        'rgba(118, 75, 162, 0.8)',
                        'rgba(40, 167, 69, 0.8)',
                        'rgba(255, 193, 7, 0.8)'
                    ]
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        position: 'right'
                    }
                }
            }
        });
    </script>
</body>
</html>'''
            
            with open(template_path, 'w') as f:
                f.write(html_template)
    
    def generate_html_report(self, results_data: Dict, output_filename: str = None) -> str:
        """Generate HTML report from benchmark results."""
        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"benchmark_report_{timestamp}.html"
        
        output_path = self.output_dir / output_filename
        
        # Load HTML template
        template_path = self.template_dir / 'benchmark_report.html'
        with open(template_path, 'r') as f:
            template = f.read()
        
        # Extract data from results
        scenarios = results_data.get('scenarios', [])
        system_info = results_data.get('benchmark_run', {}).get('system_info', {})
        
        # Calculate summary statistics
        total_scenarios = len(scenarios)
        sla_compliant = sum(1 for s in scenarios if s.get('validation', {}).get('sla_compliance', False))
        sla_rate = (sla_compliant / total_scenarios * 100) if total_scenarios > 0 else 0
        
        speedup_factors = [s.get('validation', {}).get('speedup_factor', 1.0) for s in scenarios]
        avg_speedup = sum(speedup_factors) / len(speedup_factors) if speedup_factors else 1.0
        
        actual_times = [s.get('validation', {}).get('actual_time_ms', 0) for s in scenarios]
        fastest_time = min(actual_times) if actual_times else 0
        
        # Generate scenario table rows
        scenario_rows = []
        for scenario in scenarios:
            dataset_info = scenario.get('dataset_info', {})
            validation = scenario.get('validation', {})
            memory_usage = scenario.get('memory_usage', {})
            
            sla_status = 'pass' if validation.get('sla_compliance', False) else 'fail'
            sla_symbol = '✓' if validation.get('sla_compliance', False) else '✗'
            
            row = f'''
                <tr>
                    <td>{dataset_info.get('name', 'Unknown')}</td>
                    <td>{dataset_info.get('strategy_count', 0)}</td>
                    <td>{dataset_info.get('trading_days', 0)}</td>
                    <td>{validation.get('actual_time_ms', 0):.0f}</td>
                    <td>{validation.get('target_time_ms', 0):.0f}</td>
                    <td class="status-{sla_status}">{sla_symbol}</td>
                    <td>{validation.get('speedup_factor', 1.0):.1f}x</td>
                    <td>{memory_usage.get('peak_ram_mb', 0):.1f}</td>
                </tr>
            '''
            scenario_rows.append(row)
        
        # Generate error section
        errors = []
        for scenario in scenarios:
            if scenario.get('errors'):
                errors.extend(scenario['errors'])
        
        error_section = ''
        if errors:
            error_list = '\n'.join([f'<li>{error}</li>' for error in errors])
            error_section = f'''
                <div class="error-section">
                    <h4>Errors and Warnings</h4>
                    <ul>{error_list}</ul>
                </div>
            '''
        
        # Prepare chart data
        scenario_labels = [s.get('dataset_info', {}).get('name', 'Unknown') for s in scenarios]
        target_times = [s.get('validation', {}).get('target_time_ms', 0) for s in scenarios]
        
        # Calculate pipeline breakdown (average across scenarios)
        csv_to_parquet_times = [s.get('pipeline_timings', {}).get('csv_to_parquet_ms', 0) for s in scenarios]
        parquet_to_cudf_times = [s.get('pipeline_timings', {}).get('parquet_to_cudf_ms', 0) for s in scenarios]
        total_times = [s.get('pipeline_timings', {}).get('total_pipeline_ms', 0) for s in scenarios]
        
        avg_csv_to_parquet = sum(csv_to_parquet_times) / len(csv_to_parquet_times) if csv_to_parquet_times else 0
        avg_parquet_to_cudf = sum(parquet_to_cudf_times) / len(parquet_to_cudf_times) if parquet_to_cudf_times else 0
        avg_total = sum(total_times) / len(total_times) if total_times else 0
        avg_algorithm = max(0, avg_total - avg_csv_to_parquet - avg_parquet_to_cudf)
        avg_other = max(0, avg_total - avg_csv_to_parquet - avg_parquet_to_cudf - avg_algorithm) * 0.1
        
        pipeline_breakdown = [avg_csv_to_parquet, avg_parquet_to_cudf, avg_algorithm, avg_other]
        
        # Replace template variables
        replacements = {
            '{{TIMESTAMP}}': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            '{{FULL_TIMESTAMP}}': datetime.now().isoformat(),
            '{{SCENARIOS_COUNT}}': str(total_scenarios),
            '{{SLA_STATUS}}': 'pass' if sla_rate >= 100 else 'fail',
            '{{SLA_RATE}}': f'{sla_rate:.0f}',
            '{{AVG_SPEEDUP}}': f'{avg_speedup:.1f}',
            '{{FASTEST_TIME}}': f'{fastest_time:.0f}',
            '{{SCENARIO_ROWS}}': ''.join(scenario_rows),
            '{{ERROR_SECTION}}': error_section,
            '{{GPU_AVAILABLE}}': 'Yes' if system_info.get('gpu_available', False) else 'No',
            '{{LEGACY_AVAILABLE}}': 'Yes' if system_info.get('legacy_available', False) else 'No',
            '{{SYSTEM_MEMORY}}': f"{system_info.get('memory_gb', 0):.1f}",
            '{{PYTHON_VERSION}}': system_info.get('python_version', 'Unknown'),
            '{{SCENARIO_LABELS}}': json.dumps(scenario_labels),
            '{{ACTUAL_TIMES}}': json.dumps(actual_times),
            '{{TARGET_TIMES}}': json.dumps(target_times),
            '{{SPEEDUP_FACTORS}}': json.dumps(speedup_factors),
            '{{PIPELINE_BREAKDOWN}}': json.dumps(pipeline_breakdown)
        }
        
        # Apply replacements
        html_content = template
        for placeholder, value in replacements.items():
            html_content = html_content.replace(placeholder, str(value))
        
        # Write HTML file
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"HTML report generated: {output_path}")
        return str(output_path)
    
    def generate_csv_report(self, results_data: Dict, output_filename: str = None) -> str:
        """Generate CSV report for external analysis."""
        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"benchmark_results_{timestamp}.csv"
        
        output_path = self.output_dir / output_filename
        
        # Prepare CSV data
        import csv
        
        scenarios = results_data.get('scenarios', [])
        
        with open(output_path, 'w', newline='') as csvfile:
            fieldnames = [
                'scenario_name', 'strategy_count', 'trading_days', 'file_size_mb',
                'csv_to_parquet_ms', 'parquet_to_cudf_ms', 'total_pipeline_ms',
                'actual_time_ms', 'target_time_ms', 'sla_compliance',
                'speedup_factor', 'peak_ram_mb', 'errors'
            ]
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for scenario in scenarios:
                dataset_info = scenario.get('dataset_info', {})
                pipeline_timings = scenario.get('pipeline_timings', {})
                validation = scenario.get('validation', {})
                memory_usage = scenario.get('memory_usage', {})
                
                row = {
                    'scenario_name': dataset_info.get('name', ''),
                    'strategy_count': dataset_info.get('strategy_count', 0),
                    'trading_days': dataset_info.get('trading_days', 0),
                    'file_size_mb': dataset_info.get('file_size_mb', 0),
                    'csv_to_parquet_ms': pipeline_timings.get('csv_to_parquet_ms', 0),
                    'parquet_to_cudf_ms': pipeline_timings.get('parquet_to_cudf_ms', 0),
                    'total_pipeline_ms': pipeline_timings.get('total_pipeline_ms', 0),
                    'actual_time_ms': validation.get('actual_time_ms', 0),
                    'target_time_ms': validation.get('target_time_ms', 0),
                    'sla_compliance': validation.get('sla_compliance', False),
                    'speedup_factor': validation.get('speedup_factor', 1.0),
                    'peak_ram_mb': memory_usage.get('peak_ram_mb', 0),
                    'errors': '; '.join(scenario.get('errors', []))
                }
                
                writer.writerow(row)
        
        logger.info(f"CSV report generated: {output_path}")
        return str(output_path)
    
    def generate_all_reports(self, results_file: str) -> Dict[str, str]:
        """Generate all report formats from results file."""
        with open(results_file, 'r') as f:
            results_data = json.load(f)
        
        reports = {}
        
        # Generate HTML report
        html_path = self.generate_html_report(results_data)
        reports['html'] = html_path
        
        # Generate CSV report
        csv_path = self.generate_csv_report(results_data)
        reports['csv'] = csv_path
        
        # Copy JSON results to reports directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = self.output_dir / f"benchmark_results_{timestamp}.json"
        
        import shutil
        shutil.copy2(results_file, json_path)
        reports['json'] = str(json_path)
        
        logger.info(f"All reports generated in: {self.output_dir}")
        return reports


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate benchmark reports")
    parser.add_argument('--results', '-r', required=True, help='Benchmark results JSON file')
    parser.add_argument('--output-dir', '-o', help='Output directory for reports')
    parser.add_argument('--format', choices=['html', 'csv', 'all'], default='all',
                       help='Report format to generate')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.results):
        logger.error(f"Results file not found: {args.results}")
        sys.exit(1)
    
    # Initialize report generator
    generator = BenchmarkReportGenerator(args.output_dir)
    
    try:
        if args.format == 'all':
            reports = generator.generate_all_reports(args.results)
            print("Generated reports:")
            for format_type, path in reports.items():
                print(f"  {format_type.upper()}: {path}")
        else:
            with open(args.results, 'r') as f:
                results_data = json.load(f)
            
            if args.format == 'html':
                path = generator.generate_html_report(results_data)
                print(f"HTML report: {path}")
            elif args.format == 'csv':
                path = generator.generate_csv_report(results_data)
                print(f"CSV report: {path}")
    
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()