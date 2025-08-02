"""
Metrics Collector for Heavy Optimizer Platform
Collects and aggregates performance metrics from various sources
"""

import time
import json
import os
from typing import Dict, List, Any, Optional
from datetime import datetime
import numpy as np

class MetricsCollector:
    """Collects and aggregates metrics from optimization runs"""
    
    def __init__(self):
        """Initialize metrics collector"""
        self.algorithm_runs = {}
        self.convergence_data = {}
        self.fitness_history = {}
        self.resource_snapshots = []
        
    def collect_algorithm_run(self, algorithm_name: str, run_data: Dict[str, Any]):
        """Collect data from an algorithm run"""
        if algorithm_name not in self.algorithm_runs:
            self.algorithm_runs[algorithm_name] = []
        
        # Extract key metrics
        run_metrics = {
            'timestamp': datetime.now().isoformat(),
            'execution_time': run_data.get('execution_time', 0),
            'iterations': run_data.get('iterations', 0),
            'final_fitness': run_data.get('final_fitness', 0),
            'portfolio_size': run_data.get('portfolio_size', 0),
            'success': run_data.get('success', True),
            'memory_peak_mb': run_data.get('memory_peak_mb', 0)
        }
        
        self.algorithm_runs[algorithm_name].append(run_metrics)
    
    def collect_convergence_data(self, algorithm_name: str, generation: int, 
                                best_fitness: float, avg_fitness: float = None):
        """Collect convergence data for algorithms"""
        if algorithm_name not in self.convergence_data:
            self.convergence_data[algorithm_name] = {
                'generations': [],
                'best_fitness': [],
                'avg_fitness': []
            }
        
        self.convergence_data[algorithm_name]['generations'].append(generation)
        self.convergence_data[algorithm_name]['best_fitness'].append(best_fitness)
        if avg_fitness is not None:
            self.convergence_data[algorithm_name]['avg_fitness'].append(avg_fitness)
    
    def collect_fitness_distribution(self, algorithm_name: str, fitness_values: List[float]):
        """Collect fitness distribution data"""
        if algorithm_name not in self.fitness_history:
            self.fitness_history[algorithm_name] = []
        
        # Store distribution statistics
        if fitness_values:
            distribution_stats = {
                'timestamp': time.time(),
                'count': len(fitness_values),
                'mean': np.mean(fitness_values),
                'std': np.std(fitness_values),
                'min': np.min(fitness_values),
                'max': np.max(fitness_values),
                'percentiles': {
                    '25': np.percentile(fitness_values, 25),
                    '50': np.percentile(fitness_values, 50),
                    '75': np.percentile(fitness_values, 75),
                    '90': np.percentile(fitness_values, 90),
                    '95': np.percentile(fitness_values, 95)
                }
            }
            self.fitness_history[algorithm_name].append(distribution_stats)
    
    def collect_resource_snapshot(self, snapshot: Dict[str, Any]):
        """Collect resource usage snapshot"""
        self.resource_snapshots.append({
            'timestamp': datetime.now().isoformat(),
            **snapshot
        })
    
    def get_algorithm_statistics(self) -> Dict[str, Any]:
        """Get statistics for all algorithms"""
        stats = {}
        
        for algo, runs in self.algorithm_runs.items():
            if not runs:
                continue
            
            # Calculate statistics
            execution_times = [r['execution_time'] for r in runs]
            success_runs = [r for r in runs if r.get('success', True)]
            
            stats[algo] = {
                'total_runs': len(runs),
                'success_rate': len(success_runs) / len(runs) if runs else 0,
                'execution_time': {
                    'mean': np.mean(execution_times),
                    'std': np.std(execution_times),
                    'min': np.min(execution_times),
                    'max': np.max(execution_times)
                },
                'average_iterations': np.mean([r['iterations'] for r in runs if r['iterations'] > 0]),
                'average_fitness': np.mean([r['final_fitness'] for r in success_runs]) if success_runs else 0
            }
        
        return stats
    
    def get_convergence_analysis(self) -> Dict[str, Any]:
        """Analyze convergence patterns"""
        analysis = {}
        
        for algo, data in self.convergence_data.items():
            if not data['best_fitness']:
                continue
            
            best_fitness = data['best_fitness']
            generations = data['generations']
            
            # Calculate convergence metrics
            improvement_rate = []
            for i in range(1, len(best_fitness)):
                if best_fitness[i-1] != 0:
                    rate = (best_fitness[i] - best_fitness[i-1]) / abs(best_fitness[i-1])
                    improvement_rate.append(rate)
            
            # Find convergence point (where improvement < 1%)
            convergence_gen = len(generations)
            for i in range(len(improvement_rate) - 5, 0, -1):
                if i + 5 <= len(improvement_rate):
                    recent_improvements = improvement_rate[i:i+5]
                    if all(abs(imp) < 0.01 for imp in recent_improvements):
                        convergence_gen = generations[i]
                        break
            
            analysis[algo] = {
                'total_generations': len(generations),
                'convergence_generation': convergence_gen,
                'final_fitness': best_fitness[-1] if best_fitness else 0,
                'fitness_improvement': best_fitness[-1] - best_fitness[0] if len(best_fitness) > 1 else 0,
                'average_improvement_rate': np.mean(improvement_rate) if improvement_rate else 0
            }
        
        return analysis
    
    def get_resource_utilization_summary(self) -> Dict[str, Any]:
        """Get resource utilization summary"""
        if not self.resource_snapshots:
            return {}
        
        # Extract metrics
        cpu_usage = [s.get('cpu_percent', 0) for s in self.resource_snapshots]
        memory_usage = [s.get('memory_mb', 0) for s in self.resource_snapshots]
        gpu_usage = [s.get('gpu_percent', 0) for s in self.resource_snapshots if 'gpu_percent' in s]
        gpu_memory = [s.get('gpu_memory_mb', 0) for s in self.resource_snapshots if 'gpu_memory_mb' in s]
        
        summary = {
            'cpu': {
                'average_percent': np.mean(cpu_usage) if cpu_usage else 0,
                'peak_percent': max(cpu_usage) if cpu_usage else 0,
                'utilization_efficiency': len([c for c in cpu_usage if c > 50]) / len(cpu_usage) if cpu_usage else 0
            },
            'memory': {
                'average_mb': np.mean(memory_usage) if memory_usage else 0,
                'peak_mb': max(memory_usage) if memory_usage else 0,
                'stability': 1 - (np.std(memory_usage) / np.mean(memory_usage)) if memory_usage and np.mean(memory_usage) > 0 else 0
            }
        }
        
        if gpu_usage:
            summary['gpu'] = {
                'average_percent': np.mean(gpu_usage),
                'peak_percent': max(gpu_usage),
                'utilization_efficiency': len([g for g in gpu_usage if g > 50]) / len(gpu_usage)
            }
        
        if gpu_memory:
            summary['gpu_memory'] = {
                'average_mb': np.mean(gpu_memory),
                'peak_mb': max(gpu_memory)
            }
        
        return summary
    
    def export_metrics(self, output_dir: str):
        """Export all collected metrics"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Export algorithm runs
        if self.algorithm_runs:
            with open(os.path.join(output_dir, f'algorithm_runs_{timestamp}.json'), 'w') as f:
                json.dump(self.algorithm_runs, f, indent=2)
        
        # Export convergence data
        if self.convergence_data:
            with open(os.path.join(output_dir, f'convergence_data_{timestamp}.json'), 'w') as f:
                json.dump(self.convergence_data, f, indent=2)
        
        # Export fitness history
        if self.fitness_history:
            with open(os.path.join(output_dir, f'fitness_history_{timestamp}.json'), 'w') as f:
                json.dump(self.fitness_history, f, indent=2)
        
        # Export resource snapshots
        if self.resource_snapshots:
            with open(os.path.join(output_dir, f'resource_snapshots_{timestamp}.json'), 'w') as f:
                json.dump(self.resource_snapshots, f, indent=2)