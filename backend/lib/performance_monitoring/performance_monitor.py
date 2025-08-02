"""
Performance Monitor for Heavy Optimizer Platform
Tracks execution times, memory usage, CPU/GPU utilization
"""

import time
import psutil
import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
import threading
import queue

class PerformanceMonitor:
    """Main performance monitoring class"""
    
    def __init__(self, output_dir: str = None):
        """Initialize performance monitor"""
        self.start_time = time.time()
        self.metrics = {
            'execution_times': {},
            'memory_usage': [],
            'cpu_usage': [],
            'gpu_usage': [],
            'data_throughput': {},
            'algorithm_metrics': {},
            'component_times': {}
        }
        self.active_timers = {}
        self.output_dir = output_dir
        
        # For continuous monitoring
        self.monitoring_active = False
        self.monitoring_thread = None
        self.metrics_queue = queue.Queue()
        
        # Try to import GPU monitoring (optional)
        self.gpu_available = False
        try:
            import pynvml
            pynvml.nvmlInit()
            self.gpu_available = True
            self.pynvml = pynvml
        except:
            pass
    
    def start_component_timer(self, component_name: str):
        """Start timing a component"""
        self.active_timers[component_name] = time.time()
    
    def stop_component_timer(self, component_name: str) -> float:
        """Stop timing a component and record the time"""
        if component_name not in self.active_timers:
            return 0.0
        
        elapsed = time.time() - self.active_timers[component_name]
        self.metrics['component_times'][component_name] = elapsed
        del self.active_timers[component_name]
        return elapsed
    
    def record_execution_time(self, operation: str, duration: float):
        """Record execution time for an operation"""
        if operation not in self.metrics['execution_times']:
            self.metrics['execution_times'][operation] = []
        self.metrics['execution_times'][operation].append(duration)
    
    def record_memory_usage(self):
        """Record current memory usage"""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        memory_data = {
            'timestamp': time.time() - self.start_time,
            'rss_mb': memory_info.rss / 1024 / 1024,
            'vms_mb': memory_info.vms / 1024 / 1024,
            'percent': process.memory_percent()
        }
        
        # Get GPU memory if available
        if self.gpu_available:
            try:
                handle = self.pynvml.nvmlDeviceGetHandleByIndex(0)
                mem_info = self.pynvml.nvmlDeviceGetMemoryInfo(handle)
                memory_data['gpu_used_mb'] = mem_info.used / 1024 / 1024
                memory_data['gpu_total_mb'] = mem_info.total / 1024 / 1024
                memory_data['gpu_percent'] = (mem_info.used / mem_info.total) * 100
            except:
                pass
        
        self.metrics['memory_usage'].append(memory_data)
        return memory_data
    
    def record_cpu_usage(self):
        """Record current CPU usage"""
        cpu_data = {
            'timestamp': time.time() - self.start_time,
            'percent': psutil.cpu_percent(interval=0.1),
            'per_cpu': psutil.cpu_percent(interval=0.1, percpu=True)
        }
        self.metrics['cpu_usage'].append(cpu_data)
        return cpu_data
    
    def record_gpu_usage(self):
        """Record GPU usage if available"""
        if not self.gpu_available:
            return None
        
        try:
            handle = self.pynvml.nvmlDeviceGetHandleByIndex(0)
            util = self.pynvml.nvmlDeviceGetUtilizationRates(handle)
            
            gpu_data = {
                'timestamp': time.time() - self.start_time,
                'gpu_percent': util.gpu,
                'memory_percent': util.memory
            }
            self.metrics['gpu_usage'].append(gpu_data)
            return gpu_data
        except:
            return None
    
    def record_data_throughput(self, data_size_mb: float, duration: float, operation: str):
        """Record data throughput for an operation"""
        throughput = data_size_mb / duration if duration > 0 else 0
        self.metrics['data_throughput'][operation] = {
            'size_mb': data_size_mb,
            'duration_s': duration,
            'throughput_mb_s': throughput
        }
    
    def record_algorithm_metric(self, algorithm: str, metric_name: str, value: Any):
        """Record algorithm-specific metrics"""
        if algorithm not in self.metrics['algorithm_metrics']:
            self.metrics['algorithm_metrics'][algorithm] = {}
        
        if metric_name not in self.metrics['algorithm_metrics'][algorithm]:
            self.metrics['algorithm_metrics'][algorithm][metric_name] = []
        
        self.metrics['algorithm_metrics'][algorithm][metric_name].append({
            'timestamp': time.time() - self.start_time,
            'value': value
        })
    
    def start_continuous_monitoring(self, interval: float = 1.0):
        """Start continuous resource monitoring in background"""
        self.monitoring_active = True
        
        def monitor_loop():
            while self.monitoring_active:
                # Collect metrics
                metrics_snapshot = {
                    'timestamp': datetime.now().isoformat(),
                    'elapsed_time': time.time() - self.start_time,
                    'memory': self.record_memory_usage(),
                    'cpu': self.record_cpu_usage(),
                    'gpu': self.record_gpu_usage()
                }
                
                # Put in queue for processing
                self.metrics_queue.put(metrics_snapshot)
                
                time.sleep(interval)
        
        self.monitoring_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitoring_thread.start()
    
    def stop_continuous_monitoring(self):
        """Stop continuous monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=2.0)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        total_time = time.time() - self.start_time
        
        # Calculate averages and peaks
        summary = {
            'total_execution_time': total_time,
            'component_times': self.metrics['component_times'],
            'execution_times': {},
            'memory': {},
            'cpu': {},
            'gpu': {},
            'data_throughput': self.metrics['data_throughput'],
            'algorithm_summary': {}
        }
        
        # Execution time averages
        for op, times in self.metrics['execution_times'].items():
            if times:
                summary['execution_times'][op] = {
                    'count': len(times),
                    'total': sum(times),
                    'average': sum(times) / len(times),
                    'min': min(times),
                    'max': max(times)
                }
        
        # Memory statistics
        if self.metrics['memory_usage']:
            rss_values = [m['rss_mb'] for m in self.metrics['memory_usage']]
            summary['memory'] = {
                'peak_rss_mb': max(rss_values),
                'average_rss_mb': sum(rss_values) / len(rss_values),
                'min_rss_mb': min(rss_values)
            }
            
            # GPU memory if available
            gpu_mem_values = [m.get('gpu_used_mb', 0) for m in self.metrics['memory_usage'] if 'gpu_used_mb' in m]
            if gpu_mem_values:
                summary['memory']['peak_gpu_mb'] = max(gpu_mem_values)
                summary['memory']['average_gpu_mb'] = sum(gpu_mem_values) / len(gpu_mem_values)
        
        # CPU statistics
        if self.metrics['cpu_usage']:
            cpu_values = [c['percent'] for c in self.metrics['cpu_usage']]
            summary['cpu'] = {
                'peak_percent': max(cpu_values),
                'average_percent': sum(cpu_values) / len(cpu_values),
                'min_percent': min(cpu_values)
            }
        
        # GPU statistics
        if self.metrics['gpu_usage']:
            gpu_values = [g['gpu_percent'] for g in self.metrics['gpu_usage']]
            summary['gpu'] = {
                'peak_percent': max(gpu_values),
                'average_percent': sum(gpu_values) / len(gpu_values),
                'min_percent': min(gpu_values)
            }
        
        # Algorithm metrics summary
        for algo, metrics in self.metrics['algorithm_metrics'].items():
            summary['algorithm_summary'][algo] = {}
            for metric_name, values in metrics.items():
                if values:
                    metric_values = [v['value'] for v in values]
                    if all(isinstance(v, (int, float)) for v in metric_values):
                        summary['algorithm_summary'][algo][metric_name] = {
                            'final': metric_values[-1],
                            'initial': metric_values[0],
                            'improvement': metric_values[-1] - metric_values[0],
                            'iterations': len(metric_values)
                        }
        
        return summary
    
    def save_metrics(self, filename: str = None):
        """Save detailed metrics to file"""
        if filename is None and self.output_dir:
            filename = os.path.join(self.output_dir, 'performance_metrics.json')
        
        if filename:
            metrics_data = {
                'start_time': datetime.fromtimestamp(self.start_time).isoformat(),
                'end_time': datetime.now().isoformat(),
                'total_duration': time.time() - self.start_time,
                'metrics': self.metrics,
                'summary': self.get_summary()
            }
            
            with open(filename, 'w') as f:
                json.dump(metrics_data, f, indent=2)
    
    def get_real_time_metrics(self) -> Dict[str, Any]:
        """Get current real-time metrics"""
        process = psutil.Process()
        
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'uptime': time.time() - self.start_time,
            'cpu_percent': psutil.cpu_percent(interval=0.1),
            'memory_mb': process.memory_info().rss / 1024 / 1024,
            'memory_percent': process.memory_percent()
        }
        
        # Add GPU metrics if available
        if self.gpu_available:
            try:
                handle = self.pynvml.nvmlDeviceGetHandleByIndex(0)
                util = self.pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem_info = self.pynvml.nvmlDeviceGetMemoryInfo(handle)
                
                metrics['gpu_percent'] = util.gpu
                metrics['gpu_memory_mb'] = mem_info.used / 1024 / 1024
                metrics['gpu_memory_percent'] = (mem_info.used / mem_info.total) * 100
            except:
                pass
        
        return metrics