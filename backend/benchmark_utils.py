#!/usr/bin/env python3
"""
Benchmark Utilities
Supporting utilities for performance monitoring and resource tracking
during benchmark execution
"""

import time
import psutil
import threading
import json
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import pandas as pd

@dataclass
class ResourceSnapshot:
    """Single point-in-time resource measurement"""
    timestamp: float
    memory_mb: float
    cpu_percent: float
    disk_io_read_mb: float
    disk_io_write_mb: float

@dataclass
class ResourceProfile:
    """Complete resource usage profile for a benchmark run"""
    start_time: float
    end_time: float
    duration: float
    snapshots: List[ResourceSnapshot]
    peak_memory_mb: float
    avg_memory_mb: float
    peak_cpu_percent: float
    avg_cpu_percent: float
    disk_read_mb: float
    disk_write_mb: float

class ResourceMonitor:
    """Real-time system resource monitoring for benchmark processes"""
    
    def __init__(self, sample_interval: float = 0.5):
        self.sample_interval = sample_interval
        self.monitoring = False
        self.snapshots: List[ResourceSnapshot] = []
        self.monitor_thread: Optional[threading.Thread] = None
        self.target_process: Optional[psutil.Process] = None
        
    def start_monitoring(self, process_pid: Optional[int] = None) -> None:
        """Start monitoring system resources"""
        if self.monitoring:
            return
            
        self.monitoring = True
        self.snapshots = []
        
        if process_pid:
            try:
                self.target_process = psutil.Process(process_pid)
            except psutil.NoSuchProcess:
                self.target_process = None
        
        self.monitor_thread = threading.Thread(target=self._monitoring_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_monitoring(self) -> ResourceProfile:
        """Stop monitoring and return resource profile"""
        if not self.monitoring:
            return self._empty_profile()
            
        self.monitoring = False
        
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2.0)
        
        return self._generate_profile()
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop running in separate thread"""
        start_time = time.time()
        
        # Initial disk I/O reading
        initial_disk_io = psutil.disk_io_counters()
        initial_read_mb = initial_disk_io.read_bytes / 1024 / 1024 if initial_disk_io else 0
        initial_write_mb = initial_disk_io.write_bytes / 1024 / 1024 if initial_disk_io else 0
        
        while self.monitoring:
            try:
                timestamp = time.time()
                
                # Get memory and CPU info
                if self.target_process and self.target_process.is_running():
                    # Monitor specific process
                    memory_info = self.target_process.memory_info()
                    memory_mb = memory_info.rss / 1024 / 1024
                    cpu_percent = self.target_process.cpu_percent()
                else:
                    # Monitor system-wide
                    memory_info = psutil.virtual_memory()
                    memory_mb = memory_info.used / 1024 / 1024
                    cpu_percent = psutil.cpu_percent()
                
                # Get disk I/O info
                current_disk_io = psutil.disk_io_counters()
                if current_disk_io:
                    disk_read_mb = current_disk_io.read_bytes / 1024 / 1024 - initial_read_mb
                    disk_write_mb = current_disk_io.write_bytes / 1024 / 1024 - initial_write_mb
                else:
                    disk_read_mb = disk_write_mb = 0
                
                # Create snapshot
                snapshot = ResourceSnapshot(
                    timestamp=timestamp,
                    memory_mb=memory_mb,
                    cpu_percent=cpu_percent,
                    disk_io_read_mb=disk_read_mb,
                    disk_io_write_mb=disk_write_mb
                )
                
                self.snapshots.append(snapshot)
                
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                # Process may have ended, continue with system monitoring
                self.target_process = None
            except Exception as e:
                print(f"Warning: Resource monitoring error: {e}")
            
            time.sleep(self.sample_interval)
    
    def _generate_profile(self) -> ResourceProfile:
        """Generate resource profile from collected snapshots"""
        if not self.snapshots:
            return self._empty_profile()
        
        start_time = self.snapshots[0].timestamp
        end_time = self.snapshots[-1].timestamp
        duration = end_time - start_time
        
        memory_values = [s.memory_mb for s in self.snapshots]
        cpu_values = [s.cpu_percent for s in self.snapshots]
        
        return ResourceProfile(
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            snapshots=self.snapshots,
            peak_memory_mb=max(memory_values),
            avg_memory_mb=np.mean(memory_values),
            peak_cpu_percent=max(cpu_values),
            avg_cpu_percent=np.mean(cpu_values),
            disk_read_mb=self.snapshots[-1].disk_io_read_mb if self.snapshots else 0,
            disk_write_mb=self.snapshots[-1].disk_io_write_mb if self.snapshots else 0
        )
    
    def _empty_profile(self) -> ResourceProfile:
        """Return empty resource profile"""
        return ResourceProfile(
            start_time=0, end_time=0, duration=0, snapshots=[],
            peak_memory_mb=0, avg_memory_mb=0, peak_cpu_percent=0, avg_cpu_percent=0,
            disk_read_mb=0, disk_write_mb=0
        )

class BenchmarkTimer:
    """High-precision timing utility for benchmark operations"""
    
    def __init__(self):
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.checkpoints: Dict[str, float] = {}
    
    def start(self) -> None:
        """Start timing"""
        self.start_time = time.perf_counter()
        self.checkpoints = {}
    
    def checkpoint(self, name: str) -> float:
        """Record a checkpoint and return elapsed time"""
        if self.start_time is None:
            raise RuntimeError("Timer not started")
        
        checkpoint_time = time.perf_counter()
        elapsed = checkpoint_time - self.start_time
        self.checkpoints[name] = elapsed
        return elapsed
    
    def stop(self) -> float:
        """Stop timing and return total elapsed time"""
        if self.start_time is None:
            raise RuntimeError("Timer not started")
        
        self.end_time = time.perf_counter()
        return self.end_time - self.start_time
    
    def get_elapsed(self) -> float:
        """Get current elapsed time without stopping"""
        if self.start_time is None:
            return 0.0
        
        current_time = time.perf_counter()
        return current_time - self.start_time
    
    def get_checkpoint_summary(self) -> Dict[str, float]:
        """Get summary of all checkpoints"""
        return self.checkpoints.copy()

class DatasetProcessor:
    """Utilities for processing and sampling benchmark datasets"""
    
    @staticmethod
    def create_stratified_sample(data: pd.DataFrame, n: int, strata_column: str, random_state: int = 42) -> pd.DataFrame:
        """Create stratified sample based on specified column"""
        # Calculate strata
        quartiles = data[strata_column].quantile([0.25, 0.5, 0.75]).values
        
        def assign_stratum(value):
            if value <= quartiles[0]:
                return 0
            elif value <= quartiles[1]:
                return 1
            elif value <= quartiles[2]:
                return 2
            else:
                return 3
        
        data['stratum'] = data[strata_column].apply(assign_stratum)
        
        # Sample from each stratum
        samples_per_stratum = n // 4
        stratified_samples = []
        
        for stratum in range(4):
            stratum_data = data[data['stratum'] == stratum]
            sample_size = min(samples_per_stratum, len(stratum_data))
            
            if sample_size > 0:
                stratum_sample = stratum_data.sample(n=sample_size, random_state=random_state + stratum)
                stratified_samples.append(stratum_sample)
        
        result = pd.concat(stratified_samples) if stratified_samples else pd.DataFrame()
        return result.drop('stratum', axis=1)
    
    @staticmethod
    def create_performance_based_sample(data: pd.DataFrame, n: int, performance_column: str, 
                                      top_ratio: float = 0.5, random_state: int = 42) -> pd.DataFrame:
        """Create sample mixing top performers with random selection"""
        if n >= len(data):
            return data
        
        # Sort by performance
        sorted_data = data.sort_values(performance_column, ascending=False)
        
        # Take top performers
        top_n = int(n * top_ratio)
        top_performers = sorted_data.head(top_n)
        
        # Random sample for remainder
        remaining_n = n - top_n
        random_sample = data.sample(n=remaining_n, random_state=random_state)
        
        # Combine and remove duplicates
        combined = pd.concat([top_performers, random_sample]).drop_duplicates()
        
        # If we have fewer than n after deduplication, fill with additional random samples
        if len(combined) < n:
            remaining_data = data[~data.index.isin(combined.index)]
            additional_n = n - len(combined)
            if len(remaining_data) > 0:
                additional_sample = remaining_data.sample(
                    n=min(additional_n, len(remaining_data)), 
                    random_state=random_state + 1
                )
                combined = pd.concat([combined, additional_sample])
        
        return combined.head(n)
    
    @staticmethod
    def validate_dataset_quality(data: pd.DataFrame, required_columns: List[str]) -> Dict[str, bool]:
        """Validate dataset quality for benchmark use"""
        validation_results = {}
        
        # Check required columns
        validation_results['has_required_columns'] = all(col in data.columns for col in required_columns)
        
        # Check for missing values
        validation_results['no_missing_values'] = not data[required_columns].isnull().any().any()
        
        # Check data types
        numeric_columns = ['ROI_Total', 'Max_Drawdown', 'Win_Rate', 'Profit_Factor']
        numeric_columns = [col for col in numeric_columns if col in data.columns]
        validation_results['numeric_columns_valid'] = all(
            pd.api.types.is_numeric_dtype(data[col]) for col in numeric_columns
        )
        
        # Check data ranges
        if 'ROI_Total' in data.columns:
            validation_results['roi_range_reasonable'] = (
                data['ROI_Total'].min() >= -100 and data['ROI_Total'].max() <= 1000
            )
        
        if 'Win_Rate' in data.columns:
            validation_results['win_rate_range_valid'] = (
                data['Win_Rate'].min() >= 0 and data['Win_Rate'].max() <= 100
            )
        
        return validation_results

class ResultsValidator:
    """Validation utilities for benchmark results"""
    
    @staticmethod
    def validate_performance_improvement(baseline_time: float, new_time: float, 
                                       minimum_improvement: float = 2.0) -> bool:
        """Validate that performance improvement meets minimum threshold"""
        if new_time <= 0:
            return False
        improvement_ratio = baseline_time / new_time
        return improvement_ratio >= minimum_improvement
    
    @staticmethod
    def validate_accuracy_within_tolerance(baseline_value: float, new_value: float, 
                                         tolerance_percent: float = 5.0) -> bool:
        """Validate that new value is within tolerance of baseline"""
        if baseline_value == 0:
            return new_value == 0
        
        difference_percent = abs(new_value - baseline_value) / abs(baseline_value) * 100
        return difference_percent <= tolerance_percent
    
    @staticmethod
    def validate_resource_constraints(memory_mb: float, max_memory_mb: float = 4000) -> bool:
        """Validate that resource usage is within constraints"""
        return memory_mb <= max_memory_mb
    
    @staticmethod
    def validate_portfolio_quality(portfolio_data: Dict) -> bool:
        """Validate that portfolio results meet quality standards"""
        checks = [
            portfolio_data.get('fitness', 0) > 0,
            portfolio_data.get('portfolio_size', 0) > 0,
            portfolio_data.get('roi_total', 0) != 0,
            len(portfolio_data.get('strategies_selected', [])) > 0
        ]
        return all(checks)

def save_benchmark_results(results: Dict, output_file: Path) -> None:
    """Save benchmark results to JSON file with proper formatting"""
    # Convert dataclasses and numpy types to JSON-serializable format
    def convert_types(obj):
        if hasattr(obj, '__dict__'):
            return asdict(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return str(obj)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=convert_types)

def load_benchmark_results(input_file: Path) -> Dict:
    """Load benchmark results from JSON file"""
    with open(input_file, 'r') as f:
        return json.load(f)

def compare_benchmark_runs(baseline_file: Path, new_file: Path) -> Dict:
    """Compare two benchmark runs and generate comparison report"""
    baseline_results = load_benchmark_results(baseline_file)
    new_results = load_benchmark_results(new_file)
    
    comparison = {
        'baseline_timestamp': baseline_results.get('benchmark_info', {}).get('timestamp'),
        'new_timestamp': new_results.get('benchmark_info', {}).get('timestamp'),
        'performance_changes': [],
        'accuracy_changes': [],
        'resource_changes': []
    }
    
    # Compare results by test case
    baseline_tests = {r['test_name']: r for r in baseline_results.get('results', [])}
    new_tests = {r['test_name']: r for r in new_results.get('results', [])}
    
    for test_name in baseline_tests:
        if test_name in new_tests:
            baseline_test = baseline_tests[test_name]
            new_test = new_tests[test_name]
            
            # Performance comparison
            perf_change = {
                'test_name': test_name,
                'baseline_time': baseline_test.get('heavydb_time', 0),
                'new_time': new_test.get('heavydb_time', 0),
                'change_percent': 0
            }
            
            if baseline_test.get('heavydb_time', 0) > 0:
                perf_change['change_percent'] = (
                    (new_test.get('heavydb_time', 0) - baseline_test.get('heavydb_time', 0)) /
                    baseline_test.get('heavydb_time', 0) * 100
                )
            
            comparison['performance_changes'].append(perf_change)
    
    return comparison