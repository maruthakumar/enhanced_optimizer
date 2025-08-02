"""
Historical Performance Data Storage for Heavy Optimizer Platform
Manages persistent storage of performance metrics over time
"""

import json
import os
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import gzip
import shutil

class HistoricalStorage:
    """Manages historical performance data storage"""
    
    def __init__(self, data_dir: str = '/mnt/optimizer_share/logs/performance_history'):
        """Initialize historical storage"""
        self.data_dir = data_dir
        self.db_file = os.path.join(data_dir, 'performance_metrics.db')
        
        # Create directory if needed
        os.makedirs(data_dir, exist_ok=True)
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for metrics storage"""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS run_summary (
                run_id TEXT PRIMARY KEY,
                start_time TIMESTAMP,
                end_time TIMESTAMP,
                total_duration REAL,
                input_file TEXT,
                portfolio_size INTEGER,
                strategies_count INTEGER,
                success BOOLEAN,
                error_message TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS component_times (
                run_id TEXT,
                component TEXT,
                execution_time REAL,
                timestamp TIMESTAMP,
                FOREIGN KEY (run_id) REFERENCES run_summary(run_id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS algorithm_performance (
                run_id TEXT,
                algorithm TEXT,
                execution_time REAL,
                iterations INTEGER,
                final_fitness REAL,
                convergence_generation INTEGER,
                success BOOLEAN,
                timestamp TIMESTAMP,
                FOREIGN KEY (run_id) REFERENCES run_summary(run_id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS resource_usage (
                run_id TEXT,
                timestamp TIMESTAMP,
                cpu_percent REAL,
                memory_mb REAL,
                gpu_percent REAL,
                gpu_memory_mb REAL,
                FOREIGN KEY (run_id) REFERENCES run_summary(run_id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS throughput_metrics (
                run_id TEXT,
                operation TEXT,
                data_size_mb REAL,
                duration_s REAL,
                throughput_mb_s REAL,
                timestamp TIMESTAMP,
                FOREIGN KEY (run_id) REFERENCES run_summary(run_id)
            )
        ''')
        
        # Create indices for faster queries
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_run_time ON run_summary(start_time)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_algorithm ON algorithm_performance(algorithm)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_component ON component_times(component)')
        
        conn.commit()
        conn.close()
    
    def store_run_data(self, run_id: str, performance_monitor, metrics_collector, 
                      run_metadata: Dict[str, Any]):
        """Store data from a complete optimization run"""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        try:
            # Store run summary
            summary = performance_monitor.get_summary() if performance_monitor else {}
            
            cursor.execute('''
                INSERT INTO run_summary 
                (run_id, start_time, end_time, total_duration, input_file, 
                 portfolio_size, strategies_count, success, error_message)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                run_id,
                datetime.fromtimestamp(performance_monitor.start_time).isoformat() if performance_monitor else None,
                datetime.now().isoformat(),
                summary.get('total_execution_time', 0),
                run_metadata.get('input_file', ''),
                run_metadata.get('portfolio_size', 0),
                run_metadata.get('strategies_count', 0),
                run_metadata.get('success', True),
                run_metadata.get('error_message', '')
            ))
            
            # Store component times
            if performance_monitor and 'component_times' in summary:
                for component, time_val in summary['component_times'].items():
                    cursor.execute('''
                        INSERT INTO component_times (run_id, component, execution_time, timestamp)
                        VALUES (?, ?, ?, ?)
                    ''', (run_id, component, time_val, datetime.now().isoformat()))
            
            # Store algorithm performance
            if metrics_collector:
                algo_stats = metrics_collector.get_algorithm_statistics()
                convergence = metrics_collector.get_convergence_analysis()
                
                for algo, stats in algo_stats.items():
                    cursor.execute('''
                        INSERT INTO algorithm_performance 
                        (run_id, algorithm, execution_time, iterations, final_fitness, 
                         convergence_generation, success, timestamp)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        run_id,
                        algo,
                        stats['execution_time']['mean'],
                        stats['average_iterations'],
                        stats['average_fitness'],
                        convergence.get(algo, {}).get('convergence_generation', 0),
                        stats['success_rate'] > 0.5,
                        datetime.now().isoformat()
                    ))
            
            # Store throughput metrics
            if performance_monitor and 'data_throughput' in summary:
                for operation, metrics in summary['data_throughput'].items():
                    cursor.execute('''
                        INSERT INTO throughput_metrics 
                        (run_id, operation, data_size_mb, duration_s, throughput_mb_s, timestamp)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', (
                        run_id,
                        operation,
                        metrics['size_mb'],
                        metrics['duration_s'],
                        metrics['throughput_mb_s'],
                        datetime.now().isoformat()
                    ))
            
            # Store sampled resource usage
            if performance_monitor and performance_monitor.metrics['memory_usage']:
                # Sample every 10th measurement to avoid bloat
                sample_rate = 10
                for i, snapshot in enumerate(performance_monitor.metrics['memory_usage']):
                    if i % sample_rate == 0:
                        cpu_data = performance_monitor.metrics['cpu_usage'][i] if i < len(performance_monitor.metrics['cpu_usage']) else {}
                        gpu_data = performance_monitor.metrics['gpu_usage'][i] if i < len(performance_monitor.metrics['gpu_usage']) else {}
                        
                        cursor.execute('''
                            INSERT INTO resource_usage 
                            (run_id, timestamp, cpu_percent, memory_mb, gpu_percent, gpu_memory_mb)
                            VALUES (?, ?, ?, ?, ?, ?)
                        ''', (
                            run_id,
                            datetime.now().isoformat(),
                            cpu_data.get('percent', 0),
                            snapshot.get('rss_mb', 0),
                            gpu_data.get('gpu_percent', 0) if gpu_data else 0,
                            snapshot.get('gpu_used_mb', 0)
                        ))
            
            conn.commit()
            
            # Also save detailed metrics to compressed JSON
            self._save_detailed_metrics(run_id, performance_monitor, metrics_collector)
            
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
    
    def _save_detailed_metrics(self, run_id: str, performance_monitor, metrics_collector):
        """Save detailed metrics to compressed JSON file"""
        detailed_file = os.path.join(self.data_dir, f'{run_id}_detailed.json.gz')
        
        detailed_data = {
            'run_id': run_id,
            'timestamp': datetime.now().isoformat()
        }
        
        if performance_monitor:
            detailed_data['performance_metrics'] = performance_monitor.metrics
            detailed_data['performance_summary'] = performance_monitor.get_summary()
        
        if metrics_collector:
            detailed_data['algorithm_runs'] = metrics_collector.algorithm_runs
            detailed_data['convergence_data'] = metrics_collector.convergence_data
            detailed_data['fitness_history'] = metrics_collector.fitness_history
            detailed_data['resource_snapshots'] = metrics_collector.resource_snapshots
        
        # Compress and save
        with gzip.open(detailed_file, 'wt', encoding='utf-8') as f:
            json.dump(detailed_data, f, indent=2)
    
    def get_run_summary(self, run_id: str) -> Dict[str, Any]:
        """Get summary for a specific run"""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM run_summary WHERE run_id = ?', (run_id,))
        row = cursor.fetchone()
        
        if not row:
            conn.close()
            return None
        
        columns = [description[0] for description in cursor.description]
        summary = dict(zip(columns, row))
        
        # Get component times
        cursor.execute('SELECT component, execution_time FROM component_times WHERE run_id = ?', (run_id,))
        summary['component_times'] = {row[0]: row[1] for row in cursor.fetchall()}
        
        # Get algorithm performance
        cursor.execute('''
            SELECT algorithm, execution_time, iterations, final_fitness, success 
            FROM algorithm_performance WHERE run_id = ?
        ''', (run_id,))
        summary['algorithms'] = [
            {
                'algorithm': row[0],
                'execution_time': row[1],
                'iterations': row[2],
                'final_fitness': row[3],
                'success': row[4]
            }
            for row in cursor.fetchall()
        ]
        
        conn.close()
        return summary
    
    def get_historical_trends(self, days: int = 7, algorithm: str = None) -> Dict[str, Any]:
        """Get historical performance trends"""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        start_date = datetime.now() - timedelta(days=days)
        
        # Get run summaries
        cursor.execute('''
            SELECT start_time, total_duration, portfolio_size, strategies_count, success
            FROM run_summary 
            WHERE start_time >= ? 
            ORDER BY start_time
        ''', (start_date.isoformat(),))
        
        runs = cursor.fetchall()
        
        trends = {
            'period_days': days,
            'total_runs': len(runs),
            'success_rate': sum(1 for r in runs if r[4]) / len(runs) if runs else 0,
            'average_duration': sum(r[1] for r in runs) / len(runs) if runs else 0,
            'runs_by_date': {},
            'algorithm_trends': {}
        }
        
        # Group by date
        for run in runs:
            date = run[0].split('T')[0]
            if date not in trends['runs_by_date']:
                trends['runs_by_date'][date] = 0
            trends['runs_by_date'][date] += 1
        
        # Algorithm-specific trends
        if algorithm:
            cursor.execute('''
                SELECT AVG(execution_time), AVG(iterations), AVG(final_fitness), 
                       COUNT(*), SUM(CASE WHEN success THEN 1 ELSE 0 END)
                FROM algorithm_performance
                WHERE algorithm = ? AND run_id IN (
                    SELECT run_id FROM run_summary WHERE start_time >= ?
                )
            ''', (algorithm, start_date.isoformat()))
            
            row = cursor.fetchone()
            if row[3] > 0:  # Has data
                trends['algorithm_trends'][algorithm] = {
                    'avg_execution_time': row[0],
                    'avg_iterations': row[1],
                    'avg_fitness': row[2],
                    'total_runs': row[3],
                    'success_rate': row[4] / row[3] if row[3] > 0 else 0
                }
        else:
            # Get trends for all algorithms
            cursor.execute('''
                SELECT algorithm, AVG(execution_time), AVG(final_fitness), COUNT(*)
                FROM algorithm_performance
                WHERE run_id IN (
                    SELECT run_id FROM run_summary WHERE start_time >= ?
                )
                GROUP BY algorithm
            ''', (start_date.isoformat(),))
            
            for row in cursor.fetchall():
                trends['algorithm_trends'][row[0]] = {
                    'avg_execution_time': row[1],
                    'avg_fitness': row[2],
                    'total_runs': row[3]
                }
        
        conn.close()
        return trends
    
    def cleanup_old_data(self, days_to_keep: int = 30):
        """Clean up old historical data"""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        # Get runs to delete
        cursor.execute('SELECT run_id FROM run_summary WHERE start_time < ?', 
                      (cutoff_date.isoformat(),))
        old_runs = [row[0] for row in cursor.fetchall()]
        
        # Delete from all tables
        for run_id in old_runs:
            cursor.execute('DELETE FROM component_times WHERE run_id = ?', (run_id,))
            cursor.execute('DELETE FROM algorithm_performance WHERE run_id = ?', (run_id,))
            cursor.execute('DELETE FROM resource_usage WHERE run_id = ?', (run_id,))
            cursor.execute('DELETE FROM throughput_metrics WHERE run_id = ?', (run_id,))
            cursor.execute('DELETE FROM run_summary WHERE run_id = ?', (run_id,))
            
            # Delete detailed metrics file
            detailed_file = os.path.join(self.data_dir, f'{run_id}_detailed.json.gz')
            if os.path.exists(detailed_file):
                os.remove(detailed_file)
        
        conn.commit()
        conn.close()
        
        return len(old_runs)