"""
Monitoring API for Heavy Optimizer Platform
Provides REST API endpoints for real-time performance metrics
"""

import json
import os
from datetime import datetime
import threading
import time
from typing import Dict, Any, Optional

# Make Flask optional
try:
    from flask import Flask, jsonify, request
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    Flask = None
    jsonify = None
    request = None

class MonitoringAPI:
    """API server for performance monitoring"""
    
    def __init__(self, performance_monitor=None, metrics_collector=None, 
                 host='0.0.0.0', port=5000):
        """Initialize monitoring API"""
        if not FLASK_AVAILABLE:
            self.app = None
            print("Warning: Flask not available. API functionality disabled.")
        else:
            self.app = Flask(__name__)
            
        self.performance_monitor = performance_monitor
        self.metrics_collector = metrics_collector
        self.host = host
        self.port = port
        self.server_thread = None
        self.is_running = False
        
        # Historical metrics storage
        self.historical_metrics = []
        self.max_history_size = 1000
        
        # Setup routes if Flask available
        if FLASK_AVAILABLE and self.app:
            self._setup_routes()
    
    def _setup_routes(self):
        """Setup API routes"""
        
        @self.app.route('/health', methods=['GET'])
        def health_check():
            """Health check endpoint"""
            return jsonify({
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'service': 'Heavy Optimizer Monitoring API'
            })
        
        @self.app.route('/metrics/realtime', methods=['GET'])
        def get_realtime_metrics():
            """Get real-time performance metrics"""
            if not self.performance_monitor:
                return jsonify({'error': 'Performance monitor not available'}), 503
            
            try:
                metrics = self.performance_monitor.get_real_time_metrics()
                return jsonify(metrics)
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/metrics/summary', methods=['GET'])
        def get_performance_summary():
            """Get performance summary"""
            if not self.performance_monitor:
                return jsonify({'error': 'Performance monitor not available'}), 503
            
            try:
                summary = self.performance_monitor.get_summary()
                return jsonify(summary)
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/metrics/algorithms', methods=['GET'])
        def get_algorithm_metrics():
            """Get algorithm performance metrics"""
            if not self.metrics_collector:
                return jsonify({'error': 'Metrics collector not available'}), 503
            
            try:
                stats = self.metrics_collector.get_algorithm_statistics()
                convergence = self.metrics_collector.get_convergence_analysis()
                
                return jsonify({
                    'statistics': stats,
                    'convergence': convergence
                })
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/metrics/resources', methods=['GET'])
        def get_resource_metrics():
            """Get resource utilization metrics"""
            if not self.metrics_collector:
                return jsonify({'error': 'Metrics collector not available'}), 503
            
            try:
                resources = self.metrics_collector.get_resource_utilization_summary()
                return jsonify(resources)
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/metrics/history', methods=['GET'])
        def get_historical_metrics():
            """Get historical metrics"""
            # Get query parameters
            limit = request.args.get('limit', 100, type=int)
            start_time = request.args.get('start_time', None)
            end_time = request.args.get('end_time', None)
            
            # Filter historical metrics
            filtered_metrics = self.historical_metrics
            
            if start_time:
                filtered_metrics = [m for m in filtered_metrics 
                                  if m['timestamp'] >= start_time]
            
            if end_time:
                filtered_metrics = [m for m in filtered_metrics 
                                  if m['timestamp'] <= end_time]
            
            # Apply limit
            filtered_metrics = filtered_metrics[-limit:]
            
            return jsonify({
                'count': len(filtered_metrics),
                'metrics': filtered_metrics
            })
        
        @self.app.route('/metrics/component/<component_name>', methods=['GET'])
        def get_component_metrics(component_name):
            """Get metrics for a specific component"""
            if not self.performance_monitor:
                return jsonify({'error': 'Performance monitor not available'}), 503
            
            try:
                component_times = self.performance_monitor.metrics.get('component_times', {})
                execution_times = self.performance_monitor.metrics.get('execution_times', {})
                
                if component_name in component_times:
                    return jsonify({
                        'component': component_name,
                        'total_time': component_times[component_name],
                        'execution_history': execution_times.get(component_name, [])
                    })
                else:
                    return jsonify({'error': f'Component {component_name} not found'}), 404
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/metrics/alerts', methods=['GET'])
        def get_alerts():
            """Get performance alerts based on thresholds"""
            alerts = []
            
            try:
                # Load monitoring config
                config_file = '/mnt/optimizer_share/backend/config/monitoring_config.json'
                if os.path.exists(config_file):
                    with open(config_file, 'r') as f:
                        config = json.load(f)
                    
                    thresholds = config.get('alert_thresholds', {})
                    
                    # Check current metrics against thresholds
                    if self.performance_monitor:
                        current_metrics = self.performance_monitor.get_real_time_metrics()
                        
                        # Memory threshold
                        if current_metrics.get('memory_mb', 0) > thresholds.get('max_memory_usage_mb', 250):
                            alerts.append({
                                'type': 'memory',
                                'severity': 'warning',
                                'message': f"Memory usage ({current_metrics['memory_mb']:.1f} MB) exceeds threshold ({thresholds['max_memory_usage_mb']} MB)",
                                'timestamp': datetime.now().isoformat()
                            })
                        
                        # CPU threshold (if > 90%)
                        if current_metrics.get('cpu_percent', 0) > 90:
                            alerts.append({
                                'type': 'cpu',
                                'severity': 'warning',
                                'message': f"High CPU usage: {current_metrics['cpu_percent']:.1f}%",
                                'timestamp': datetime.now().isoformat()
                            })
                    
                    # Check algorithm success rates
                    if self.metrics_collector:
                        algo_stats = self.metrics_collector.get_algorithm_statistics()
                        min_success_rate = thresholds.get('min_algorithm_success_rate', 0.8)
                        
                        for algo, stats in algo_stats.items():
                            if stats['success_rate'] < min_success_rate:
                                alerts.append({
                                    'type': 'algorithm',
                                    'severity': 'error',
                                    'message': f"Algorithm {algo} success rate ({stats['success_rate']*100:.1f}%) below threshold ({min_success_rate*100:.0f}%)",
                                    'timestamp': datetime.now().isoformat()
                                })
                
                return jsonify({'alerts': alerts})
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/metrics/export', methods=['POST'])
        def export_metrics():
            """Export metrics to file"""
            try:
                data = request.get_json()
                output_dir = data.get('output_dir', '/mnt/optimizer_share/output')
                
                # Export all metrics
                if self.performance_monitor:
                    self.performance_monitor.save_metrics(
                        os.path.join(output_dir, 'performance_metrics.json'))
                
                if self.metrics_collector:
                    self.metrics_collector.export_metrics(output_dir)
                
                return jsonify({
                    'status': 'success',
                    'message': 'Metrics exported successfully',
                    'output_dir': output_dir
                })
            except Exception as e:
                return jsonify({'error': str(e)}), 500
    
    def start(self, background=True):
        """Start the API server"""
        if self.is_running:
            return
        
        self.is_running = True
        
        if background:
            self.server_thread = threading.Thread(
                target=self._run_server,
                daemon=True
            )
            self.server_thread.start()
            
            # Start metrics collection thread
            self._start_metrics_collection()
        else:
            self._run_server()
    
    def _run_server(self):
        """Run the Flask server"""
        self.app.run(host=self.host, port=self.port, debug=False)
    
    def _start_metrics_collection(self):
        """Start background metrics collection"""
        def collect_metrics():
            while self.is_running:
                if self.performance_monitor:
                    try:
                        metrics = self.performance_monitor.get_real_time_metrics()
                        
                        # Add to historical data
                        self.historical_metrics.append(metrics)
                        
                        # Trim history if too large
                        if len(self.historical_metrics) > self.max_history_size:
                            self.historical_metrics = self.historical_metrics[-self.max_history_size:]
                    except:
                        pass
                
                time.sleep(5)  # Collect every 5 seconds
        
        collection_thread = threading.Thread(target=collect_metrics, daemon=True)
        collection_thread.start()
    
    def stop(self):
        """Stop the API server"""
        self.is_running = False
        # Note: Properly stopping Flask server requires more complex handling
        # This is simplified for the example