#!/usr/bin/env python3
"""
Heavy Optimizer Platform - Enhanced Samba Job Queue Processor
Server-side daemon with comprehensive error handling, checkpoints, and recovery
"""

import os
import sys
import time
import json
import subprocess
import traceback
from datetime import datetime, timedelta
from pathlib import Path
import threading
import signal
import psutil

# Add error handling module to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import error handling components
from lib.error_handling import (
    CheckpointManager, retry, retry_on_network_error, retry_on_resource_error,
    ErrorNotifier, ContextLogger, setup_error_logging, ErrorRecoveryManager,
    JobProcessingError, JobQueueError, JobTimeoutError, FileSystemError,
    ConfigurationError, ResourceError
)

# Setup comprehensive error logging
setup_error_logging()

class EnhancedSambaJobQueueProcessor:
    def __init__(self, samba_share_path="/mnt/optimizer_share"):
        self.logger = ContextLogger(__name__)
        self.samba_share_path = Path(samba_share_path)
        self.backend_path = self.samba_share_path / "backend"
        self.jobs_path = self.samba_share_path / "jobs"
        self.queue_path = self.jobs_path / "queue"
        self.processing_path = self.jobs_path / "processing"
        self.completed_path = self.jobs_path / "completed"
        self.failed_path = self.jobs_path / "failed"
        
        # Initialize error handling components
        self.checkpoint_manager = CheckpointManager(
            base_dir=str(self.samba_share_path / "checkpoints" / "job_processor")
        )
        self.error_notifier = ErrorNotifier()
        self.recovery_manager = ErrorRecoveryManager(
            self.checkpoint_manager,
            self.error_notifier
        )
        
        # Load configuration
        self._load_configuration()
        
        # Ensure all directories exist
        self._ensure_directories()
        
        # Control flags
        self.running = True
        self.processing_jobs = {}
        self.failed_job_count = 0
        self.max_consecutive_failures = 5
        
        # Job processing statistics
        self.stats = {
            'jobs_processed': 0,
            'jobs_succeeded': 0,
            'jobs_failed': 0,
            'total_processing_time': 0,
            'start_time': datetime.now()
        }
        
        # Setup signal handlers
        signal.signal(signal.SIGTERM, self.signal_handler)
        signal.signal(signal.SIGINT, self.signal_handler)
        
        # Recovery thread for stuck jobs
        self.recovery_thread = threading.Thread(target=self._recovery_monitor)
        self.recovery_thread.daemon = True
        self.recovery_thread.start()
        
        self.logger.info("Enhanced Samba Job Queue Processor initialized")
        self.logger.info(f"Monitoring queue: {self.queue_path}")
        self.logger.info("Error handling: COMPREHENSIVE")
    
    def _load_configuration(self):
        """Load processor configuration with error handling"""
        config_path = self.backend_path / "config" / "job_processor_config.json"
        
        try:
            if config_path.exists():
                with open(config_path, 'r') as f:
                    self.config = json.load(f)
            else:
                self.config = self._get_default_config()
                
        except Exception as e:
            self.logger.warning(f"Failed to load config: {e}, using defaults")
            self.config = self._get_default_config()
    
    def _get_default_config(self):
        """Get default configuration"""
        return {
            'job_timeout': 300,  # 5 minutes
            'max_retry_attempts': 3,
            'scan_interval': 2,
            'stuck_job_threshold': 600,  # 10 minutes
            'enable_notifications': True,
            'max_concurrent_jobs': 3,
            'checkpoint_interval': 60  # Save stats every minute
        }
    
    @retry_on_resource_error(max_attempts=3)
    def _ensure_directories(self):
        """Ensure all required directories exist"""
        directories = [
            self.queue_path,
            self.processing_path,
            self.completed_path,
            self.failed_path,
            self.samba_share_path / "checkpoints" / "job_processor",
            self.samba_share_path / "logs" / "job_processor"
        ]
        
        for directory in directories:
            try:
                directory.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                raise FileSystemError(
                    f"Failed to create directory: {directory}",
                    str(directory),
                    'mkdir'
                )
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.running = False
        
        # Save final checkpoint
        self._save_processor_checkpoint()
        
        # Wait for active jobs to complete (with timeout)
        timeout = 30  # 30 seconds
        start = time.time()
        
        while self.processing_jobs and (time.time() - start) < timeout:
            self.logger.info(f"Waiting for {len(self.processing_jobs)} active jobs to complete...")
            time.sleep(1)
        
        if self.processing_jobs:
            self.logger.warning(f"{len(self.processing_jobs)} jobs still active after timeout")
    
    @retry(max_attempts=3, exceptions=(OSError, IOError))
    def scan_job_queue(self):
        """Scan for new job files with error handling"""
        try:
            job_files = list(self.queue_path.glob("*.json"))
            return sorted(job_files, key=lambda x: x.stat().st_mtime)
            
        except Exception as e:
            self.logger.error(f"Error scanning job queue: {e}", exc_info=True)
            
            # Attempt recovery
            recovery_result = self.recovery_manager.handle_error(
                JobQueueError(f"Failed to scan job queue: {e}"),
                {'queue_path': str(self.queue_path)}
            )
            
            if recovery_result and recovery_result.get('recovered'):
                return []
            
            raise
    
    def validate_job_file(self, job_file):
        """Validate job file with comprehensive error handling"""
        self.logger.set_context(job_file=str(job_file))
        
        try:
            with open(job_file, 'r') as f:
                job_data = json.load(f)
            
            # Required fields validation
            required_fields = ['job_id', 'input_file', 'portfolio_size', 'job_type', 'timestamp']
            missing_fields = [field for field in required_fields if field not in job_data]
            
            if missing_fields:
                raise JobProcessingError(
                    f"Missing required fields: {missing_fields}",
                    job_data.get('job_id', 'unknown'),
                    'VALIDATION_ERROR'
                )
            
            # Validate input file
            input_file = self.samba_share_path / "input" / job_data['input_file']
            if not input_file.exists():
                raise JobProcessingError(
                    f"Input file not found: {job_data['input_file']}",
                    job_data['job_id'],
                    'INPUT_FILE_NOT_FOUND'
                )
            
            if not job_data['input_file'].lower().endswith('.csv'):
                raise JobProcessingError(
                    "Only CSV input files are supported",
                    job_data['job_id'],
                    'INVALID_FILE_TYPE'
                )
            
            # Validate portfolio size
            portfolio_size = int(job_data['portfolio_size'])
            if not (10 <= portfolio_size <= 100):
                raise JobProcessingError(
                    f"Portfolio size must be between 10-100, got: {portfolio_size}",
                    job_data['job_id'],
                    'INVALID_PORTFOLIO_SIZE'
                )
            
            self.logger.info(f"Job {job_data['job_id']} validated successfully")
            return job_data
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in job file {job_file}: {e}")
            self._move_to_failed(job_file, {'error': f"Invalid JSON: {e}"})
            return None
            
        except JobProcessingError as e:
            self.logger.error(f"Job validation failed: {e}")
            self._move_to_failed(job_file, {
                'error': str(e),
                'error_code': e.error_code
            })
            return None
            
        except Exception as e:
            self.logger.error(f"Unexpected validation error: {e}", exc_info=True)
            self._move_to_failed(job_file, {'error': f"Validation error: {e}"})
            return None
            
        finally:
            self.logger.clear_context()
    
    @retry_on_resource_error(max_attempts=3)
    def move_job_file(self, job_file, destination_dir):
        """Move job file with retry and error handling"""
        try:
            destination = destination_dir / job_file.name
            
            # Check if destination already exists
            if destination.exists():
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                destination = destination_dir / f"{job_file.stem}_{timestamp}.json"
            
            job_file.rename(destination)
            return destination
            
        except Exception as e:
            raise FileSystemError(
                f"Failed to move job file {job_file}",
                str(job_file),
                'rename'
            )
    
    def execute_optimization_job(self, job_data, job_file):
        """Execute optimization job with comprehensive error handling"""
        job_id = job_data['job_id']
        self.logger.set_context(job_id=job_id)
        
        # Save checkpoint before execution
        self.checkpoint_manager.save_checkpoint(
            {
                'job_id': job_id,
                'job_data': job_data,
                'status': 'starting',
                'timestamp': datetime.now().isoformat()
            },
            f'job_{job_id}_start',
            f'Before executing job {job_id}'
        )
        
        try:
            self.logger.info(f"Starting optimization job {job_id}")
            
            # Move to processing directory
            processing_job_file = self.move_job_file(job_file, self.processing_path)
            if not processing_job_file:
                return False
            
            # Track active job
            self.processing_jobs[job_id] = {
                'start_time': time.time(),
                'job_file': processing_job_file,
                'job_data': job_data
            }
            
            # Use enhanced workflow with error handling
            backend_script = self.backend_path / "csv_only_heavydb_workflow_enhanced.py"
            if not backend_script.exists():
                # Fallback to original workflow
                backend_script = self.backend_path / "csv_only_heavydb_workflow.py"
            
            input_file = self.samba_share_path / "input" / job_data['input_file']
            portfolio_size = job_data['portfolio_size']
            
            # Set up environment
            env = os.environ.copy()
            env['PYTHONPATH'] = str(self.backend_path / "lib")
            env['HEAVYDB_OPTIMIZER_HOME'] = str(self.backend_path)
            
            # Execute with enhanced error handling
            cmd = [
                sys.executable,
                str(backend_script),
                "--input", str(input_file),
                "--portfolio-size", str(portfolio_size),
                "--job-id", job_id
            ]
            
            self.logger.info(f"Executing: {' '.join(cmd)}")
            
            # Execute with timeout and monitoring
            start_time = time.time()
            
            # Create process for better control
            process = subprocess.Popen(
                cmd,
                cwd=str(self.backend_path),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Monitor process with timeout
            timeout = self.config.get('job_timeout', 300)
            
            try:
                stdout, stderr = process.communicate(timeout=timeout)
                return_code = process.returncode
                
            except subprocess.TimeoutExpired:
                # Kill process
                process.kill()
                stdout, stderr = process.communicate()
                
                raise JobTimeoutError(job_id, timeout)
            
            execution_time = time.time() - start_time
            
            # Update job data
            job_data['execution_time'] = execution_time
            job_data['completion_timestamp'] = datetime.now().isoformat()
            job_data['return_code'] = return_code
            job_data['stdout'] = stdout[-10000:]  # Last 10K chars
            job_data['stderr'] = stderr[-10000:]  # Last 10K chars
            
            # Save execution checkpoint
            self.checkpoint_manager.save_checkpoint(
                {
                    'job_id': job_id,
                    'status': 'completed',
                    'return_code': return_code,
                    'execution_time': execution_time
                },
                f'job_{job_id}_complete',
                f'After executing job {job_id}'
            )
            
            if return_code == 0:
                self._handle_successful_job(job_id, job_data, processing_job_file)
                return True
            else:
                self._handle_failed_job(job_id, job_data, processing_job_file,
                                      f"Return code: {return_code}")
                return False
                
        except JobTimeoutError as e:
            self.logger.error(f"Job {job_id} timed out: {e}")
            self._handle_timeout_job(job_id, job_data, e)
            return False
            
        except Exception as e:
            self.logger.error(f"Unexpected error executing job {job_id}: {e}", exc_info=True)
            
            # Attempt recovery
            recovery_result = self.recovery_manager.handle_error(
                JobProcessingError(f"Job execution failed: {e}", job_id),
                {'job_data': job_data, 'job_id': job_id}
            )
            
            if recovery_result and recovery_result.get('recovered'):
                # Retry job
                if recovery_result.get('retry'):
                    self.logger.info(f"Retrying job {job_id} after recovery")
                    # Move job back to queue for retry
                    self.move_job_file(
                        self.processing_jobs[job_id]['job_file'],
                        self.queue_path
                    )
                    return None
            
            self._handle_failed_job(job_id, job_data, 
                                  self.processing_jobs.get(job_id, {}).get('job_file'),
                                  str(e))
            return False
            
        finally:
            # Remove from active jobs
            self.processing_jobs.pop(job_id, None)
            self.logger.clear_context()
    
    def _handle_successful_job(self, job_id, job_data, processing_job_file):
        """Handle successful job completion"""
        self.logger.info(f"Job {job_id} completed successfully in {job_data['execution_time']:.2f}s")
        
        # Update statistics
        self.stats['jobs_succeeded'] += 1
        self.stats['total_processing_time'] += job_data['execution_time']
        self.failed_job_count = 0  # Reset consecutive failure count
        
        # Save to completed directory
        completed_job_file = self.completed_path / f"{job_id}_completed.json"
        with open(completed_job_file, 'w') as f:
            json.dump(job_data, f, indent=2)
        
        # Remove from processing
        if processing_job_file and processing_job_file.exists():
            processing_job_file.unlink()
    
    def _handle_failed_job(self, job_id, job_data, processing_job_file, error_message):
        """Handle failed job with notifications"""
        self.logger.error(f"Job {job_id} failed: {error_message}")
        
        # Update statistics
        self.stats['jobs_failed'] += 1
        self.failed_job_count += 1
        
        # Add error information
        job_data['error'] = error_message
        job_data['failure_timestamp'] = datetime.now().isoformat()
        
        # Save to failed directory
        failed_job_file = self.failed_path / f"{job_id}_failed.json"
        with open(failed_job_file, 'w') as f:
            json.dump(job_data, f, indent=2)
        
        # Remove from processing
        if processing_job_file and processing_job_file.exists():
            processing_job_file.unlink()
        
        # Send notification if too many consecutive failures
        if self.failed_job_count >= self.max_consecutive_failures:
            self.error_notifier.notify_critical_error(
                JobProcessingError(
                    f"Too many consecutive job failures: {self.failed_job_count}",
                    job_id
                ),
                {'job_id': job_id, 'consecutive_failures': self.failed_job_count},
                'CRITICAL'
            )
    
    def _handle_timeout_job(self, job_id, job_data, timeout_error):
        """Handle job timeout"""
        job_data['error'] = str(timeout_error)
        job_data['timeout'] = True
        job_data['failure_timestamp'] = datetime.now().isoformat()
        
        # Save to failed directory
        failed_job_file = self.failed_path / f"{job_id}_timeout.json"
        with open(failed_job_file, 'w') as f:
            json.dump(job_data, f, indent=2)
        
        # Remove from processing if exists
        processing_file = self.processing_jobs.get(job_id, {}).get('job_file')
        if processing_file and processing_file.exists():
            processing_file.unlink()
    
    def _move_to_failed(self, job_file, error_info):
        """Move job to failed directory with error information"""
        try:
            # Create error job data
            error_job_data = {
                'original_file': job_file.name,
                'error_timestamp': datetime.now().isoformat(),
                'error_info': error_info
            }
            
            # Try to read original job data
            try:
                with open(job_file, 'r') as f:
                    original_data = json.load(f)
                    error_job_data.update(original_data)
            except:
                pass
            
            # Save to failed directory
            failed_file = self.failed_path / f"invalid_{job_file.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(failed_file, 'w') as f:
                json.dump(error_job_data, f, indent=2)
            
            # Remove original
            job_file.unlink()
            
        except Exception as e:
            self.logger.error(f"Failed to move invalid job to failed directory: {e}")
    
    def _recovery_monitor(self):
        """Monitor for stuck jobs and attempt recovery"""
        while self.running:
            try:
                time.sleep(60)  # Check every minute
                
                # Check for stuck jobs in processing directory
                stuck_threshold = self.config.get('stuck_job_threshold', 600)
                current_time = time.time()
                
                for job_file in self.processing_path.glob("*.json"):
                    try:
                        # Check file age
                        file_age = current_time - job_file.stat().st_mtime
                        
                        if file_age > stuck_threshold:
                            self.logger.warning(f"Found stuck job: {job_file.name} (age: {file_age:.0f}s)")
                            
                            # Try to read job data
                            with open(job_file, 'r') as f:
                                job_data = json.load(f)
                            
                            # Check if job is actually being processed
                            job_id = job_data.get('job_id', 'unknown')
                            if job_id not in self.processing_jobs:
                                # Job is stuck, move to failed
                                self.logger.error(f"Moving stuck job {job_id} to failed")
                                self._handle_failed_job(
                                    job_id, job_data, job_file,
                                    f"Job stuck in processing for {file_age:.0f}s"
                                )
                            
                    except Exception as e:
                        self.logger.error(f"Error checking stuck job {job_file}: {e}")
                
                # Save periodic checkpoint
                self._save_processor_checkpoint()
                
            except Exception as e:
                self.logger.error(f"Error in recovery monitor: {e}", exc_info=True)
    
    def _save_processor_checkpoint(self):
        """Save processor state checkpoint"""
        try:
            checkpoint_data = {
                'stats': self.stats,
                'timestamp': datetime.now().isoformat(),
                'active_jobs': list(self.processing_jobs.keys()),
                'failed_job_count': self.failed_job_count
            }
            
            self.checkpoint_manager.save_checkpoint(
                checkpoint_data,
                'processor_state',
                'Periodic processor state checkpoint'
            )
            
        except Exception as e:
            self.logger.error(f"Failed to save processor checkpoint: {e}")
    
    def process_job_queue(self):
        """Main job processing loop with error handling"""
        self.logger.info("Starting enhanced job queue processing loop")
        
        # Load last checkpoint if available
        last_state = self.checkpoint_manager.load_checkpoint('processor_state')
        if last_state:
            self.stats.update(last_state.get('stats', {}))
            self.failed_job_count = last_state.get('failed_job_count', 0)
            self.logger.info(f"Resumed from checkpoint with {self.stats['jobs_processed']} jobs processed")
        
        while self.running:
            try:
                # Update statistics
                self.stats['jobs_processed'] = self.stats['jobs_succeeded'] + self.stats['jobs_failed']
                
                # Check system resources
                memory_percent = psutil.virtual_memory().percent
                if memory_percent > 90:
                    self.logger.warning(f"High memory usage: {memory_percent}%")
                    time.sleep(10)  # Wait for resources to free up
                    continue
                
                # Scan for new jobs
                job_files = self.scan_job_queue()
                
                if not job_files:
                    time.sleep(self.config.get('scan_interval', 2))
                    continue
                
                # Check concurrent job limit
                max_concurrent = self.config.get('max_concurrent_jobs', 3)
                if len(self.processing_jobs) >= max_concurrent:
                    self.logger.info(f"Max concurrent jobs ({max_concurrent}) reached, waiting...")
                    time.sleep(5)
                    continue
                
                for job_file in job_files:
                    if not self.running:
                        break
                    
                    if len(self.processing_jobs) >= max_concurrent:
                        break
                    
                    self.logger.info(f"Processing job file: {job_file.name}")
                    
                    # Validate job
                    job_data = self.validate_job_file(job_file)
                    if not job_data:
                        continue
                    
                    # Execute job in thread for concurrency
                    job_thread = threading.Thread(
                        target=self.execute_optimization_job,
                        args=(job_data, job_file)
                    )
                    job_thread.start()
                    
                    # Small delay between job starts
                    time.sleep(0.5)
                
            except Exception as e:
                self.logger.error(f"Error in main processing loop: {e}", exc_info=True)
                
                # Send notification for critical loop error
                self.error_notifier.notify_critical_error(
                    Exception(f"Job processor main loop error: {e}"),
                    {'processor_stats': self.stats},
                    'CRITICAL'
                )
                
                # Wait before retrying
                time.sleep(10)
        
        self.logger.info("Job queue processing stopped")
        self._print_final_statistics()
    
    def _print_final_statistics(self):
        """Print final processing statistics"""
        uptime = datetime.now() - self.stats['start_time']
        
        self.logger.info("=" * 60)
        self.logger.info("FINAL PROCESSING STATISTICS")
        self.logger.info("=" * 60)
        self.logger.info(f"Uptime: {uptime}")
        self.logger.info(f"Total jobs processed: {self.stats['jobs_processed']}")
        self.logger.info(f"Successful jobs: {self.stats['jobs_succeeded']}")
        self.logger.info(f"Failed jobs: {self.stats['jobs_failed']}")
        if self.stats['jobs_succeeded'] > 0:
            avg_time = self.stats['total_processing_time'] / self.stats['jobs_succeeded']
            self.logger.info(f"Average processing time: {avg_time:.2f}s")
        self.logger.info("=" * 60)
    
    def run(self):
        """Run the job processor with error handling"""
        try:
            self.logger.info("Starting Enhanced Samba Job Queue Processor")
            self.logger.info(f"Configuration: {json.dumps(self.config, indent=2)}")
            
            # Start processing
            self.process_job_queue()
            
        except Exception as e:
            self.logger.critical(f"Fatal error in job processor: {e}", exc_info=True)
            
            # Send critical notification
            self.error_notifier.notify_critical_error(
                Exception(f"Job processor fatal error: {e}"),
                {'config': self.config},
                'CRITICAL'
            )
            
            raise
        
        finally:
            # Save final checkpoint
            self._save_processor_checkpoint()
            
            # Cleanup
            self.logger.info("Job processor shutdown complete")


def main():
    """Main entry point for the enhanced job processor"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Enhanced Samba Job Queue Processor with Error Handling'
    )
    parser.add_argument('--samba-share', default='/mnt/optimizer_share',
                       help='Path to Samba share (default: /mnt/optimizer_share)')
    parser.add_argument('--daemon', action='store_true',
                       help='Run as daemon in background')
    
    args = parser.parse_args()
    
    # Create and run processor
    processor = EnhancedSambaJobQueueProcessor(args.samba_share)
    
    try:
        if args.daemon:
            # Run as daemon
            import daemon
            with daemon.DaemonContext():
                processor.run()
        else:
            # Run in foreground
            processor.run()
            
    except KeyboardInterrupt:
        processor.logger.info("Received keyboard interrupt")
        processor.running = False
        
    except Exception as e:
        processor.logger.critical(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()