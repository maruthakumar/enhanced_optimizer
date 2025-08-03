#!/usr/bin/env python3
"""
Heavy Optimizer Platform - Samba Job Queue Processor
Server-side daemon that processes optimization jobs submitted via Samba share
Uses modern Parquet/Arrow/cuDF pipeline for GPU acceleration
"""

import os
import sys
import time
import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path
import threading
import signal

class SambaJobQueueProcessor:
    def __init__(self, samba_share_path="/mnt/optimizer_share"):
        self.samba_share_path = Path(samba_share_path)
        self.backend_path = self.samba_share_path / "backend"
        self.jobs_path = self.samba_share_path / "jobs"
        self.queue_path = self.jobs_path / "queue"
        self.processing_path = self.jobs_path / "processing"
        self.completed_path = self.jobs_path / "completed"
        self.failed_path = self.jobs_path / "failed"
        
        # Ensure all directories exist
        for path in [self.queue_path, self.processing_path, self.completed_path, self.failed_path]:
            path.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Control flags
        self.running = True
        self.processing_jobs = {}
        
        # Setup signal handlers
        signal.signal(signal.SIGTERM, self.signal_handler)
        signal.signal(signal.SIGINT, self.signal_handler)
        
        self.logger.info("Samba Job Queue Processor initialized")
        self.logger.info(f"Monitoring queue: {self.queue_path}")
    
    def setup_logging(self):
        """Setup comprehensive logging"""
        log_dir = self.samba_share_path / "logs"
        log_dir.mkdir(exist_ok=True)
        
        log_file = log_dir / f"job_processor_{datetime.now().strftime('%Y%m%d')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        self.logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.running = False
    
    def scan_job_queue(self):
        """Scan for new job files in the queue directory"""
        try:
            job_files = list(self.queue_path.glob("*.json"))
            return sorted(job_files, key=lambda x: x.stat().st_mtime)
        except Exception as e:
            self.logger.error(f"Error scanning job queue: {e}")
            return []
    
    def validate_job_file(self, job_file):
        """Validate job file format and requirements"""
        try:
            with open(job_file, 'r') as f:
                job_data = json.load(f)
            
            required_fields = ['job_id', 'input_file', 'portfolio_size', 'job_type', 'timestamp']
            for field in required_fields:
                if field not in job_data:
                    raise ValueError(f"Missing required field: {field}")
            
            # Validate input file exists and is CSV
            input_file = self.samba_share_path / "input" / job_data['input_file']
            if not input_file.exists():
                raise FileNotFoundError(f"Input file not found: {job_data['input_file']}")
            
            if not job_data['input_file'].lower().endswith('.csv'):
                raise ValueError("Only CSV input files are supported in Samba-only architecture")
            
            # Validate portfolio size
            portfolio_size = int(job_data['portfolio_size'])
            if not (10 <= portfolio_size <= 100):
                raise ValueError(f"Portfolio size must be between 10-100, got: {portfolio_size}")
            
            return job_data
            
        except Exception as e:
            self.logger.error(f"Job validation failed for {job_file}: {e}")
            return None
    
    def move_job_file(self, job_file, destination_dir):
        """Move job file to specified directory"""
        try:
            destination = destination_dir / job_file.name
            job_file.rename(destination)
            return destination
        except Exception as e:
            self.logger.error(f"Failed to move job file {job_file} to {destination_dir}: {e}")
            return None
    
    def execute_optimization_job(self, job_data, job_file):
        """Execute optimization job with GPU acceleration"""
        job_id = job_data['job_id']
        
        try:
            self.logger.info(f"Starting optimization job {job_id}")
            
            # Move job to processing directory
            processing_job_file = self.move_job_file(job_file, self.processing_path)
            if not processing_job_file:
                return False
            
            # Prepare execution environment
            backend_script = self.backend_path / "parquet_cudf_workflow.py"
            input_file = self.samba_share_path / "input" / job_data['input_file']
            portfolio_size = job_data['portfolio_size']
            
            # Set up environment for GPU acceleration
            env = os.environ.copy()
            env['PYTHONPATH'] = str(self.backend_path / "lib")
            env['OPTIMIZER_HOME'] = str(self.backend_path)
            
            # Execute optimization with server-side GPU acceleration
            cmd = [
                sys.executable,
                str(backend_script),
                '--input', str(input_file),
                '--portfolio-size', str(portfolio_size)
            ]
            
            self.logger.info(f"Executing: {' '.join(cmd)}")
            
            # Execute with timeout and capture output
            start_time = time.time()
            result = subprocess.run(
                cmd,
                cwd=str(self.backend_path),
                env=env,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            execution_time = time.time() - start_time
            
            # Update job data with results
            job_data['execution_time'] = execution_time
            job_data['completion_timestamp'] = datetime.now().isoformat()
            job_data['return_code'] = result.returncode
            job_data['stdout'] = result.stdout
            job_data['stderr'] = result.stderr
            
            if result.returncode == 0:
                self.logger.info(f"Job {job_id} completed successfully in {execution_time:.2f}s")
                
                # Move to completed directory
                completed_job_file = self.completed_path / processing_job_file.name
                with open(completed_job_file, 'w') as f:
                    json.dump(job_data, f, indent=2)
                
                processing_job_file.unlink()  # Remove from processing
                return True
            else:
                self.logger.error(f"Job {job_id} failed with return code {result.returncode}")
                self.logger.error(f"STDERR: {result.stderr}")
                
                # Move to failed directory
                failed_job_file = self.failed_path / processing_job_file.name
                with open(failed_job_file, 'w') as f:
                    json.dump(job_data, f, indent=2)
                
                processing_job_file.unlink()  # Remove from processing
                return False
                
        except subprocess.TimeoutExpired:
            self.logger.error(f"Job {job_id} timed out after 5 minutes")
            job_data['error'] = "Job execution timed out"
            job_data['completion_timestamp'] = datetime.now().isoformat()
            
            # Move to failed directory
            failed_job_file = self.failed_path / f"{job_id}_timeout.json"
            with open(failed_job_file, 'w') as f:
                json.dump(job_data, f, indent=2)
            
            return False
            
        except Exception as e:
            self.logger.error(f"Unexpected error executing job {job_id}: {e}")
            job_data['error'] = str(e)
            job_data['completion_timestamp'] = datetime.now().isoformat()
            
            # Move to failed directory
            failed_job_file = self.failed_path / f"{job_id}_error.json"
            with open(failed_job_file, 'w') as f:
                json.dump(job_data, f, indent=2)
            
            return False
    
    def process_job_queue(self):
        """Main job processing loop"""
        self.logger.info("Starting job queue processing loop")
        
        while self.running:
            try:
                # Scan for new jobs
                job_files = self.scan_job_queue()
                
                if not job_files:
                    time.sleep(2)  # Wait 2 seconds before next scan
                    continue
                
                for job_file in job_files:
                    if not self.running:
                        break
                    
                    self.logger.info(f"Processing job file: {job_file.name}")
                    
                    # Validate job file
                    job_data = self.validate_job_file(job_file)
                    if not job_data:
                        # Move invalid job to failed directory
                        self.move_job_file(job_file, self.failed_path)
                        continue
                    
                    # Execute job with GPU acceleration
                    success = self.execute_optimization_job(job_data, job_file)
                    
                    if success:
                        self.logger.info(f"Job {job_data['job_id']} completed successfully")
                    else:
                        self.logger.error(f"Job {job_data['job_id']} failed")
                
            except Exception as e:
                self.logger.error(f"Error in job processing loop: {e}")
                time.sleep(5)  # Wait longer on error
        
        self.logger.info("Job queue processor stopped")
    
    def start(self):
        """Start the job queue processor"""
        self.logger.info("Starting Samba Job Queue Processor for GPU acceleration")
        self.logger.info(f"Backend path: {self.backend_path}")
        self.logger.info(f"Jobs path: {self.jobs_path}")
        
        try:
            self.process_job_queue()
        except KeyboardInterrupt:
            self.logger.info("Received keyboard interrupt, shutting down...")
        except Exception as e:
            self.logger.error(f"Fatal error in job processor: {e}")
        finally:
            self.logger.info("Samba Job Queue Processor stopped")

def main():
    """Main entry point"""
    processor = SambaJobQueueProcessor()
    processor.start()

if __name__ == "__main__":
    main()
