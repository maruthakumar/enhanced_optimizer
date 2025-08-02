#!/usr/bin/env python3
"""
Heavy Optimizer Platform - Story-Compliant Samba Job Queue Processor
Server-side daemon that meets all story requirements:
- Monitors input directory for CSV files
- Maintains complete status history
- Supports concurrent job processing
- Integrates with Pipeline Orchestrator
"""

import os
import sys
import time
import json
import logging
import subprocess
import threading
import signal
import uuid
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Any
import hashlib

class PipelineOrchestrator:
    """Pipeline Orchestrator interface for job queuing"""
    def __init__(self, backend_path: Path):
        self.backend_path = backend_path
        self.workflow_script = backend_path / "csv_only_heavydb_workflow.py"
        self.logger = logging.getLogger(__name__)
    
    def queue_job(self, job_data: Dict[str, Any]) -> subprocess.Popen:
        """Queue job for processing and return process handle"""
        input_file = job_data['input_file']
        portfolio_size = job_data['portfolio_size']
        output_dir = job_data.get('output_dir', '/mnt/optimizer_share/output')
        
        # Set up environment for HeavyDB acceleration
        env = os.environ.copy()
        env['PYTHONPATH'] = str(self.backend_path / "lib")
        env['HEAVYDB_OPTIMIZER_HOME'] = str(self.backend_path)
        
        # Execute optimization with server-side HeavyDB acceleration
        cmd = [
            sys.executable,
            str(self.workflow_script),
            "--input", str(input_file),
            "--portfolio-size", str(portfolio_size),
            "--output-dir", str(output_dir)
        ]
        
        self.logger.info(f"Orchestrator queuing job: {' '.join(cmd)}")
        
        # Start process asynchronously
        process = subprocess.Popen(
            cmd,
            cwd=str(self.backend_path),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        return process

class StoryCompliantSambaJobQueueProcessor:
    def __init__(self, samba_share_path="/mnt/optimizer_share", max_concurrent_jobs=4):
        self.samba_share_path = Path(samba_share_path)
        self.backend_path = self.samba_share_path / "backend"
        self.input_path = self.samba_share_path / "input"
        self.jobs_path = self.samba_share_path / "jobs"
        self.queue_path = self.jobs_path / "queue"
        self.processing_path = self.jobs_path / "processing"
        self.completed_path = self.jobs_path / "completed"
        self.failed_path = self.jobs_path / "failed"
        self.status_path = self.jobs_path / "status"
        
        # Ensure all directories exist
        for path in [self.queue_path, self.processing_path, self.completed_path, 
                     self.failed_path, self.status_path, self.input_path]:
            path.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Control flags and tracking
        self.running = True
        self.processing_jobs = {}
        self.max_concurrent_jobs = max_concurrent_jobs
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent_jobs)
        self.processed_files = set()  # Track processed input files
        self.status_history = []  # In-memory status history
        self.status_lock = threading.Lock()
        
        # Initialize Pipeline Orchestrator
        self.orchestrator = PipelineOrchestrator(self.backend_path)
        
        # Load existing status history
        self.load_status_history()
        
        # Setup signal handlers
        signal.signal(signal.SIGTERM, self.signal_handler)
        signal.signal(signal.SIGINT, self.signal_handler)
        
        self.logger.info("Story-Compliant Samba Job Queue Processor initialized")
        self.logger.info(f"Monitoring input directory: {self.input_path}")
        self.logger.info(f"Monitoring queue directory: {self.queue_path}")
        self.logger.info(f"Max concurrent jobs: {self.max_concurrent_jobs}")
    
    def setup_logging(self):
        """Setup comprehensive logging"""
        log_dir = self.samba_share_path / "logs"
        log_dir.mkdir(exist_ok=True)
        
        log_file = log_dir / f"story_compliant_job_processor_{datetime.now().strftime('%Y%m%d')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s',
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
        self.executor.shutdown(wait=True)
        self.save_status_history()
    
    def load_status_history(self):
        """Load status history from persistent storage"""
        history_file = self.status_path / "job_history.json"
        if history_file.exists():
            try:
                with open(history_file, 'r') as f:
                    self.status_history = json.load(f)
                self.logger.info(f"Loaded {len(self.status_history)} job history entries")
            except Exception as e:
                self.logger.error(f"Failed to load status history: {e}")
                self.status_history = []
        
        # Load processed files list
        processed_file = self.status_path / "processed_files.json"
        if processed_file.exists():
            try:
                with open(processed_file, 'r') as f:
                    self.processed_files = set(json.load(f))
                self.logger.info(f"Loaded {len(self.processed_files)} processed file entries")
            except Exception as e:
                self.logger.error(f"Failed to load processed files: {e}")
                self.processed_files = set()
    
    def save_status_history(self):
        """Save status history to persistent storage"""
        with self.status_lock:
            try:
                # Save job history
                history_file = self.status_path / "job_history.json"
                with open(history_file, 'w') as f:
                    json.dump(self.status_history, f, indent=2)
                
                # Save processed files
                processed_file = self.status_path / "processed_files.json"
                with open(processed_file, 'w') as f:
                    json.dump(list(self.processed_files), f, indent=2)
                
                # Save current status summary
                summary_file = self.status_path / "current_status.json"
                with open(summary_file, 'w') as f:
                    json.dump(self.get_status_summary(), f, indent=2)
                
                self.logger.info("Status history saved successfully")
            except Exception as e:
                self.logger.error(f"Failed to save status history: {e}")
    
    def add_status_entry(self, job_id: str, status: str, details: Dict[str, Any]):
        """Add entry to status history"""
        with self.status_lock:
            entry = {
                'job_id': job_id,
                'status': status,
                'timestamp': datetime.now().isoformat(),
                'details': details
            }
            self.status_history.append(entry)
            
            # Keep only last 1000 entries in memory
            if len(self.status_history) > 1000:
                self.status_history = self.status_history[-1000:]
            
            # Save periodically (every 10 updates)
            if len(self.status_history) % 10 == 0:
                self.save_status_history()
    
    def scan_input_directory(self):
        """Scan input directory for new CSV files"""
        try:
            csv_files = []
            for csv_file in self.input_path.glob("*.csv"):
                # Generate unique identifier for file
                file_stat = csv_file.stat()
                file_id = f"{csv_file.name}_{file_stat.st_size}_{file_stat.st_mtime}"
                
                if file_id not in self.processed_files:
                    csv_files.append(csv_file)
            
            return sorted(csv_files, key=lambda x: x.stat().st_mtime)
        except Exception as e:
            self.logger.error(f"Error scanning input directory: {e}")
            return []
    
    def create_job_from_csv(self, csv_file: Path) -> Optional[Path]:
        """Create job file from CSV submission"""
        try:
            # Generate job ID
            job_id = f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
            
            # Create job data
            job_data = {
                'job_id': job_id,
                'input_file': str(csv_file),
                'input_filename': csv_file.name,
                'portfolio_size': 35,  # Default portfolio size
                'job_type': 'csv_optimization',
                'timestamp': datetime.now().isoformat(),
                'source': 'input_directory_monitor',
                'file_size': csv_file.stat().st_size,
                'file_checksum': self.calculate_file_checksum(csv_file)
            }
            
            # Check if file has portfolio size hint in name
            # Format: filename_ps50.csv for portfolio size 50
            if '_ps' in csv_file.stem:
                try:
                    ps_part = csv_file.stem.split('_ps')[-1]
                    portfolio_size = int(''.join(filter(str.isdigit, ps_part)))
                    if 10 <= portfolio_size <= 100:
                        job_data['portfolio_size'] = portfolio_size
                        self.logger.info(f"Detected portfolio size {portfolio_size} from filename")
                except:
                    pass
            
            # Save job file to queue
            job_file = self.queue_path / f"{job_id}.json"
            with open(job_file, 'w') as f:
                json.dump(job_data, f, indent=2)
            
            # Mark file as processed
            file_stat = csv_file.stat()
            file_id = f"{csv_file.name}_{file_stat.st_size}_{file_stat.st_mtime}"
            self.processed_files.add(file_id)
            
            # Add status entry
            self.add_status_entry(job_id, 'queued', {
                'input_file': csv_file.name,
                'portfolio_size': job_data['portfolio_size'],
                'file_size': job_data['file_size']
            })
            
            self.logger.info(f"Created job {job_id} for CSV file: {csv_file.name}")
            return job_file
            
        except Exception as e:
            self.logger.error(f"Failed to create job from CSV {csv_file}: {e}")
            return None
    
    def calculate_file_checksum(self, file_path: Path) -> str:
        """Calculate MD5 checksum of file"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
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
            
            required_fields = ['job_id', 'timestamp']
            for field in required_fields:
                if field not in job_data:
                    raise ValueError(f"Missing required field: {field}")
            
            # Get input file path
            if 'input_file' in job_data:
                input_file = Path(job_data['input_file'])
            else:
                # Fallback for old format
                input_filename = job_data.get('input_filename', '')
                input_file = self.input_path / input_filename
                job_data['input_file'] = str(input_file)
            
            if not input_file.exists():
                raise FileNotFoundError(f"Input file not found: {input_file}")
            
            if not str(input_file).lower().endswith('.csv'):
                raise ValueError("Only CSV input files are supported")
            
            # Validate portfolio size
            portfolio_size = int(job_data.get('portfolio_size', 35))
            if not (10 <= portfolio_size <= 100):
                raise ValueError(f"Portfolio size must be between 10-100, got: {portfolio_size}")
            
            job_data['portfolio_size'] = portfolio_size
            return job_data
            
        except Exception as e:
            self.logger.error(f"Job validation failed for {job_file}: {e}")
            return None
    
    def execute_optimization_job(self, job_data: Dict[str, Any], job_file: Path) -> bool:
        """Execute optimization job using Pipeline Orchestrator"""
        job_id = job_data['job_id']
        
        try:
            self.logger.info(f"Starting optimization job {job_id}")
            
            # Move job to processing directory
            processing_job_file = self.processing_path / job_file.name
            job_file.rename(processing_job_file)
            
            # Update status
            self.add_status_entry(job_id, 'processing', {
                'start_time': datetime.now().isoformat(),
                'thread': threading.current_thread().name
            })
            
            # Queue job with Pipeline Orchestrator
            start_time = time.time()
            process = self.orchestrator.queue_job(job_data)
            
            # Wait for completion with timeout
            try:
                stdout, stderr = process.communicate(timeout=300)  # 5 minute timeout
                return_code = process.returncode
                execution_time = time.time() - start_time
                
                # Update job data with results
                job_data['execution_time'] = execution_time
                job_data['completion_timestamp'] = datetime.now().isoformat()
                job_data['return_code'] = return_code
                job_data['stdout'] = stdout
                job_data['stderr'] = stderr
                
                if return_code == 0:
                    self.logger.info(f"Job {job_id} completed successfully in {execution_time:.2f}s")
                    
                    # Move to completed directory
                    completed_job_file = self.completed_path / processing_job_file.name
                    with open(completed_job_file, 'w') as f:
                        json.dump(job_data, f, indent=2)
                    
                    processing_job_file.unlink()
                    
                    # Update status
                    self.add_status_entry(job_id, 'completed', {
                        'execution_time': execution_time,
                        'output_files': self.find_output_files(job_id, stdout)
                    })
                    
                    return True
                else:
                    self.logger.error(f"Job {job_id} failed with return code {return_code}")
                    self.handle_failed_job(job_id, job_data, processing_job_file, 
                                         f"Process returned code {return_code}")
                    return False
                    
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                self.logger.error(f"Job {job_id} timed out after 5 minutes")
                self.handle_failed_job(job_id, job_data, processing_job_file, "Execution timeout")
                return False
                
        except Exception as e:
            self.logger.error(f"Unexpected error executing job {job_id}: {e}")
            self.handle_failed_job(job_id, job_data, processing_job_file, str(e))
            return False
    
    def handle_failed_job(self, job_id: str, job_data: Dict[str, Any], 
                         processing_job_file: Path, error_message: str):
        """Handle failed job"""
        job_data['error'] = error_message
        job_data['completion_timestamp'] = datetime.now().isoformat()
        
        # Move to failed directory
        failed_job_file = self.failed_path / processing_job_file.name
        with open(failed_job_file, 'w') as f:
            json.dump(job_data, f, indent=2)
        
        if processing_job_file.exists():
            processing_job_file.unlink()
        
        # Update status
        self.add_status_entry(job_id, 'failed', {
            'error': error_message,
            'completion_time': datetime.now().isoformat()
        })
    
    def find_output_files(self, job_id: str, stdout: str) -> List[str]:
        """Find output files generated by job"""
        output_files = []
        output_dir = self.samba_share_path / "output"
        
        # Extract timestamp from stdout if available
        timestamp_pattern = r"run_(\d{8}_\d{6})"
        import re
        matches = re.findall(timestamp_pattern, stdout)
        
        if matches:
            # Look for files with the extracted timestamp
            for timestamp in matches:
                for file in output_dir.glob(f"*{timestamp}*"):
                    output_files.append(str(file.relative_to(self.samba_share_path)))
        else:
            # Fallback: look for recent files
            job_time = datetime.now()
            for file in output_dir.iterdir():
                if file.is_file():
                    file_time = datetime.fromtimestamp(file.stat().st_mtime)
                    if (job_time - file_time).total_seconds() < 600:  # Files created within 10 minutes
                        output_files.append(str(file.relative_to(self.samba_share_path)))
        
        return output_files
    
    def process_jobs_concurrently(self):
        """Process jobs concurrently using thread pool"""
        active_futures = {}
        
        while self.running:
            try:
                # Check for new CSV files in input directory
                csv_files = self.scan_input_directory()
                for csv_file in csv_files[:self.max_concurrent_jobs - len(active_futures)]:
                    job_file = self.create_job_from_csv(csv_file)
                    if job_file:
                        self.logger.info(f"Created job from input CSV: {csv_file.name}")
                
                # Check for queued jobs
                if len(active_futures) < self.max_concurrent_jobs:
                    job_files = self.scan_job_queue()
                    
                    for job_file in job_files[:self.max_concurrent_jobs - len(active_futures)]:
                        if not self.running:
                            break
                        
                        # Validate job
                        job_data = self.validate_job_file(job_file)
                        if not job_data:
                            self.move_job_file(job_file, self.failed_path)
                            self.add_status_entry(job_file.stem, 'failed', {
                                'error': 'Invalid job file'
                            })
                            continue
                        
                        # Submit job for concurrent execution
                        future = self.executor.submit(
                            self.execute_optimization_job, job_data, job_file
                        )
                        active_futures[future] = job_data['job_id']
                        self.logger.info(f"Submitted job {job_data['job_id']} for processing")
                
                # Check completed futures
                if active_futures:
                    done_futures = []
                    for future in as_completed(active_futures, timeout=0.1):
                        job_id = active_futures[future]
                        try:
                            success = future.result()
                            if success:
                                self.logger.info(f"Job {job_id} completed successfully")
                            else:
                                self.logger.error(f"Job {job_id} failed")
                        except Exception as e:
                            self.logger.error(f"Job {job_id} raised exception: {e}")
                        done_futures.append(future)
                    
                    # Remove completed futures
                    for future in done_futures:
                        del active_futures[future]
                
                # Save status history periodically
                if len(self.status_history) % 5 == 0:
                    self.save_status_history()
                
                # Brief sleep to prevent CPU spinning
                time.sleep(2)
                
            except Exception as e:
                self.logger.error(f"Error in concurrent processing loop: {e}")
                time.sleep(5)
        
        # Wait for remaining jobs to complete
        if active_futures:
            self.logger.info(f"Waiting for {len(active_futures)} jobs to complete...")
            for future in as_completed(active_futures):
                job_id = active_futures[future]
                try:
                    future.result()
                except Exception as e:
                    self.logger.error(f"Job {job_id} failed during shutdown: {e}")
    
    def move_job_file(self, job_file: Path, destination_dir: Path) -> Optional[Path]:
        """Move job file to specified directory"""
        try:
            destination = destination_dir / job_file.name
            job_file.rename(destination)
            return destination
        except Exception as e:
            self.logger.error(f"Failed to move job file {job_file} to {destination_dir}: {e}")
            return None
    
    def get_status_summary(self) -> Dict[str, Any]:
        """Get summary of job processing status"""
        with self.status_lock:
            summary = {
                'total_jobs': len(self.status_history),
                'queued': len(list(self.queue_path.glob("*.json"))),
                'processing': len(list(self.processing_path.glob("*.json"))),
                'completed': len(list(self.completed_path.glob("*.json"))),
                'failed': len(list(self.failed_path.glob("*.json"))),
                'processed_files': len(self.processed_files),
                'max_concurrent_jobs': self.max_concurrent_jobs,
                'last_update': datetime.now().isoformat()
            }
            
            # Calculate success rate
            if summary['total_jobs'] > 0:
                completed_count = sum(1 for entry in self.status_history 
                                    if entry['status'] == 'completed')
                summary['success_rate'] = (completed_count / summary['total_jobs']) * 100
            else:
                summary['success_rate'] = 0
            
            # Get recent jobs
            recent_jobs = sorted(self.status_history, 
                               key=lambda x: x['timestamp'], 
                               reverse=True)[:10]
            summary['recent_jobs'] = recent_jobs
            
            return summary
    
    def start(self):
        """Start the story-compliant job queue processor"""
        self.logger.info("Starting Story-Compliant Samba Job Queue Processor")
        self.logger.info(f"Backend path: {self.backend_path}")
        self.logger.info(f"Input path: {self.input_path}")
        self.logger.info(f"Jobs path: {self.jobs_path}")
        self.logger.info(f"Status tracking enabled with persistent history")
        
        # Write initial status summary
        self.save_status_history()
        
        try:
            self.process_jobs_concurrently()
        except KeyboardInterrupt:
            self.logger.info("Received keyboard interrupt, shutting down...")
        except Exception as e:
            self.logger.error(f"Fatal error in job processor: {e}")
        finally:
            self.save_status_history()
            self.logger.info("Story-Compliant Samba Job Queue Processor stopped")

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Story-Compliant Samba Job Queue Processor')
    parser.add_argument('--max-jobs', type=int, default=4, 
                       help='Maximum concurrent jobs (default: 4)')
    parser.add_argument('--share-path', default='/mnt/optimizer_share',
                       help='Samba share mount path')
    
    args = parser.parse_args()
    
    processor = StoryCompliantSambaJobQueueProcessor(
        samba_share_path=args.share_path,
        max_concurrent_jobs=args.max_jobs
    )
    processor.start()

if __name__ == "__main__":
    main()