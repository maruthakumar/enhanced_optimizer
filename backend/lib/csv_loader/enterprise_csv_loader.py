#!/usr/bin/env python3
"""
Enterprise CSV Loader for Heavy Optimizer Platform

This module provides efficient CSV data loading with all enterprise features:
- Streaming for large files
- Batch inserts
- Progress tracking
- Data validation
- GPU memory monitoring
- Audit trail
- Interrupted load recovery
"""

import os
import time
import json
import logging
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Iterator, Any, Callable
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
from tqdm import tqdm
import psutil
import threading
import queue


@dataclass
class LoadMetrics:
    """Metrics for a CSV load operation"""
    file_path: str
    file_size_mb: float
    total_rows: int
    total_columns: int
    chunks_processed: int
    load_time_seconds: float
    throughput_mbps: float
    peak_memory_mb: float
    validation_errors: int
    strategies_validated: int
    gpu_memory_used_mb: float = 0.0
    status: str = "SUCCESS"
    error_message: Optional[str] = None
    

@dataclass
class ValidationResult:
    """Result of data validation"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    numeric_columns: List[str]
    datetime_columns: List[str]
    missing_value_count: int
    duplicate_strategies: List[str]


@dataclass
class LoadCheckpoint:
    """Checkpoint for resumable loads"""
    file_path: str
    file_hash: str
    total_size: int
    bytes_processed: int
    rows_processed: int
    chunks_completed: int
    timestamp: str
    metrics: Optional[LoadMetrics] = None


class EnterpriseCSVLoader:
    """Enterprise-grade CSV loader with all required features"""
    
    def __init__(self, 
                 chunk_size: int = 10000,
                 batch_size: int = 1000,
                 enable_gpu_monitoring: bool = True,
                 checkpoint_dir: str = "/tmp/csv_loader_checkpoints",
                 audit_log_path: str = "/mnt/optimizer_share/logs/csv_loader_audit.json"):
        """
        Initialize the enterprise CSV loader
        
        Args:
            chunk_size: Number of rows to read per chunk
            batch_size: Number of rows to insert per batch
            enable_gpu_monitoring: Whether to monitor GPU memory
            checkpoint_dir: Directory for checkpoint files
            audit_log_path: Path to audit log file
        """
        self.chunk_size = chunk_size
        self.batch_size = batch_size
        self.enable_gpu_monitoring = enable_gpu_monitoring
        self.checkpoint_dir = Path(checkpoint_dir)
        self.audit_log_path = Path(audit_log_path)
        
        # Create directories
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Memory monitoring
        self.peak_memory = 0
        self.memory_monitor_thread = None
        self.monitoring = False
        
        # Progress tracking
        self.progress_callback = None
        
        # GPU monitoring (if available)
        self.gpu_available = self._check_gpu_availability()
        
    def _check_gpu_availability(self) -> bool:
        """Check if GPU monitoring is available"""
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            return len(gpus) > 0 and self.enable_gpu_monitoring
        except ImportError:
            self.logger.warning("GPUtil not available. GPU monitoring disabled.")
            return False
    
    def _get_file_hash(self, file_path: str, sample_size: int = 1024*1024) -> str:
        """Get hash of file (first MB for performance)"""
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            hasher.update(f.read(sample_size))
        return hasher.hexdigest()
    
    def _monitor_memory(self):
        """Monitor memory usage in background thread"""
        process = psutil.Process()
        while self.monitoring:
            memory_mb = process.memory_info().rss / 1024 / 1024
            self.peak_memory = max(self.peak_memory, memory_mb)
            time.sleep(0.1)
    
    def _start_memory_monitoring(self):
        """Start memory monitoring thread"""
        self.monitoring = True
        self.peak_memory = 0
        self.memory_monitor_thread = threading.Thread(target=self._monitor_memory)
        self.memory_monitor_thread.daemon = True
        self.memory_monitor_thread.start()
    
    def _stop_memory_monitoring(self):
        """Stop memory monitoring thread"""
        self.monitoring = False
        if self.memory_monitor_thread:
            self.memory_monitor_thread.join(timeout=1)
    
    def _get_gpu_memory_usage(self) -> float:
        """Get current GPU memory usage in MB"""
        if not self.gpu_available:
            return 0.0
        
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                return gpus[0].memoryUsed  # MB
            return 0.0
        except:
            return 0.0
    
    def load_csv_streaming(self, 
                          file_path: str,
                          table_name: str,
                          data_handler: Optional[Callable] = None,
                          resume_from_checkpoint: bool = True,
                          validate_data: bool = True,
                          fill_missing_values: bool = True,
                          missing_fill_value: float = 0.0,
                          progress_callback: Optional[Callable] = None) -> LoadMetrics:
        """
        Load CSV file using streaming with all enterprise features
        
        Args:
            file_path: Path to CSV file
            table_name: Target table name
            data_handler: Optional function to handle each chunk (e.g., insert to DB)
            resume_from_checkpoint: Whether to resume from checkpoint if available
            validate_data: Whether to validate data
            fill_missing_values: Whether to fill missing values
            missing_fill_value: Value to use for missing data
            progress_callback: Function to call with progress updates
            
        Returns:
            LoadMetrics object with load statistics
        """
        start_time = time.time()
        self.progress_callback = progress_callback
        
        # Start memory monitoring
        self._start_memory_monitoring()
        
        # Get file info
        file_size = os.path.getsize(file_path)
        file_size_mb = file_size / 1024 / 1024
        file_hash = self._get_file_hash(file_path)
        
        # Check for checkpoint
        checkpoint = None
        if resume_from_checkpoint:
            checkpoint = self._load_checkpoint(file_path, file_hash)
            if checkpoint:
                self.logger.info(f"Resuming from checkpoint: {checkpoint.rows_processed} rows processed")
        
        # Initialize metrics
        total_rows = 0
        total_columns = 0
        chunks_processed = 0
        validation_errors = 0
        strategies_validated = set()
        
        # Create progress bar
        pbar = tqdm(total=file_size, unit='B', unit_scale=True, 
                   desc=f"Loading {Path(file_path).name}")
        
        if checkpoint:
            pbar.update(checkpoint.bytes_processed)
        
        try:
            # Open file for streaming
            skiprows = checkpoint.rows_processed if checkpoint else 0
            
            # Read file in chunks
            chunk_iterator = pd.read_csv(
                file_path,
                chunksize=self.chunk_size,
                skiprows=skiprows,
                low_memory=True,  # Enable memory optimization
                parse_dates=True,
                infer_datetime_format=True
            )
            
            for chunk_idx, chunk in enumerate(chunk_iterator):
                # Update progress
                bytes_read = min((skiprows + total_rows + len(chunk)) * 
                               (file_size / self._estimate_total_rows(file_path)), file_size)
                pbar.update(bytes_read - pbar.n)
                
                # Validate chunk if requested
                if validate_data:
                    validation_result = self._validate_chunk(chunk)
                    validation_errors += len(validation_result.errors)
                    
                    if not validation_result.is_valid:
                        self.logger.warning(f"Validation errors in chunk {chunk_idx}: {validation_result.errors}")
                    
                    # Track unique strategies
                    if 'strategy_name' in chunk.columns:
                        strategies_validated.update(chunk['strategy_name'].unique())
                
                # Handle missing values
                if fill_missing_values:
                    chunk = self._handle_missing_values(chunk, missing_fill_value)
                
                # Process chunk in batches
                if data_handler:
                    self._process_chunk_in_batches(chunk, table_name, data_handler)
                
                # Update metrics
                total_rows += len(chunk)
                total_columns = len(chunk.columns)
                chunks_processed += 1
                
                # Save checkpoint periodically
                if chunks_processed % 10 == 0:
                    self._save_checkpoint(
                        file_path, file_hash, file_size,
                        bytes_read, skiprows + total_rows, chunks_processed
                    )
                
                # Report progress
                if self.progress_callback:
                    self.progress_callback({
                        'rows_processed': skiprows + total_rows,
                        'chunks_processed': chunks_processed,
                        'percentage': (bytes_read / file_size) * 100
                    })
            
            # Close progress bar
            pbar.close()
            
            # Stop memory monitoring
            self._stop_memory_monitoring()
            
            # Calculate metrics
            load_time = time.time() - start_time
            throughput = file_size_mb / load_time if load_time > 0 else 0
            
            # Create final metrics
            metrics = LoadMetrics(
                file_path=file_path,
                file_size_mb=file_size_mb,
                total_rows=skiprows + total_rows if checkpoint else total_rows,
                total_columns=total_columns,
                chunks_processed=chunks_processed,
                load_time_seconds=load_time,
                throughput_mbps=throughput,
                peak_memory_mb=self.peak_memory,
                validation_errors=validation_errors,
                strategies_validated=len(strategies_validated),
                gpu_memory_used_mb=self._get_gpu_memory_usage(),
                status="SUCCESS"
            )
            
            # Clean up checkpoint
            self._remove_checkpoint(file_path)
            
            # Log to audit trail
            self._log_to_audit(metrics)
            
            self.logger.info(f"CSV load completed: {total_rows} rows in {load_time:.2f}s "
                           f"({throughput:.2f} MB/s)")
            
            return metrics
            
        except Exception as e:
            # Stop memory monitoring
            self._stop_memory_monitoring()
            pbar.close()
            
            # Create error metrics
            metrics = LoadMetrics(
                file_path=file_path,
                file_size_mb=file_size_mb,
                total_rows=total_rows,
                total_columns=total_columns,
                chunks_processed=chunks_processed,
                load_time_seconds=time.time() - start_time,
                throughput_mbps=0,
                peak_memory_mb=self.peak_memory,
                validation_errors=validation_errors,
                strategies_validated=len(strategies_validated),
                status="FAILED",
                error_message=str(e)
            )
            
            # Log to audit trail
            self._log_to_audit(metrics)
            
            self.logger.error(f"CSV load failed: {e}")
            raise
    
    def _estimate_total_rows(self, file_path: str, sample_size: int = 1000) -> int:
        """Estimate total rows in file by sampling"""
        try:
            # Read sample
            sample = pd.read_csv(file_path, nrows=sample_size)
            
            # Estimate bytes per row
            sample_bytes = sample.memory_usage(deep=True).sum()
            bytes_per_row = sample_bytes / len(sample)
            
            # Estimate total rows
            file_size = os.path.getsize(file_path)
            estimated_rows = int(file_size / bytes_per_row)
            
            return estimated_rows
        except:
            return 0
    
    def _validate_chunk(self, chunk: pd.DataFrame) -> ValidationResult:
        """Validate a data chunk"""
        errors = []
        warnings = []
        
        # Check for numeric columns
        numeric_columns = chunk.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_columns) == 0:
            errors.append("No numeric columns found")
        
        # Check for datetime columns
        datetime_columns = chunk.select_dtypes(include=['datetime']).columns.tolist()
        
        # Check for missing values
        missing_count = chunk.isnull().sum().sum()
        if missing_count > 0:
            warnings.append(f"Found {missing_count} missing values")
        
        # Validate numeric data
        for col in numeric_columns:
            if chunk[col].dtype == 'object':
                errors.append(f"Column {col} contains non-numeric data")
            
            # Check for infinite values
            if np.isinf(chunk[col]).any():
                errors.append(f"Column {col} contains infinite values")
        
        # Check for duplicate strategy names
        duplicate_strategies = []
        if 'strategy_name' in chunk.columns:
            duplicates = chunk[chunk['strategy_name'].duplicated()]['strategy_name'].tolist()
            if duplicates:
                duplicate_strategies = list(set(duplicates))
                warnings.append(f"Found {len(duplicate_strategies)} duplicate strategy names")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            numeric_columns=numeric_columns,
            datetime_columns=datetime_columns,
            missing_value_count=missing_count,
            duplicate_strategies=duplicate_strategies
        )
    
    def _handle_missing_values(self, chunk: pd.DataFrame, fill_value: float) -> pd.DataFrame:
        """Handle missing values in chunk"""
        # Fill numeric columns
        numeric_columns = chunk.select_dtypes(include=[np.number]).columns
        chunk[numeric_columns] = chunk[numeric_columns].fillna(fill_value)
        
        # Forward fill datetime columns
        datetime_columns = chunk.select_dtypes(include=['datetime']).columns
        if len(datetime_columns) > 0:
            chunk[datetime_columns] = chunk[datetime_columns].fillna(method='ffill')
        
        # Fill string columns
        string_columns = chunk.select_dtypes(include=['object']).columns
        chunk[string_columns] = chunk[string_columns].fillna('')
        
        return chunk
    
    def _process_chunk_in_batches(self, chunk: pd.DataFrame, table_name: str, 
                                 data_handler: Callable):
        """Process chunk in batches"""
        num_rows = len(chunk)
        
        for i in range(0, num_rows, self.batch_size):
            batch = chunk.iloc[i:i + self.batch_size]
            
            # Call data handler (e.g., insert to database)
            data_handler(table_name, batch)
            
            # Monitor GPU memory if transferring to GPU
            if self.gpu_available:
                gpu_memory = self._get_gpu_memory_usage()
                if gpu_memory > 20000:  # 20GB threshold
                    self.logger.warning(f"High GPU memory usage: {gpu_memory:.1f}MB")
    
    def _save_checkpoint(self, file_path: str, file_hash: str, total_size: int,
                        bytes_processed: int, rows_processed: int, 
                        chunks_completed: int):
        """Save checkpoint for resumable loads"""
        checkpoint = LoadCheckpoint(
            file_path=file_path,
            file_hash=file_hash,
            total_size=total_size,
            bytes_processed=bytes_processed,
            rows_processed=rows_processed,
            chunks_completed=chunks_completed,
            timestamp=datetime.now().isoformat()
        )
        
        checkpoint_file = self.checkpoint_dir / f"{Path(file_path).name}_{file_hash}.checkpoint"
        
        with open(checkpoint_file, 'w') as f:
            json.dump(asdict(checkpoint), f, indent=2)
    
    def _load_checkpoint(self, file_path: str, file_hash: str) -> Optional[LoadCheckpoint]:
        """Load checkpoint if available"""
        checkpoint_file = self.checkpoint_dir / f"{Path(file_path).name}_{file_hash}.checkpoint"
        
        if checkpoint_file.exists():
            try:
                with open(checkpoint_file, 'r') as f:
                    data = json.load(f)
                    return LoadCheckpoint(**data)
            except Exception as e:
                self.logger.warning(f"Failed to load checkpoint: {e}")
        
        return None
    
    def _remove_checkpoint(self, file_path: str):
        """Remove checkpoint file"""
        checkpoint_files = self.checkpoint_dir.glob(f"{Path(file_path).name}_*.checkpoint")
        for checkpoint_file in checkpoint_files:
            try:
                checkpoint_file.unlink()
            except:
                pass
    
    def _log_to_audit(self, metrics: LoadMetrics):
        """Log metrics to audit trail"""
        audit_entry = {
            'timestamp': datetime.now().isoformat(),
            'metrics': asdict(metrics)
        }
        
        # Append to audit log
        try:
            # Read existing log
            audit_log = []
            if self.audit_log_path.exists():
                with open(self.audit_log_path, 'r') as f:
                    audit_log = json.load(f)
            
            # Append new entry
            audit_log.append(audit_entry)
            
            # Keep last 1000 entries
            if len(audit_log) > 1000:
                audit_log = audit_log[-1000:]
            
            # Write back
            with open(self.audit_log_path, 'w') as f:
                json.dump(audit_log, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to write audit log: {e}")
    
    def get_audit_trail(self, limit: int = 100) -> List[Dict]:
        """Get recent audit trail entries"""
        try:
            if self.audit_log_path.exists():
                with open(self.audit_log_path, 'r') as f:
                    audit_log = json.load(f)
                    return audit_log[-limit:]
            return []
        except:
            return []
    
    def validate_csv_file(self, file_path: str, sample_size: int = 1000) -> ValidationResult:
        """Validate CSV file before loading"""
        try:
            # Read sample
            sample = pd.read_csv(file_path, nrows=sample_size)
            
            # Validate sample
            return self._validate_chunk(sample)
            
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                errors=[f"Failed to read file: {str(e)}"],
                warnings=[],
                numeric_columns=[],
                datetime_columns=[],
                missing_value_count=0,
                duplicate_strategies=[]
            )