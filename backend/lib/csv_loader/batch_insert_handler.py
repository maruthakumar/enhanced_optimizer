#!/usr/bin/env python3
"""
Batch Insert Handler for Heavy Optimizer Platform

Handles efficient batch inserts to HeavyDB with GPU memory management.
"""

import logging
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
import threading
import queue


@dataclass
class BatchInsertMetrics:
    """Metrics for batch insert operations"""
    table_name: str
    total_rows_inserted: int
    total_batches: int
    total_time_seconds: float
    rows_per_second: float
    gpu_memory_peak_mb: float
    failed_batches: int
    retry_count: int


class BatchInsertHandler:
    """Handles batch inserts with GPU memory management"""
    
    def __init__(self,
                 connection: Optional[Any] = None,
                 batch_size: int = 1000,
                 max_retries: int = 3,
                 gpu_memory_threshold_mb: float = 20000,
                 use_gpu_transfer: bool = True):
        """
        Initialize batch insert handler
        
        Args:
            connection: Database connection (HeavyDB)
            batch_size: Number of rows per batch
            max_retries: Maximum retry attempts for failed batches
            gpu_memory_threshold_mb: GPU memory threshold for chunking
            use_gpu_transfer: Whether to use GPU-accelerated transfers
        """
        self.connection = connection
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.gpu_memory_threshold_mb = gpu_memory_threshold_mb
        self.use_gpu_transfer = use_gpu_transfer
        
        self.logger = logging.getLogger(__name__)
        
        # Metrics
        self.metrics = {}
        
        # GPU monitoring
        self.gpu_available = self._check_gpu_availability()
        self.gpu_memory_monitor = None
        
        # Batch queue for async processing
        self.batch_queue = queue.Queue(maxsize=10)
        self.processing_thread = None
        self.is_processing = False
        
    def _check_gpu_availability(self) -> bool:
        """Check if GPU is available for transfers"""
        if not self.use_gpu_transfer:
            return False
            
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            return len(gpus) > 0
        except ImportError:
            self.logger.warning("GPUtil not available. GPU transfers disabled.")
            return False
    
    def _get_gpu_memory_usage(self) -> float:
        """Get current GPU memory usage in MB"""
        if not self.gpu_available:
            return 0.0
            
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                return gpus[0].memoryUsed
            return 0.0
        except:
            return 0.0
    
    def _monitor_gpu_memory(self) -> float:
        """Monitor and return peak GPU memory usage"""
        if not self.gpu_available:
            return 0.0
            
        peak_memory = 0.0
        current_memory = self._get_gpu_memory_usage()
        peak_memory = max(peak_memory, current_memory)
        
        return peak_memory
    
    def insert_batch(self, table_name: str, batch_data: pd.DataFrame,
                    retry_count: int = 0) -> bool:
        """
        Insert a single batch of data
        
        Args:
            table_name: Target table name
            batch_data: DataFrame containing batch data
            retry_count: Current retry attempt
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.connection is None:
                # If no HeavyDB connection, simulate insert
                time.sleep(0.001 * len(batch_data))  # Simulate insert time
                return True
            
            # Check GPU memory before insert
            if self.gpu_available:
                current_gpu_memory = self._get_gpu_memory_usage()
                if current_gpu_memory > self.gpu_memory_threshold_mb:
                    self.logger.warning(f"GPU memory high ({current_gpu_memory:.1f}MB), "
                                      f"waiting before insert...")
                    time.sleep(1)  # Wait for memory to free up
            
            # Convert DataFrame to appropriate format for HeavyDB
            # Note: Actual implementation would use pymapd or similar
            if hasattr(self.connection, 'load_table_columnar'):
                # HeavyDB columnar insert
                self.connection.load_table_columnar(
                    table_name,
                    batch_data
                )
            else:
                # Fallback to row-wise insert
                for _, row in batch_data.iterrows():
                    # This is inefficient and should be avoided in production
                    self._insert_row(table_name, row)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Batch insert failed: {e}")
            
            if retry_count < self.max_retries:
                self.logger.info(f"Retrying batch insert (attempt {retry_count + 1})")
                time.sleep(2 ** retry_count)  # Exponential backoff
                return self.insert_batch(table_name, batch_data, retry_count + 1)
            
            return False
    
    def _insert_row(self, table_name: str, row: pd.Series):
        """Insert a single row (fallback method)"""
        # This is a placeholder for actual row insert
        # In production, use proper SQL or HeavyDB API
        pass
    
    def insert_dataframe_in_batches(self, table_name: str, df: pd.DataFrame,
                                   progress_callback: Optional[Callable] = None) -> BatchInsertMetrics:
        """
        Insert entire DataFrame in batches
        
        Args:
            table_name: Target table name
            df: DataFrame to insert
            progress_callback: Optional callback for progress updates
            
        Returns:
            BatchInsertMetrics with insert statistics
        """
        start_time = time.time()
        total_rows = len(df)
        total_batches = (total_rows + self.batch_size - 1) // self.batch_size
        
        rows_inserted = 0
        batches_processed = 0
        failed_batches = 0
        retry_count = 0
        peak_gpu_memory = 0.0
        
        self.logger.info(f"Starting batch insert: {total_rows} rows in {total_batches} batches")
        
        # Process in batches
        for i in range(0, total_rows, self.batch_size):
            batch_start = i
            batch_end = min(i + self.batch_size, total_rows)
            batch_data = df.iloc[batch_start:batch_end]
            
            # Insert batch
            success = self.insert_batch(table_name, batch_data)
            
            if success:
                rows_inserted += len(batch_data)
                batches_processed += 1
            else:
                failed_batches += 1
            
            # Monitor GPU memory
            if self.gpu_available:
                current_gpu_memory = self._get_gpu_memory_usage()
                peak_gpu_memory = max(peak_gpu_memory, current_gpu_memory)
            
            # Progress callback
            if progress_callback:
                progress_callback({
                    'current_batch': batches_processed,
                    'total_batches': total_batches,
                    'rows_inserted': rows_inserted,
                    'percentage': (rows_inserted / total_rows) * 100
                })
            
            # Log progress periodically
            if batches_processed % 10 == 0:
                elapsed = time.time() - start_time
                rate = rows_inserted / elapsed if elapsed > 0 else 0
                self.logger.info(f"Progress: {rows_inserted}/{total_rows} rows "
                               f"({rate:.1f} rows/s)")
        
        # Calculate final metrics
        total_time = time.time() - start_time
        rows_per_second = rows_inserted / total_time if total_time > 0 else 0
        
        metrics = BatchInsertMetrics(
            table_name=table_name,
            total_rows_inserted=rows_inserted,
            total_batches=batches_processed,
            total_time_seconds=total_time,
            rows_per_second=rows_per_second,
            gpu_memory_peak_mb=peak_gpu_memory,
            failed_batches=failed_batches,
            retry_count=retry_count
        )
        
        self.logger.info(f"Batch insert complete: {rows_inserted} rows in {total_time:.2f}s "
                        f"({rows_per_second:.1f} rows/s)")
        
        return metrics
    
    def start_async_processing(self):
        """Start asynchronous batch processing thread"""
        if self.is_processing:
            return
            
        self.is_processing = True
        self.processing_thread = threading.Thread(target=self._process_batch_queue)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        self.logger.info("Async batch processing started")
    
    def stop_async_processing(self):
        """Stop asynchronous batch processing"""
        self.is_processing = False
        
        # Add sentinel to queue to wake up thread
        self.batch_queue.put(None)
        
        if self.processing_thread:
            self.processing_thread.join(timeout=5)
            
        self.logger.info("Async batch processing stopped")
    
    def _process_batch_queue(self):
        """Process batches from queue asynchronously"""
        while self.is_processing:
            try:
                # Get batch from queue
                item = self.batch_queue.get(timeout=1)
                
                if item is None:  # Sentinel value
                    break
                    
                table_name, batch_data = item
                
                # Insert batch
                self.insert_batch(table_name, batch_data)
                
                self.batch_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error in async batch processing: {e}")
    
    def queue_batch_async(self, table_name: str, batch_data: pd.DataFrame,
                         timeout: float = 5.0) -> bool:
        """
        Queue batch for asynchronous processing
        
        Args:
            table_name: Target table name
            batch_data: Batch data to insert
            timeout: Queue timeout in seconds
            
        Returns:
            True if queued successfully, False otherwise
        """
        try:
            self.batch_queue.put((table_name, batch_data), timeout=timeout)
            return True
        except queue.Full:
            self.logger.warning("Batch queue is full")
            return False
    
    def optimize_batch_size(self, sample_data: pd.DataFrame) -> int:
        """
        Optimize batch size based on data characteristics and GPU memory
        
        Args:
            sample_data: Sample of data to analyze
            
        Returns:
            Optimized batch size
        """
        # Estimate memory usage per row
        memory_per_row = sample_data.memory_usage(deep=True).sum() / len(sample_data)
        
        # Get available GPU memory
        if self.gpu_available:
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                if gpus:
                    available_memory = gpus[0].memoryFree * 1024 * 1024  # Convert to bytes
                    
                    # Use 50% of available GPU memory for safety
                    target_memory = available_memory * 0.5
                    
                    # Calculate optimal batch size
                    optimal_batch_size = int(target_memory / memory_per_row)
                    
                    # Clamp to reasonable range
                    optimal_batch_size = max(100, min(optimal_batch_size, 10000))
                    
                    self.logger.info(f"Optimized batch size: {optimal_batch_size} "
                                   f"(based on {available_memory/1024/1024:.1f}MB available GPU memory)")
                    
                    return optimal_batch_size
            except:
                pass
        
        # Default batch size if GPU optimization not available
        return self.batch_size


class GPUMemoryManager:
    """Manages GPU memory for data transfers"""
    
    def __init__(self, threshold_mb: float = 20000):
        """
        Initialize GPU memory manager
        
        Args:
            threshold_mb: Memory threshold in MB
        """
        self.threshold_mb = threshold_mb
        self.logger = logging.getLogger(__name__)
        
    def chunk_data_for_gpu(self, df: pd.DataFrame, target_memory_mb: float = 1000) -> List[pd.DataFrame]:
        """
        Chunk DataFrame to fit in GPU memory
        
        Args:
            df: DataFrame to chunk
            target_memory_mb: Target memory per chunk in MB
            
        Returns:
            List of DataFrame chunks
        """
        # Estimate DataFrame memory usage
        memory_usage_bytes = df.memory_usage(deep=True).sum()
        memory_usage_mb = memory_usage_bytes / 1024 / 1024
        
        if memory_usage_mb <= target_memory_mb:
            return [df]
        
        # Calculate chunk size
        num_chunks = int(np.ceil(memory_usage_mb / target_memory_mb))
        chunk_size = len(df) // num_chunks
        
        chunks = []
        for i in range(0, len(df), chunk_size):
            chunk = df.iloc[i:i + chunk_size]
            chunks.append(chunk)
            
        self.logger.info(f"Chunked data into {len(chunks)} chunks for GPU transfer")
        
        return chunks
    
    def transfer_to_gpu(self, data: pd.DataFrame) -> Optional[Any]:
        """
        Transfer data to GPU memory
        
        Args:
            data: Data to transfer
            
        Returns:
            GPU data object or None if transfer failed
        """
        try:
            # Check if CuPy is available for GPU operations
            import cupy as cp
            
            # Convert to CuPy array
            gpu_data = cp.asarray(data.values)
            
            self.logger.info(f"Transferred {data.shape} to GPU")
            
            return gpu_data
            
        except ImportError:
            self.logger.warning("CuPy not available for GPU transfers")
            return None
        except Exception as e:
            self.logger.error(f"GPU transfer failed: {e}")
            return None