"""
Custom Exception Types for Heavy Optimizer Platform

Provides a hierarchy of exception types for better error handling
and recovery strategies.
"""

from typing import Optional, Dict, Any

class HeavyOptimizerError(Exception):
    """Base exception class for all Heavy Optimizer errors"""
    
    def __init__(self, message: str, error_code: Optional[str] = None,
                 context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.error_code = error_code
        self.context = context or {}
        self.recoverable = False


class RecoverableError(HeavyOptimizerError):
    """Base class for errors that can be recovered from"""
    
    def __init__(self, message: str, error_code: Optional[str] = None,
                 context: Optional[Dict[str, Any]] = None,
                 recovery_action: Optional[str] = None):
        super().__init__(message, error_code, context)
        self.recoverable = True
        self.recovery_action = recovery_action


class CriticalError(HeavyOptimizerError):
    """Base class for critical errors that require immediate attention"""
    
    def __init__(self, message: str, error_code: Optional[str] = None,
                 context: Optional[Dict[str, Any]] = None,
                 notify: bool = True):
        super().__init__(message, error_code, context)
        self.recoverable = False
        self.notify = notify


# Data Processing Errors

class DataProcessingError(RecoverableError):
    """Errors related to data processing and validation"""
    pass


class CSVLoadError(DataProcessingError):
    """Error loading CSV data"""
    
    def __init__(self, message: str, file_path: str, 
                 line_number: Optional[int] = None):
        context = {'file_path': file_path}
        if line_number:
            context['line_number'] = line_number
        super().__init__(message, 'CSV_LOAD_ERROR', context,
                        recovery_action='Check file format and retry')


class DataValidationError(DataProcessingError):
    """Error validating data integrity"""
    
    def __init__(self, message: str, field: str, value: Any):
        super().__init__(message, 'DATA_VALIDATION_ERROR',
                        {'field': field, 'value': value},
                        recovery_action='Fix data and retry')


# Algorithm Errors

class AlgorithmError(HeavyOptimizerError):
    """Base class for algorithm-related errors"""
    pass


class AlgorithmTimeoutError(AlgorithmError):
    """Algorithm exceeded maximum execution time"""
    
    def __init__(self, algorithm_name: str, timeout: float):
        super().__init__(
            f"Algorithm {algorithm_name} exceeded timeout of {timeout}s",
            'ALGORITHM_TIMEOUT',
            {'algorithm': algorithm_name, 'timeout': timeout}
        )
        self.recoverable = True


class AlgorithmConvergenceError(AlgorithmError):
    """Algorithm failed to converge"""
    
    def __init__(self, algorithm_name: str, iterations: int):
        super().__init__(
            f"Algorithm {algorithm_name} failed to converge after {iterations} iterations",
            'ALGORITHM_CONVERGENCE',
            {'algorithm': algorithm_name, 'iterations': iterations}
        )


class OptimizationError(AlgorithmError):
    """Error during optimization process"""
    pass


# Infrastructure Errors

class NetworkError(RecoverableError):
    """Network-related errors"""
    
    def __init__(self, message: str, url: Optional[str] = None,
                 status_code: Optional[int] = None):
        context = {}
        if url:
            context['url'] = url
        if status_code:
            context['status_code'] = status_code
        super().__init__(message, 'NETWORK_ERROR', context,
                        recovery_action='Check network and retry')


class DatabaseError(RecoverableError):
    """Database-related errors"""
    
    def __init__(self, message: str, query: Optional[str] = None,
                 database: Optional[str] = None):
        context = {}
        if query:
            context['query'] = query[:500]  # Truncate long queries
        if database:
            context['database'] = database
        super().__init__(message, 'DATABASE_ERROR', context,
                        recovery_action='Check database connection and retry')


class FileSystemError(RecoverableError):
    """File system related errors"""
    
    def __init__(self, message: str, path: str, operation: str):
        super().__init__(message, 'FILESYSTEM_ERROR',
                        {'path': path, 'operation': operation},
                        recovery_action='Check file permissions and disk space')


# Checkpoint Errors

class CheckpointError(HeavyOptimizerError):
    """Base class for checkpoint-related errors"""
    pass


class CheckpointSaveError(CheckpointError):
    """Error saving checkpoint"""
    
    def __init__(self, message: str, checkpoint_name: str):
        super().__init__(message, 'CHECKPOINT_SAVE_ERROR',
                        {'checkpoint_name': checkpoint_name})


class CheckpointLoadError(CheckpointError):
    """Error loading checkpoint"""
    
    def __init__(self, message: str, checkpoint_name: str):
        super().__init__(message, 'CHECKPOINT_LOAD_ERROR',
                        {'checkpoint_name': checkpoint_name})


# Configuration Errors

class ConfigurationError(CriticalError):
    """Configuration-related errors"""
    
    def __init__(self, message: str, config_file: Optional[str] = None,
                 parameter: Optional[str] = None):
        context = {}
        if config_file:
            context['config_file'] = config_file
        if parameter:
            context['parameter'] = parameter
        super().__init__(message, 'CONFIGURATION_ERROR', context)


# Resource Errors

class ResourceError(RecoverableError):
    """Resource-related errors (memory, CPU, etc.)"""
    pass


class MemoryError(ResourceError):
    """Out of memory error"""
    
    def __init__(self, message: str, required_memory: Optional[int] = None,
                 available_memory: Optional[int] = None):
        context = {}
        if required_memory:
            context['required_memory'] = required_memory
        if available_memory:
            context['available_memory'] = available_memory
        super().__init__(message, 'MEMORY_ERROR', context,
                        recovery_action='Reduce data size or increase memory')


class DiskSpaceError(ResourceError):
    """Insufficient disk space"""
    
    def __init__(self, message: str, required_space: Optional[int] = None,
                 available_space: Optional[int] = None):
        context = {}
        if required_space:
            context['required_space'] = required_space
        if available_space:
            context['available_space'] = available_space
        super().__init__(message, 'DISK_SPACE_ERROR', context,
                        recovery_action='Free disk space and retry')


# Job Processing Errors

class JobProcessingError(HeavyOptimizerError):
    """Errors related to job processing"""
    
    def __init__(self, message: str, job_id: str, 
                 error_code: Optional[str] = None):
        super().__init__(message, error_code or 'JOB_PROCESSING_ERROR',
                        {'job_id': job_id})


class JobQueueError(JobProcessingError):
    """Error in job queue management"""
    pass


class JobTimeoutError(JobProcessingError):
    """Job exceeded maximum execution time"""
    
    def __init__(self, job_id: str, timeout: float):
        super().__init__(
            f"Job {job_id} exceeded timeout of {timeout}s",
            job_id,
            'JOB_TIMEOUT'
        )
        self.recoverable = True