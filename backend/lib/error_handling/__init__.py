"""
Error Handling Infrastructure for Heavy Optimizer Platform

This module provides comprehensive error handling, recovery, and monitoring
capabilities including checkpoint/restore, retry logic, notifications, and
context-aware logging.
"""

from .checkpoint_manager import CheckpointManager
from .retry_decorator import retry, RetryError
from .error_notifier import ErrorNotifier
from .context_logger import ContextLogger, setup_error_logging
from .error_recovery import ErrorRecoveryManager
from .error_types import (
    OptimizationError,
    DataProcessingError,
    AlgorithmError,
    NetworkError,
    DatabaseError,
    CheckpointError,
    RecoverableError,
    CriticalError
)

__all__ = [
    'CheckpointManager',
    'retry',
    'RetryError',
    'ErrorNotifier',
    'ContextLogger',
    'setup_error_logging',
    'ErrorRecoveryManager',
    'OptimizationError',
    'DataProcessingError',
    'AlgorithmError',
    'NetworkError',
    'DatabaseError',
    'CheckpointError',
    'RecoverableError',
    'CriticalError'
]