"""
Retry Decorator with Exponential Backoff

Provides retry logic for transient failures such as network issues,
database connection problems, or temporary resource unavailability.
"""

import time
import random
import functools
import logging
from typing import Callable, Type, Tuple, Union, Optional, Any

class RetryError(Exception):
    """Raised when all retry attempts have been exhausted"""
    def __init__(self, last_exception: Exception, attempts: int):
        self.last_exception = last_exception
        self.attempts = attempts
        super().__init__(f"Failed after {attempts} attempts. Last error: {last_exception}")

def retry(max_attempts: int = 3,
          delay: float = 1.0,
          backoff: Union[str, float] = "exponential",
          max_delay: float = 60.0,
          exceptions: Tuple[Type[Exception], ...] = (Exception,),
          on_retry: Optional[Callable] = None,
          jitter: bool = True) -> Callable:
    """
    Decorator that retries a function on failure with configurable backoff
    
    Args:
        max_attempts: Maximum number of retry attempts (including initial call)
        delay: Initial delay between retries in seconds
        backoff: Backoff strategy - "exponential", "linear", or a multiplier float
        max_delay: Maximum delay between retries
        exceptions: Tuple of exception types to catch and retry
        on_retry: Optional callback function called on each retry
        jitter: Add random jitter to prevent thundering herd
    
    Examples:
        @retry(max_attempts=3, delay=1.0, backoff="exponential")
        def fetch_data():
            return requests.get("https://api.example.com/data")
        
        @retry(max_attempts=5, exceptions=(ConnectionError, TimeoutError))
        def connect_to_database():
            return db.connect()
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            logger = logging.getLogger(func.__module__)
            attempt = 0
            current_delay = delay
            
            while attempt < max_attempts:
                attempt += 1
                
                try:
                    # Log attempt
                    if attempt > 1:
                        logger.info(f"Retry attempt {attempt}/{max_attempts} for {func.__name__}")
                    
                    # Execute function
                    result = func(*args, **kwargs)
                    
                    # Success - log if it was a retry
                    if attempt > 1:
                        logger.info(f"Function {func.__name__} succeeded after {attempt} attempts")
                    
                    return result
                    
                except exceptions as e:
                    # Log the error
                    logger.warning(f"Attempt {attempt}/{max_attempts} failed for {func.__name__}: {e}")
                    
                    # Check if we should retry
                    if attempt >= max_attempts:
                        logger.error(f"All {max_attempts} attempts failed for {func.__name__}")
                        raise RetryError(e, max_attempts)
                    
                    # Call retry callback if provided
                    if on_retry:
                        try:
                            on_retry(e, attempt, func.__name__)
                        except Exception as callback_error:
                            logger.error(f"Retry callback failed: {callback_error}")
                    
                    # Calculate next delay
                    if isinstance(backoff, str):
                        if backoff == "exponential":
                            current_delay = min(delay * (2 ** (attempt - 1)), max_delay)
                        elif backoff == "linear":
                            current_delay = min(delay * attempt, max_delay)
                        else:
                            raise ValueError(f"Unknown backoff strategy: {backoff}")
                    else:
                        # Backoff is a multiplier
                        current_delay = min(current_delay * backoff, max_delay)
                    
                    # Add jitter if enabled
                    if jitter:
                        jitter_amount = current_delay * 0.1  # 10% jitter
                        current_delay += random.uniform(-jitter_amount, jitter_amount)
                    
                    # Log delay
                    logger.info(f"Waiting {current_delay:.2f}s before retry...")
                    
                    # Wait before retry
                    time.sleep(current_delay)
                
                except Exception as e:
                    # Unexpected exception - don't retry
                    logger.error(f"Unexpected error in {func.__name__}: {e}", exc_info=True)
                    raise
            
            # Should never reach here
            raise RuntimeError(f"Retry loop exited unexpectedly for {func.__name__}")
        
        return wrapper
    return decorator


# Specialized retry decorators for common scenarios

def retry_on_network_error(max_attempts: int = 5, initial_delay: float = 2.0):
    """
    Retry decorator specifically for network-related errors
    
    Retries on common network exceptions with exponential backoff
    """
    network_exceptions = []
    
    # Try to import network-related exceptions
    try:
        import requests
        network_exceptions.extend([
            requests.ConnectionError,
            requests.Timeout,
            requests.HTTPError
        ])
    except ImportError:
        pass
    
    try:
        import urllib.error
        network_exceptions.extend([
            urllib.error.URLError,
            urllib.error.HTTPError
        ])
    except ImportError:
        pass
    
    # Always include built-in network errors
    network_exceptions.extend([
        ConnectionError,
        TimeoutError,
        OSError
    ])
    
    return retry(
        max_attempts=max_attempts,
        delay=initial_delay,
        backoff="exponential",
        exceptions=tuple(network_exceptions),
        jitter=True
    )


def retry_on_database_error(max_attempts: int = 3, initial_delay: float = 1.0):
    """
    Retry decorator specifically for database-related errors
    
    Retries on common database exceptions with exponential backoff
    """
    db_exceptions = []
    
    # Try to import database-related exceptions
    try:
        import pymongo.errors
        db_exceptions.extend([
            pymongo.errors.ConnectionFailure,
            pymongo.errors.ServerSelectionTimeoutError
        ])
    except ImportError:
        pass
    
    try:
        import psycopg2
        db_exceptions.extend([
            psycopg2.OperationalError,
            psycopg2.DatabaseError
        ])
    except ImportError:
        pass
    
    try:
        import sqlite3
        db_exceptions.extend([
            sqlite3.OperationalError,
            sqlite3.DatabaseError
        ])
    except ImportError:
        pass
    
    # Add generic database errors
    db_exceptions.append(Exception)  # Fallback
    
    return retry(
        max_attempts=max_attempts,
        delay=initial_delay,
        backoff="exponential",
        exceptions=tuple(db_exceptions)
    )


def retry_on_resource_error(max_attempts: int = 4, initial_delay: float = 0.5):
    """
    Retry decorator for resource-related errors (file locks, memory, etc.)
    
    Retries on resource exceptions with shorter delays
    """
    resource_exceptions = (
        OSError,
        IOError,
        MemoryError,
        BlockingIOError,
        PermissionError
    )
    
    return retry(
        max_attempts=max_attempts,
        delay=initial_delay,
        backoff="linear",
        exceptions=resource_exceptions,
        max_delay=5.0
    )


# Context manager for retry blocks
class RetryContext:
    """
    Context manager for retrying code blocks
    
    Usage:
        with RetryContext(max_attempts=3) as retry_ctx:
            # Code that might fail
            result = potentially_failing_operation()
    """
    
    def __init__(self, max_attempts: int = 3, delay: float = 1.0, 
                 backoff: str = "exponential", exceptions: Tuple[Type[Exception], ...] = (Exception,)):
        self.max_attempts = max_attempts
        self.delay = delay
        self.backoff = backoff
        self.exceptions = exceptions
        self.attempt = 0
        self.logger = logging.getLogger(__name__)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            return True
        
        if not issubclass(exc_type, self.exceptions):
            return False
        
        self.attempt += 1
        if self.attempt >= self.max_attempts:
            self.logger.error(f"Retry context exhausted after {self.attempt} attempts")
            return False
        
        # Calculate delay
        if self.backoff == "exponential":
            wait_time = self.delay * (2 ** (self.attempt - 1))
        else:
            wait_time = self.delay * self.attempt
        
        self.logger.info(f"Retrying after {wait_time}s (attempt {self.attempt}/{self.max_attempts})")
        time.sleep(wait_time)
        
        # Suppress exception to continue
        return True