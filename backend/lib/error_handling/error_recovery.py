"""
Error Recovery Manager

Implements recovery procedures and strategies for handling various
types of errors in the Heavy Optimizer Platform.
"""

import os
import time
import logging
from typing import Dict, Any, Optional, Callable, List, Type
from datetime import datetime
from pathlib import Path

from .error_types import (
    HeavyOptimizerError, RecoverableError, CriticalError,
    DataProcessingError, AlgorithmError, NetworkError,
    DatabaseError, CheckpointError, ResourceError
)
from .checkpoint_manager import CheckpointManager
from .error_notifier import ErrorNotifier
from .context_logger import ContextLogger

class ErrorRecoveryManager:
    """Manages error recovery strategies and procedures"""
    
    def __init__(self, checkpoint_manager: Optional[CheckpointManager] = None,
                 error_notifier: Optional[ErrorNotifier] = None,
                 job_id: Optional[str] = None):
        """
        Initialize ErrorRecoveryManager
        
        Args:
            checkpoint_manager: CheckpointManager instance
            error_notifier: ErrorNotifier instance
            job_id: Current job ID
        """
        self.checkpoint_manager = checkpoint_manager or CheckpointManager(job_id=job_id)
        self.error_notifier = error_notifier or ErrorNotifier()
        self.logger = ContextLogger(__name__)
        self.job_id = job_id
        
        # Recovery strategies registry
        self.recovery_strategies: Dict[Type[Exception], Callable] = {
            DataProcessingError: self._recover_data_processing_error,
            AlgorithmError: self._recover_algorithm_error,
            NetworkError: self._recover_network_error,
            DatabaseError: self._recover_database_error,
            CheckpointError: self._recover_checkpoint_error,
            ResourceError: self._recover_resource_error
        }
        
        # Recovery history
        self.recovery_history: List[Dict[str, Any]] = []
        
        # Recovery configuration
        self.max_recovery_attempts = 3
        self.recovery_timeout = 300  # 5 minutes
    
    def handle_error(self, error: Exception, context: Dict[str, Any],
                    allow_recovery: bool = True) -> Optional[Any]:
        """
        Handle an error with appropriate recovery strategy
        
        Args:
            error: The exception that occurred
            context: Error context including state and parameters
            allow_recovery: Whether to attempt recovery
            
        Returns:
            Recovery result if successful, None otherwise
        """
        self.logger.set_context(job_id=self.job_id, error_type=type(error).__name__)
        
        try:
            # Log the error with full context
            self.logger.error(f"Error occurred: {error}", exc_info=True)
            
            # Check if error is recoverable
            if not allow_recovery or not self._is_recoverable(error):
                return self._handle_critical_error(error, context)
            
            # Attempt recovery
            recovery_result = self._attempt_recovery(error, context)
            
            if recovery_result is not None:
                self.logger.info(f"Successfully recovered from {type(error).__name__}")
                return recovery_result
            else:
                self.logger.error(f"Recovery failed for {type(error).__name__}")
                return self._handle_critical_error(error, context)
                
        except Exception as recovery_error:
            self.logger.critical(
                f"Error during error recovery: {recovery_error}",
                exc_info=True
            )
            return None
        finally:
            self.logger.clear_context()
    
    def _is_recoverable(self, error: Exception) -> bool:
        """Check if an error is recoverable"""
        # Check if it's a known recoverable error type
        if isinstance(error, RecoverableError):
            return True
        
        # Check if it's a critical error
        if isinstance(error, CriticalError):
            return False
        
        # Check if we have a recovery strategy for this error type
        for error_type in self.recovery_strategies:
            if isinstance(error, error_type):
                return True
        
        return False
    
    def _attempt_recovery(self, error: Exception, context: Dict[str, Any]) -> Optional[Any]:
        """Attempt to recover from an error"""
        # Record recovery attempt
        recovery_record = {
            'timestamp': datetime.now(),
            'error_type': type(error).__name__,
            'error_message': str(error),
            'attempts': 0
        }
        
        # Find appropriate recovery strategy
        recovery_strategy = None
        for error_type, strategy in self.recovery_strategies.items():
            if isinstance(error, error_type):
                recovery_strategy = strategy
                break
        
        if not recovery_strategy:
            self.logger.warning(f"No recovery strategy found for {type(error).__name__}")
            return None
        
        # Attempt recovery with retries
        for attempt in range(self.max_recovery_attempts):
            recovery_record['attempts'] = attempt + 1
            
            try:
                self.logger.info(
                    f"Recovery attempt {attempt + 1}/{self.max_recovery_attempts} "
                    f"for {type(error).__name__}"
                )
                
                # Execute recovery strategy
                result = recovery_strategy(error, context, attempt)
                
                if result is not None:
                    recovery_record['status'] = 'success'
                    self.recovery_history.append(recovery_record)
                    return result
                
            except Exception as recovery_error:
                self.logger.error(
                    f"Recovery attempt {attempt + 1} failed: {recovery_error}"
                )
                
            # Wait before next attempt (exponential backoff)
            if attempt < self.max_recovery_attempts - 1:
                wait_time = min(2 ** attempt, 30)  # Max 30 seconds
                time.sleep(wait_time)
        
        recovery_record['status'] = 'failed'
        self.recovery_history.append(recovery_record)
        return None
    
    def _handle_critical_error(self, error: Exception, context: Dict[str, Any]) -> None:
        """Handle critical errors that cannot be recovered from"""
        # Save current state as emergency checkpoint
        try:
            self.checkpoint_manager.save_checkpoint(
                context.get('state', {}),
                f"emergency_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                f"Emergency checkpoint due to {type(error).__name__}"
            )
        except Exception as checkpoint_error:
            self.logger.error(f"Failed to save emergency checkpoint: {checkpoint_error}")
        
        # Send notification
        if isinstance(error, CriticalError) and error.notify:
            self.error_notifier.notify_critical_error(error, context, "CRITICAL")
        elif not isinstance(error, HeavyOptimizerError):
            # Unknown critical error
            self.error_notifier.notify_critical_error(error, context, "CRITICAL")
        
        return None
    
    # Recovery Strategies
    
    def _recover_data_processing_error(self, error: DataProcessingError,
                                     context: Dict[str, Any],
                                     attempt: int) -> Optional[Any]:
        """Recover from data processing errors"""
        self.logger.info(f"Attempting data processing recovery: {error.recovery_action}")
        
        # Try to load from last checkpoint
        if attempt == 0:
            last_checkpoint = self.checkpoint_manager.get_latest_checkpoint()
            if last_checkpoint:
                state = self.checkpoint_manager.load_checkpoint(last_checkpoint)
                if state:
                    self.logger.info("Recovered from checkpoint")
                    return {'recovered': True, 'state': state}
        
        # Try to process with relaxed validation
        if attempt == 1 and 'data' in context:
            self.logger.info("Attempting recovery with relaxed validation")
            return {'recovered': True, 'relaxed_validation': True}
        
        return None
    
    def _recover_algorithm_error(self, error: AlgorithmError,
                               context: Dict[str, Any],
                               attempt: int) -> Optional[Any]:
        """Recover from algorithm errors"""
        # Try different algorithm parameters
        if attempt == 0:
            self.logger.info("Attempting recovery with adjusted parameters")
            return {'recovered': True, 'adjust_parameters': True}
        
        # Try fallback algorithm
        if attempt == 1:
            self.logger.info("Attempting recovery with fallback algorithm")
            return {'recovered': True, 'use_fallback': True}
        
        return None
    
    def _recover_network_error(self, error: NetworkError,
                             context: Dict[str, Any],
                             attempt: int) -> Optional[Any]:
        """Recover from network errors"""
        # Network errors are handled by retry decorator
        # This is for additional recovery if retries fail
        
        if attempt == 0:
            self.logger.info("Waiting for network stabilization")
            time.sleep(10)  # Wait 10 seconds
            return {'recovered': True, 'retry': True}
        
        return None
    
    def _recover_database_error(self, error: DatabaseError,
                              context: Dict[str, Any],
                              attempt: int) -> Optional[Any]:
        """Recover from database errors"""
        # Try to reconnect
        if attempt == 0:
            self.logger.info("Attempting database reconnection")
            return {'recovered': True, 'reconnect': True}
        
        # Try alternative database
        if attempt == 1 and 'fallback_db' in context:
            self.logger.info("Switching to fallback database")
            return {'recovered': True, 'use_fallback_db': True}
        
        return None
    
    def _recover_checkpoint_error(self, error: CheckpointError,
                                context: Dict[str, Any],
                                attempt: int) -> Optional[Any]:
        """Recover from checkpoint errors"""
        # Try older checkpoint
        if attempt == 0:
            checkpoints = self.checkpoint_manager.list_checkpoints()
            if len(checkpoints) > 1:
                self.logger.info("Attempting recovery from older checkpoint")
                state = self.checkpoint_manager.load_checkpoint(checkpoints[1]['name'])
                if state:
                    return {'recovered': True, 'state': state}
        
        return None
    
    def _recover_resource_error(self, error: ResourceError,
                              context: Dict[str, Any],
                              attempt: int) -> Optional[Any]:
        """Recover from resource errors"""
        # Free up resources
        if attempt == 0:
            self.logger.info("Attempting to free resources")
            
            # Clear caches
            import gc
            gc.collect()
            
            # Reduce batch size if applicable
            if 'batch_size' in context:
                return {
                    'recovered': True,
                    'reduced_batch_size': context['batch_size'] // 2
                }
        
        return None
    
    def create_recovery_report(self) -> Dict[str, Any]:
        """Create a report of all recovery attempts"""
        return {
            'job_id': self.job_id,
            'recovery_history': self.recovery_history,
            'total_attempts': len(self.recovery_history),
            'successful_recoveries': sum(
                1 for r in self.recovery_history if r.get('status') == 'success'
            ),
            'failed_recoveries': sum(
                1 for r in self.recovery_history if r.get('status') == 'failed'
            )
        }
    
    def save_recovery_report(self, output_dir: str) -> None:
        """Save recovery report to file"""
        import json
        
        report = self.create_recovery_report()
        report_path = Path(output_dir) / f"recovery_report_{self.job_id}.json"
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)