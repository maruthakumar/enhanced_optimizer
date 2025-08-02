"""
Context-Aware Error Logger

Enhanced logging that captures full context including stack traces,
input parameters, system state, and execution environment.
"""

import os
import sys
import json
import logging
import traceback
import inspect
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import threading
import psutil
import platform

class ContextLogger:
    """Enhanced logger that captures full execution context"""
    
    def __init__(self, name: str, log_dir: str = "/mnt/optimizer_share/logs"):
        """
        Initialize ContextLogger
        
        Args:
            name: Logger name (usually module name)
            log_dir: Directory for log files
        """
        self.name = name
        self.logger = logging.getLogger(name)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Thread-local storage for context
        self.context = threading.local()
        
        # Configure detailed formatter
        self._setup_handlers()
    
    def set_context(self, **kwargs) -> None:
        """
        Set context variables that will be included in all log messages
        
        Args:
            **kwargs: Context key-value pairs
        """
        if not hasattr(self.context, 'data'):
            self.context.data = {}
        self.context.data.update(kwargs)
    
    def clear_context(self) -> None:
        """Clear all context variables"""
        if hasattr(self.context, 'data'):
            self.context.data.clear()
    
    def log_with_context(self, level: int, message: str, 
                        exc_info: Optional[Exception] = None,
                        include_locals: bool = True) -> None:
        """
        Log message with full context
        
        Args:
            level: Logging level
            message: Log message
            exc_info: Exception information
            include_locals: Whether to include local variables
        """
        # Gather context information
        context_data = self._gather_context(exc_info, include_locals)
        
        # Create log record with extra context
        self.logger.log(level, message, exc_info=exc_info, extra={
            'context': context_data,
            'timestamp': datetime.now().isoformat()
        })
        
        # For errors, also write to dedicated error file
        if level >= logging.ERROR:
            self._write_error_detail(message, context_data, exc_info)
    
    def error(self, message: str, exc_info: bool = True, **kwargs) -> None:
        """Log error with full context"""
        if exc_info and sys.exc_info()[0] is not None:
            exception = sys.exc_info()[1]
        else:
            exception = None
        
        self.log_with_context(logging.ERROR, message, exception, **kwargs)
    
    def critical(self, message: str, exc_info: bool = True, **kwargs) -> None:
        """Log critical error with full context"""
        if exc_info and sys.exc_info()[0] is not None:
            exception = sys.exc_info()[1]
        else:
            exception = None
        
        self.log_with_context(logging.CRITICAL, message, exception, **kwargs)
    
    def _gather_context(self, exc_info: Optional[Exception] = None,
                       include_locals: bool = True) -> Dict[str, Any]:
        """Gather comprehensive context information"""
        context = {
            'thread_id': threading.current_thread().ident,
            'thread_name': threading.current_thread().name,
            'process_id': os.getpid(),
            'timestamp': datetime.now().isoformat()
        }
        
        # Add custom context
        if hasattr(self.context, 'data'):
            context['custom_context'] = self.context.data.copy()
        
        # Add caller information
        frame = inspect.currentframe()
        if frame:
            # Skip internal frames
            for _ in range(3):
                frame = frame.f_back
                if not frame:
                    break
            
            if frame:
                context['caller'] = {
                    'filename': frame.f_code.co_filename,
                    'function': frame.f_code.co_name,
                    'line_number': frame.f_lineno
                }
                
                # Add local variables if requested
                if include_locals:
                    context['locals'] = self._serialize_locals(frame.f_locals)
        
        # Add exception information
        if exc_info:
            context['exception'] = {
                'type': type(exc_info).__name__,
                'message': str(exc_info),
                'traceback': traceback.format_exception(
                    type(exc_info), exc_info, exc_info.__traceback__
                )
            }
        elif sys.exc_info()[0] is not None:
            # Current exception
            exc_type, exc_value, exc_tb = sys.exc_info()
            context['exception'] = {
                'type': exc_type.__name__ if exc_type else 'Unknown',
                'message': str(exc_value) if exc_value else 'No message',
                'traceback': traceback.format_exception(exc_type, exc_value, exc_tb)
            }
        
        # Add system information
        context['system'] = self._get_system_info()
        
        return context
    
    def _serialize_locals(self, locals_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize local variables for logging"""
        serialized = {}
        
        for key, value in locals_dict.items():
            # Skip internal variables
            if key.startswith('_'):
                continue
            
            try:
                # Try to serialize directly
                if isinstance(value, (str, int, float, bool, type(None))):
                    serialized[key] = value
                elif isinstance(value, (list, tuple)):
                    serialized[key] = str(value)[:200]  # Truncate long lists
                elif isinstance(value, dict):
                    serialized[key] = str(value)[:200]  # Truncate long dicts
                else:
                    # For objects, try to get useful information
                    serialized[key] = {
                        'type': type(value).__name__,
                        'str': str(value)[:100]
                    }
            except Exception:
                serialized[key] = f"<Error serializing {type(value).__name__}>"
        
        return serialized
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get current system information"""
        try:
            return {
                'platform': platform.platform(),
                'python_version': platform.python_version(),
                'cpu_percent': psutil.cpu_percent(interval=0.1),
                'memory': {
                    'total': psutil.virtual_memory().total,
                    'available': psutil.virtual_memory().available,
                    'percent': psutil.virtual_memory().percent
                },
                'disk': {
                    'total': psutil.disk_usage('/').total,
                    'free': psutil.disk_usage('/').free,
                    'percent': psutil.disk_usage('/').percent
                }
            }
        except Exception:
            return {'error': 'Failed to gather system info'}
    
    def _write_error_detail(self, message: str, context: Dict[str, Any],
                           exc_info: Optional[Exception]) -> None:
        """Write detailed error information to file"""
        try:
            error_dir = self.log_dir / "errors"
            error_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            error_file = error_dir / f"error_{self.name}_{timestamp}.json"
            
            error_data = {
                'timestamp': datetime.now().isoformat(),
                'logger': self.name,
                'message': message,
                'context': context
            }
            
            with open(error_file, 'w') as f:
                json.dump(error_data, f, indent=2, default=str)
                
        except Exception as e:
            # Fallback to basic logging
            self.logger.error(f"Failed to write error detail file: {e}")
    
    def _setup_handlers(self) -> None:
        """Setup logging handlers with enhanced formatting"""
        # Remove existing handlers
        self.logger.handlers.clear()
        self.logger.setLevel(logging.DEBUG)
        
        # Console handler with color
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = ColoredFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler with full details
        file_handler = logging.FileHandler(
            self.log_dir / f"{self.name}.log",
            mode='a'
        )
        file_handler.setLevel(logging.DEBUG)
        file_formatter = ContextFormatter()
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # Error file handler
        error_handler = logging.FileHandler(
            self.log_dir / f"{self.name}_errors.log",
            mode='a'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(ContextFormatter(include_context=True))
        self.logger.addHandler(error_handler)


class ContextFormatter(logging.Formatter):
    """Custom formatter that includes context information"""
    
    def __init__(self, include_context: bool = False):
        super().__init__()
        self.include_context = include_context
    
    def format(self, record: logging.LogRecord) -> str:
        # Base format
        base_format = (
            f"{record.asctime} | {record.levelname:8} | "
            f"{record.name} | {record.funcName}:{record.lineno} | "
            f"{record.getMessage()}"
        )
        
        # Add context if available and requested
        if self.include_context and hasattr(record, 'context'):
            context_str = json.dumps(record.context, indent=2, default=str)
            base_format += f"\nContext:\n{context_str}"
        
        # Add exception info if present
        if record.exc_info:
            base_format += f"\n{''.join(traceback.format_exception(*record.exc_info))}"
        
        return base_format


class ColoredFormatter(logging.Formatter):
    """Formatter with color support for console output"""
    
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m'       # Reset
    }
    
    def format(self, record: logging.LogRecord) -> str:
        # Add color to level name
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"
        
        # Format the message
        formatted = super().format(record)
        
        # Reset level name
        record.levelname = levelname
        
        return formatted


def setup_error_logging(log_dir: str = "/mnt/optimizer_share/logs") -> None:
    """
    Setup comprehensive error logging for the entire application
    
    Args:
        log_dir: Directory for log files
    """
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    
    # Remove existing handlers
    root_logger.handlers.clear()
    
    # Create log directory
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(ColoredFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    root_logger.addHandler(console_handler)
    
    # Main log file
    file_handler = logging.FileHandler(
        log_path / "heavy_optimizer.log",
        mode='a'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)s | %(name)s | %(funcName)s:%(lineno)d | %(message)s'
    ))
    root_logger.addHandler(file_handler)
    
    # Error log file
    error_handler = logging.FileHandler(
        log_path / "heavy_optimizer_errors.log",
        mode='a'
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(ContextFormatter(include_context=True))
    root_logger.addHandler(error_handler)
    
    # Set up exception hook
    def exception_hook(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        
        root_logger.critical(
            "Uncaught exception",
            exc_info=(exc_type, exc_value, exc_traceback)
        )
    
    sys.excepthook = exception_hook