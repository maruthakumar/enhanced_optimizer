#!/usr/bin/env python3
"""
GPU Utilities and cuDF Import Handler
Centralized GPU detection and cuDF import management for the Heavy Optimizer Platform
"""

import logging
import warnings
import os
from typing import Optional, Tuple

# Suppress specific cuDF/CUDA warnings during import
with warnings.catch_warnings():
    warnings.filterwarnings('ignore', message='.*libcudart.*')
    warnings.filterwarnings('ignore', message='.*CUDA.*')
    warnings.filterwarnings('ignore', message='.*cuDF.*')
    
    try:
        import cudf
        import cupy as cp
        CUDF_AVAILABLE = True
        CUDF_VERSION = cudf.__version__
        CUDA_AVAILABLE = True
    except (ImportError, RuntimeError, OSError) as e:
        cudf = None
        cp = None
        CUDF_AVAILABLE = False
        CUDF_VERSION = None
        CUDA_AVAILABLE = False
        _IMPORT_ERROR = str(e)

# Configure logging
logger = logging.getLogger(__name__)

class GPUStatus:
    """Centralized GPU status management"""
    
    def __init__(self):
        self._gpu_available = None
        self._gpu_checked = False
        
    def is_gpu_available(self) -> bool:
        """Check if GPU is available and working"""
        if not self._gpu_checked:
            self._check_gpu_status()
            self._gpu_checked = True
        return self._gpu_available
    
    def _check_gpu_status(self) -> None:
        """Internal GPU status check"""
        if not CUDF_AVAILABLE:
            self._gpu_available = False
            return
            
        try:
            # Test basic cuDF operation
            test_df = cudf.DataFrame({'test': [1, 2, 3]})
            _ = test_df.sum()
            self._gpu_available = True
            logger.info("GPU acceleration available and functional")
        except Exception as e:
            self._gpu_available = False
            logger.info(f"GPU available but not functional: {str(e)}")
    
    def get_gpu_info(self) -> dict:
        """Get comprehensive GPU information"""
        info = {
            'cudf_available': CUDF_AVAILABLE,
            'cudf_version': CUDF_VERSION,
            'cuda_available': CUDA_AVAILABLE,
            'gpu_functional': self.is_gpu_available()
        }
        
        if CUDF_AVAILABLE:
            try:
                info['gpu_count'] = cp.cuda.runtime.getDeviceCount()
                info['current_device'] = cp.cuda.Device().id
            except:
                info['gpu_count'] = 0
                info['current_device'] = None
        else:
            info['gpu_count'] = 0
            info['current_device'] = None
            info['import_error'] = _IMPORT_ERROR
            
        return info
    
    def log_gpu_status(self, logger_instance: Optional[logging.Logger] = None) -> None:
        """Log current GPU status (once per session)"""
        log = logger_instance or logger
        
        if hasattr(self, '_status_logged'):
            return
            
        info = self.get_gpu_info()
        
        if info['gpu_functional']:
            log.info(f"🚀 GPU acceleration enabled (cuDF v{info['cudf_version']}, {info['gpu_count']} GPU(s))")
        elif info['cudf_available']:
            log.info("💻 cuDF available but GPU not functional - using CPU mode")
        else:
            log.info("💻 GPU not available - using CPU mode (this is normal on CPU-only systems)")
            
        self._status_logged = True

# Global GPU status instance
gpu_status = GPUStatus()

def get_cudf_safe():
    """
    Get cuDF module safely with proper error handling
    
    Returns:
        cudf module if available, None otherwise
    """
    return cudf if CUDF_AVAILABLE else None

def get_cupy_safe():
    """
    Get cuPy module safely with proper error handling
    
    Returns:
        cupy module if available, None otherwise
    """
    return cp if CUDF_AVAILABLE else None

def ensure_gpu_compatibility(use_gpu: bool = True) -> Tuple[bool, str]:
    """
    Ensure GPU compatibility and return appropriate settings
    
    Args:
        use_gpu: Whether GPU usage is requested
        
    Returns:
        Tuple of (actual_gpu_usage, data_type)
    """
    if not use_gpu:
        return False, 'numpy'
    
    if gpu_status.is_gpu_available():
        return True, 'cudf'
    else:
        return False, 'numpy'

def log_gpu_environment():
    """Log the GPU environment once per session"""
    gpu_status.log_gpu_status()

# Initialize logging on import
if not hasattr(log_gpu_environment, '_logged'):
    log_gpu_environment._logged = True
    log_gpu_environment()