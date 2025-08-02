import numpy as np
import logging
from typing import Optional, Tuple
import time

logger = logging.getLogger(__name__)


class GPUAccelerator:
    """
    GPU acceleration layer for correlation calculations.
    Supports both HeavyDB and CuPy backends.
    """
    
    def __init__(self):
        self.backend = self._detect_backend()
        self.is_available = self.backend is not None
        
    def _detect_backend(self) -> Optional[str]:
        """Detect available GPU acceleration backend"""
        # Try CuPy first (more common)
        try:
            import cupy as cp
            logger.info("🚀 CuPy GPU acceleration detected")
            return 'cupy'
        except ImportError:
            pass
        
        # Try HeavyDB
        try:
            import pymapd
            logger.info("🚀 HeavyDB GPU acceleration detected")
            return 'heavydb'
        except ImportError:
            pass
        
        logger.warning("⚠️ No GPU acceleration available, using CPU")
        return None
    
    def calculate_correlation_matrix(self, data: np.ndarray) -> np.ndarray:
        """
        Calculate correlation matrix using GPU acceleration if available.
        Falls back to NumPy if GPU is not available.
        """
        start_time = time.time()
        
        if self.backend == 'cupy':
            result = self._calculate_with_cupy(data)
        elif self.backend == 'heavydb':
            result = self._calculate_with_heavydb(data)
        else:
            result = np.corrcoef(data.T)
        
        elapsed = time.time() - start_time
        logger.info(f"⏱️ Correlation matrix calculated in {elapsed:.3f} seconds using {self.backend or 'numpy'}")
        
        return result
    
    def _calculate_with_cupy(self, data: np.ndarray) -> np.ndarray:
        """Calculate correlation matrix using CuPy GPU acceleration"""
        try:
            import cupy as cp
            
            # Transfer data to GPU
            gpu_data = cp.asarray(data)
            
            # Calculate correlation on GPU
            gpu_corr = cp.corrcoef(gpu_data.T)
            
            # Transfer result back to CPU
            result = cp.asnumpy(gpu_corr)
            
            # Clean up GPU memory
            cp.get_default_memory_pool().free_all_blocks()
            
            return result
            
        except Exception as e:
            logger.error(f"❌ CuPy calculation failed: {str(e)}")
            return np.corrcoef(data.T)
    
    def _calculate_with_heavydb(self, data: np.ndarray) -> np.ndarray:
        """
        Calculate correlation matrix using HeavyDB GPU acceleration.
        This uses HeavyDB's SQL-based correlation functions.
        """
        try:
            import pymapd
            from contextlib import closing
            
            # Connect to HeavyDB
            connection = pymapd.connect(
                host='localhost',
                port=6274,
                user='admin',
                password='HyperInteractive',
                dbname='heavyai'
            )
            
            with closing(connection) as conn:
                cursor = conn.cursor()
                
                # Create temporary table
                table_name = f"temp_corr_{int(time.time() * 1000)}"
                n_days, n_strategies = data.shape
                
                # Create table with appropriate columns
                columns = [f"s{i} DOUBLE" for i in range(n_strategies)]
                create_query = f"CREATE TEMPORARY TABLE {table_name} (day_id INTEGER, {', '.join(columns)})"
                cursor.execute(create_query)
                
                # Prepare data for insertion
                # Note: In production, use COPY for bulk loading
                for day in range(n_days):
                    values = [str(day)] + [str(data[day, i]) for i in range(n_strategies)]
                    insert_query = f"INSERT INTO {table_name} VALUES ({', '.join(values)})"
                    cursor.execute(insert_query)
                
                # Calculate correlations using HeavyDB's CORR function
                correlation_matrix = np.eye(n_strategies)
                
                for i in range(n_strategies):
                    for j in range(i + 1, n_strategies):
                        corr_query = f"SELECT CORR(s{i}, s{j}) FROM {table_name}"
                        cursor.execute(corr_query)
                        corr_value = cursor.fetchone()[0]
                        correlation_matrix[i, j] = corr_value
                        correlation_matrix[j, i] = corr_value
                
                # Clean up
                cursor.execute(f"DROP TABLE {table_name}")
                
                return correlation_matrix
                
        except Exception as e:
            logger.error(f"❌ HeavyDB calculation failed: {str(e)}")
            return np.corrcoef(data.T)
    
    def calculate_pairwise_correlations(self, data: np.ndarray, indices: np.ndarray) -> np.ndarray:
        """
        Calculate correlations only for specific strategy pairs.
        Useful for portfolio correlation calculations.
        """
        if len(indices) < 2:
            return np.array([[1.0]])
        
        subset_data = data[:, indices]
        
        if self.backend == 'cupy':
            return self._calculate_with_cupy(subset_data)
        else:
            return np.corrcoef(subset_data.T)
    
    def estimate_memory_usage(self, n_strategies: int) -> Tuple[float, str]:
        """
        Estimate memory usage for correlation matrix calculation.
        Returns (size_in_gb, recommendation)
        """
        # Correlation matrix size: n x n floats (64 bits each)
        matrix_size_gb = (n_strategies * n_strategies * 8) / (1024 ** 3)
        
        # Add overhead for computation (roughly 2x for intermediate calculations)
        total_size_gb = matrix_size_gb * 2
        
        if total_size_gb < 1:
            recommendation = "Standard memory sufficient"
        elif total_size_gb < 8:
            recommendation = "GPU acceleration recommended"
        elif total_size_gb < 24:
            recommendation = "GPU acceleration required, consider chunking"
        else:
            recommendation = "Must use chunked processing"
        
        return total_size_gb, recommendation
    
    def get_gpu_info(self) -> dict:
        """Get information about available GPU resources"""
        info = {
            'backend': self.backend,
            'is_available': self.is_available
        }
        
        if self.backend == 'cupy':
            try:
                import cupy as cp
                device = cp.cuda.Device()
                info['device_name'] = device.name.decode() if hasattr(device, 'name') else 'Unknown'
                mem_info = device.mem_info
                if hasattr(mem_info, '__len__') and len(mem_info) >= 2:
                    info['memory_free_gb'] = mem_info[0] / (1024 ** 3)
                    info['memory_total_gb'] = mem_info[1] / (1024 ** 3)
                else:
                    info['memory_free_gb'] = 0
                    info['memory_total_gb'] = 0
            except Exception as e:
                logger.debug(f"Could not get GPU info: {str(e)}")
                info['device_name'] = 'Unknown'
                info['memory_free_gb'] = 0
                info['memory_total_gb'] = 0
        
        return info