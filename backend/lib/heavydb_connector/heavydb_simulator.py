"""
HeavyDB Simulator for Development/Testing
Simulates GPU-accelerated operations when HeavyDB is unavailable
"""

import time
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union
import logging

logger = logging.getLogger(__name__)


class HeavyDBSimulator:
    """Simulates HeavyDB GPU operations for testing"""
    
    def __init__(self):
        self.tables = {}
        self.gpu_memory = {
            'total_gb': 40.0,  # A100 GPU memory
            'used_gb': 0.0
        }
        logger.info("🎮 HeavyDB Simulator initialized (GPU simulation mode)")
    
    def create_table(self, table_name: str, df: pd.DataFrame) -> bool:
        """Simulate table creation with GPU optimization"""
        try:
            # Simulate GPU memory usage
            table_size_gb = df.memory_usage(deep=True).sum() / (1024**3)
            if self.gpu_memory['used_gb'] + table_size_gb > self.gpu_memory['total_gb']:
                raise MemoryError(f"Insufficient GPU memory: need {table_size_gb:.2f}GB")
            
            self.tables[table_name] = df.copy()
            self.gpu_memory['used_gb'] += table_size_gb
            
            # Simulate GPU processing time (faster than CPU)
            time.sleep(0.001 * len(df))  # 1ms per 1000 rows
            
            logger.info(f"✅ Table '{table_name}' created (simulated) - {len(df)} rows, {table_size_gb:.3f}GB")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create table: {e}")
            return False
    
    def drop_table(self, table_name: str) -> bool:
        """Simulate table drop"""
        if table_name in self.tables:
            # Free GPU memory
            table_size_gb = self.tables[table_name].memory_usage(deep=True).sum() / (1024**3)
            self.gpu_memory['used_gb'] -= table_size_gb
            del self.tables[table_name]
            return True
        return False
    
    def calculate_correlations_gpu(self, table_name: str) -> Optional[np.ndarray]:
        """Simulate GPU-accelerated correlation calculation"""
        if table_name not in self.tables:
            return None
        
        df = self.tables[table_name]
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            return None
        
        # Simulate GPU correlation calculation
        start_time = time.time()
        
        # For simulation, calculate a subset for speed
        sample_size = min(len(numeric_cols), 100)
        sample_cols = np.random.choice(numeric_cols, sample_size, replace=False)
        
        # Calculate correlation matrix
        corr_matrix = df[sample_cols].corr().values
        
        # Simulate GPU speedup (2-5x faster than CPU)
        gpu_speedup = 3.5
        simulated_time = (time.time() - start_time) / gpu_speedup
        time.sleep(max(0, simulated_time))
        
        logger.info(f"✅ GPU correlation calculated (simulated) - {sample_size}x{sample_size} matrix in {simulated_time:.3f}s")
        
        # Return full-size matrix filled with simulated values
        full_matrix = np.random.rand(len(numeric_cols), len(numeric_cols))
        np.fill_diagonal(full_matrix, 1.0)
        full_matrix = (full_matrix + full_matrix.T) / 2  # Make symmetric
        
        return full_matrix
    
    def execute_query(self, query: str) -> pd.DataFrame:
        """Simulate query execution"""
        # Simple query parsing for simulation
        if "SELECT 1 as test" in query:
            return pd.DataFrame({'test': [1]})
        
        if "AVG" in query and "STDDEV" in query:
            # Simulate statistics query
            return pd.DataFrame({
                'mean_value': [np.random.randn() * 100],
                'std_value': [np.random.rand() * 50],
                'min_value': [np.random.randn() * 100 - 200],
                'max_value': [np.random.randn() * 100 + 200],
                'count_value': [np.random.randint(1000, 10000)]
            })
        
        if "omnisci_memory_summary" in query:
            # Simulate GPU memory info
            return pd.DataFrame({
                'device_id': [0],
                'device_type': ['GPU'],
                'max_page_count': [10485760],  # 40GB / 4KB pages
                'page_count': [10485760],
                'allocated_page_count': [int(self.gpu_memory['used_gb'] * 262144)],  # GB to 4KB pages
                'page_size': [4096]
            })
        
        # Default empty result
        return pd.DataFrame()
    
    def get_gpu_memory_info(self) -> Dict[str, Any]:
        """Get simulated GPU memory info"""
        return {
            'available': True,
            'gpus': [{
                'device_id': 0,
                'total_memory_gb': self.gpu_memory['total_gb'],
                'used_memory_gb': round(self.gpu_memory['used_gb'], 2),
                'free_memory_gb': round(self.gpu_memory['total_gb'] - self.gpu_memory['used_gb'], 2),
                'usage_percent': round((self.gpu_memory['used_gb'] / self.gpu_memory['total_gb']) * 100, 1)
            }],
            'total_gpus': 1,
            'simulated': True
        }


class SimulatedConnection:
    """Simulated HeavyDB connection"""
    
    def __init__(self, simulator: HeavyDBSimulator):
        self.simulator = simulator
        self.closed = False
    
    def cursor(self):
        return SimulatedCursor(self.simulator)
    
    def close(self):
        self.closed = True
        logger.info("✅ Simulated connection closed")


class SimulatedCursor:
    """Simulated cursor for query execution"""
    
    def __init__(self, simulator: HeavyDBSimulator):
        self.simulator = simulator
        self.description = None
        self.results = None
    
    def execute(self, query: str):
        """Execute simulated query"""
        result_df = self.simulator.execute_query(query)
        
        if not result_df.empty:
            self.description = [(col, None) for col in result_df.columns]
            self.results = result_df.values.tolist()
        else:
            self.description = None
            self.results = []
    
    def fetchall(self):
        """Fetch all results"""
        return self.results if self.results else []
    
    def fetchone(self):
        """Fetch one result"""
        if self.results and len(self.results) > 0:
            return self.results[0]
        return None


# Global simulator instance
_simulator = HeavyDBSimulator()


def get_simulated_connection() -> SimulatedConnection:
    """Get a simulated HeavyDB connection"""
    logger.info("🎮 Using HeavyDB simulator (development mode)")
    return SimulatedConnection(_simulator)


def get_simulator() -> HeavyDBSimulator:
    """Get the simulator instance"""
    return _simulator