#!/usr/bin/env python3
"""
Chunked Processing Module for Large Datasets
Handles 25,544 strategies efficiently through HeavyDB
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lib.heavydb_connector import get_connection, execute_query
from config.config_manager import get_config_manager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class ChunkedProcessor:
    """Process large datasets in chunks to avoid HeavyDB session timeouts"""
    
    def __init__(self):
        self.config = get_config_manager()
        self.logger = logging.getLogger(__name__)
        
        # Chunking parameters optimized for 25,544 strategies
        self.strategy_chunk_size = 2500  # Process 2500 strategies at a time
        self.batch_insert_size = 5000    # Insert 5000 rows per batch
        self.session_timeout = 45        # Conservative timeout (seconds)
        self.reconnect_interval = 10     # Reconnect after N chunks
        
        # Table management
        self.chunk_tables = []
        self.master_table = None
        
    def process_large_dataset(self, df: pd.DataFrame, 
                            base_table_name: str = "strategies",
                            portfolio_size: int = 35) -> Dict[str, Any]:
        """
        Process large dataset with intelligent chunking
        
        Args:
            df: Full dataset DataFrame
            base_table_name: Base name for tables
            portfolio_size: Target portfolio size
            
        Returns:
            Dictionary with processing results
        """
        start_time = time.time()
        
        # Identify strategy columns
        date_cols = ['Date', 'Day']
        strategy_cols = [col for col in df.columns if col not in date_cols]
        n_strategies = len(strategy_cols)
        n_days = len(df)
        
        self.logger.info(f"📊 Processing large dataset: {n_strategies} strategies x {n_days} days")
        self.logger.info(f"📦 Using chunk size: {self.strategy_chunk_size} strategies")
        
        # Calculate chunks needed
        n_chunks = (n_strategies + self.strategy_chunk_size - 1) // self.strategy_chunk_size
        self.logger.info(f"🔄 Will process in {n_chunks} chunks")
        
        # Process each chunk
        chunk_results = []
        chunk_tables = []
        
        for chunk_idx in range(n_chunks):
            try:
                # Get strategy columns for this chunk
                start_idx = chunk_idx * self.strategy_chunk_size
                end_idx = min((chunk_idx + 1) * self.strategy_chunk_size, n_strategies)
                chunk_strategies = strategy_cols[start_idx:end_idx]
                
                self.logger.info(f"\n{'='*60}")
                self.logger.info(f"📦 Processing chunk {chunk_idx+1}/{n_chunks}")
                self.logger.info(f"   Strategies: {start_idx} to {end_idx} ({len(chunk_strategies)} strategies)")
                
                # Create chunk DataFrame
                chunk_df = df[date_cols + chunk_strategies].copy()
                
                # Process this chunk
                chunk_result = self._process_single_chunk(
                    chunk_df, 
                    chunk_idx, 
                    base_table_name,
                    portfolio_size
                )
                
                if chunk_result['success']:
                    chunk_results.append(chunk_result)
                    chunk_tables.append(chunk_result['table_name'])
                else:
                    self.logger.error(f"❌ Failed to process chunk {chunk_idx+1}")
                
                # Reconnect periodically to avoid session timeout
                if (chunk_idx + 1) % self.reconnect_interval == 0:
                    self._reconnect_heavydb()
                
            except Exception as e:
                self.logger.error(f"❌ Error processing chunk {chunk_idx+1}: {e}")
                continue
        
        # Combine results
        combined_result = self._combine_chunk_results(chunk_results, chunk_tables)
        
        # Cleanup chunk tables
        self._cleanup_chunk_tables(chunk_tables)
        
        total_time = time.time() - start_time
        self.logger.info(f"\n✅ Chunked processing completed in {total_time:.1f}s")
        self.logger.info(f"📈 Processing rate: {n_strategies/total_time:.0f} strategies/second")
        
        return combined_result
    
    def _process_single_chunk(self, chunk_df: pd.DataFrame, 
                            chunk_idx: int,
                            base_table_name: str,
                            portfolio_size: int) -> Dict[str, Any]:
        """Process a single chunk of strategies"""
        chunk_start = time.time()
        
        # Create unique table name for this chunk
        table_name = f"{base_table_name}_chunk_{chunk_idx}_{int(time.time())}"
        
        try:
            # Load chunk to HeavyDB with optimized batch size
            success = self._load_chunk_to_heavydb(chunk_df, table_name)
            
            if not success:
                return {'success': False, 'error': 'Failed to load chunk'}
            
            # Calculate basic statistics for this chunk
            stats = self._calculate_chunk_statistics(table_name, chunk_df)
            
            # Calculate correlations if needed (smaller chunk = faster)
            if len(chunk_df.columns) > 2:  # More than just date columns
                corr_matrix = self._calculate_chunk_correlations(table_name, chunk_df)
            else:
                corr_matrix = None
            
            chunk_time = time.time() - chunk_start
            
            return {
                'success': True,
                'chunk_idx': chunk_idx,
                'table_name': table_name,
                'n_strategies': len(chunk_df.columns) - 2,
                'statistics': stats,
                'correlation_matrix': corr_matrix,
                'processing_time': chunk_time
            }
            
        except Exception as e:
            self.logger.error(f"❌ Chunk processing error: {e}")
            return {
                'success': False,
                'chunk_idx': chunk_idx,
                'error': str(e)
            }
    
    def _load_chunk_to_heavydb(self, chunk_df: pd.DataFrame, 
                              table_name: str) -> bool:
        """Load chunk data to HeavyDB with optimized batching"""
        conn = get_connection(force_new=True)
        if not conn:
            return False
        
        try:
            cursor = conn.cursor()
            
            # Drop if exists
            cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
            
            # Create table schema
            columns = []
            for col_name, dtype in chunk_df.dtypes.items():
                # Sanitize column names - replace special characters
                safe_name = col_name.replace(' ', '_').replace('-', '_').replace('%', 'pct').replace('(', '').replace(')', '')
                
                if safe_name.lower() in ['date', 'day']:
                    safe_name = f'"{safe_name}"'
                
                if pd.api.types.is_integer_dtype(dtype):
                    sql_type = "BIGINT"
                elif pd.api.types.is_float_dtype(dtype):
                    sql_type = "DOUBLE"
                elif pd.api.types.is_datetime64_any_dtype(dtype):
                    sql_type = "DATE"
                else:
                    sql_type = "TEXT ENCODING DICT(32)"
                
                columns.append(f"{safe_name} {sql_type}")
            
            # Create table with GPU optimization
            create_sql = f"""
            CREATE TABLE {table_name} (
                {', '.join(columns)}
            ) WITH (fragment_size=10000000)
            """
            cursor.execute(create_sql)
            
            # Insert data in smaller batches to avoid timeout
            total_rows = len(chunk_df)
            batch_size = self.batch_insert_size
            
            for i in range(0, total_rows, batch_size):
                batch = chunk_df.iloc[i:i+batch_size]
                
                # Prepare batch insert
                values_list = []
                for _, row in batch.iterrows():
                    values = []
                    for val in row:
                        if pd.isna(val):
                            values.append('NULL')
                        elif isinstance(val, pd.Timestamp):
                            values.append(f"'{val.strftime('%Y-%m-%d')}'")
                        elif isinstance(val, str):
                            values.append(f"'{val.replace(chr(39), chr(39)*2)}'")
                        else:
                            values.append(str(val))
                    values_list.append(f"({', '.join(values)})")
                
                # Execute batch insert
                if values_list:
                    insert_sql = f"INSERT INTO {table_name} VALUES {', '.join(values_list)}"
                    cursor.execute(insert_sql)
                
                progress = ((i + len(batch)) / total_rows) * 100
                if progress % 20 == 0:
                    self.logger.info(f"   Loading: {progress:.0f}%")
            
            self.logger.info(f"✅ Loaded chunk to table: {table_name}")
            conn.close()
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load chunk: {e}")
            if conn:
                conn.close()
            return False
    
    def _calculate_chunk_statistics(self, table_name: str, 
                                  chunk_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate basic statistics for chunk"""
        try:
            conn = get_connection()
            if not conn:
                return {}
            
            # Get strategy columns
            strategy_cols = [col for col in chunk_df.columns if col not in ['Date', 'Day']]
            
            stats = {
                'n_strategies': len(strategy_cols),
                'n_days': len(chunk_df),
                'total_data_points': len(strategy_cols) * len(chunk_df)
            }
            
            # Calculate aggregate statistics via HeavyDB
            for col in strategy_cols[:5]:  # Sample first 5 strategies
                safe_col = col.replace(' ', '_').replace('-', '_')
                if col.lower() in ['date', 'day']:
                    safe_col = f'"{safe_col}"'
                
                try:
                    query = f"""
                    SELECT 
                        AVG({safe_col}) as avg_return,
                        STDDEV({safe_col}) as std_dev,
                        MIN({safe_col}) as min_return,
                        MAX({safe_col}) as max_return
                    FROM {table_name}
                    """
                    result = execute_query(query, connection=conn)
                    if result is not None and not result.empty:
                        stats[f'sample_{col}'] = result.iloc[0].to_dict()
                except:
                    pass
            
            return stats
            
        except Exception as e:
            self.logger.error(f"❌ Failed to calculate statistics: {e}")
            return {}
    
    def _calculate_chunk_correlations(self, table_name: str, 
                                    chunk_df: pd.DataFrame) -> Optional[np.ndarray]:
        """Calculate correlations for chunk"""
        try:
            # For chunks, use pandas for efficiency
            strategy_cols = [col for col in chunk_df.columns if col not in ['Date', 'Day']]
            
            if len(strategy_cols) > 1:
                corr_matrix = chunk_df[strategy_cols].corr().values
                self.logger.info(f"✅ Calculated correlations: {corr_matrix.shape}")
                return corr_matrix
            
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Failed to calculate correlations: {e}")
            return None
    
    def _combine_chunk_results(self, chunk_results: List[Dict], 
                             chunk_tables: List[str]) -> Dict[str, Any]:
        """Combine results from all chunks"""
        if not chunk_results:
            return {'success': False, 'error': 'No chunks processed'}
        
        # Aggregate statistics
        total_strategies = sum(r['n_strategies'] for r in chunk_results if r['success'])
        total_time = sum(r['processing_time'] for r in chunk_results if r['success'])
        
        # Create summary
        summary = {
            'success': True,
            'total_strategies': total_strategies,
            'chunks_processed': len([r for r in chunk_results if r['success']]),
            'total_chunks': len(chunk_results),
            'total_processing_time': total_time,
            'chunk_tables': chunk_tables,
            'statistics': {
                'avg_chunk_time': total_time / len(chunk_results) if chunk_results else 0,
                'strategies_per_second': total_strategies / total_time if total_time > 0 else 0
            }
        }
        
        return summary
    
    def _cleanup_chunk_tables(self, chunk_tables: List[str]):
        """Clean up temporary chunk tables"""
        conn = get_connection()
        if not conn:
            return
        
        try:
            cursor = conn.cursor()
            for table in chunk_tables:
                try:
                    cursor.execute(f"DROP TABLE IF EXISTS {table}")
                    self.logger.info(f"🧹 Cleaned up table: {table}")
                except:
                    pass
            conn.close()
        except:
            pass
    
    def _reconnect_heavydb(self):
        """Reconnect to HeavyDB to avoid session timeout"""
        self.logger.info("🔄 Reconnecting to HeavyDB...")
        
        # Close existing connection
        conn = get_connection()
        if conn:
            try:
                conn.close()
            except:
                pass
        
        # Force new connection
        time.sleep(1)  # Brief pause
        new_conn = get_connection(force_new=True)
        if new_conn:
            self.logger.info("✅ Reconnected successfully")
        else:
            self.logger.error("❌ Reconnection failed")


def test_chunked_processing():
    """Test chunked processing with sample data"""
    print("\n🧪 Testing Chunked Processing")
    print("="*60)
    
    # Create test data simulating 25,544 strategies
    n_strategies = 25544
    n_days = 82
    
    print(f"Creating test data: {n_strategies} strategies x {n_days} days")
    
    # Create in chunks to avoid memory issues
    processor = ChunkedProcessor()
    
    # Test with smaller dataset first
    test_strategies = 100
    test_data = pd.DataFrame({
        'Date': pd.date_range('2024-01-01', periods=n_days),
        'Day': range(1, n_days + 1)
    })
    
    # Add strategy columns
    for i in range(test_strategies):
        test_data[f'SENSEX_Strategy_{i}'] = np.random.randn(n_days) * 100
    
    print(f"\nProcessing test dataset ({test_strategies} strategies)...")
    
    result = processor.process_large_dataset(test_data, portfolio_size=35)
    
    if result['success']:
        print("\n✅ Test successful!")
        print(f"   Processed: {result['total_strategies']} strategies")
        print(f"   Time: {result['total_processing_time']:.1f}s")
        print(f"   Rate: {result['statistics']['strategies_per_second']:.0f} strategies/second")
    else:
        print("\n❌ Test failed!")
        print(f"   Error: {result.get('error', 'Unknown error')}")


if __name__ == "__main__":
    test_chunked_processing()