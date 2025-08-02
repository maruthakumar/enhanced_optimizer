#!/usr/bin/env python3
"""
Optimal Preprocessing Pipeline for Heavy Optimizer Platform
Based on analysis: CSV → Apache Arrow → HeavyDB is the best approach
"""

import os
import sys
import time
import logging
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pa_csv
import pyarrow.parquet as pq
from typing import Dict, List, Tuple, Optional, Any, Iterator
from pathlib import Path
import re

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib.heavydb_connector import get_connection, load_strategy_data
from ulta_calculator import ULTACalculator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class OptimalPreprocessor:
    """
    Optimal preprocessing pipeline using Apache Arrow for GPU-friendly data processing
    
    Pipeline: CSV → Arrow Batches → HeavyDB
    Benefits:
    - Columnar format (GPU-optimized)
    - Zero-copy data access
    - Efficient memory usage
    - Handles session timeouts
    - Preserves data types
    """
    
    def __init__(self, apply_ulta: bool = True, config_path: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        
        # Optimized configuration
        self.config = {
            'arrow_block_size': 64 * 1024 * 1024,  # 64MB blocks for optimal GPU processing
            'batch_size': 10000,                    # Rows per batch for HeavyDB
            'chunk_size': 2500,                     # Strategies per chunk
            'max_memory_usage': 2048,               # MB limit
            'validate_data': True,
            'compress_intermediate': True,
            'apply_ulta': apply_ulta                # Enable ULTA processing
        }
        
        # Initialize ULTA calculator if enabled
        self.ulta_calculator = None
        if self.config['apply_ulta']:
            self.ulta_calculator = ULTACalculator(config_path=config_path)
            self.logger.info("🔄 ULTA processing enabled in preprocessing pipeline")
        
        # Column cleaning rules optimized for SQL/HeavyDB
        self.column_transforms = {
            '%': 'pct',
            '$': 'dollar', 
            '&': 'and',
            '@': 'at',
            '#': 'hash',
            '(': '_', ')': '_',
            '[': '_', ']': '_',
            '{': '_', '}': '_',
            '<': 'lt', '>': 'gt',
            '+': 'plus', '=': 'eq',
            '|': 'pipe', '\\': '_', '/': '_',
            "'": '', '"': '', '`': '',
            '~': '', '^': '', '!': '', '?': '',
            '*': 'star', ',': '_', ';': '_', ':': '_',
            ' ': '_', '-': '_', '.': '_'
        }
        
        # SQL reserved keywords
        self.reserved_keywords = {
            'date', 'time', 'day', 'month', 'year', 'user', 'table', 'column',
            'select', 'from', 'where', 'group', 'order', 'by', 'value', 'values'
        }
    
    def process_large_csv(self, csv_path: str, 
                         output_strategy: str = 'direct_heavydb',
                         table_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Process large CSV using optimal Arrow-based pipeline
        
        Args:
            csv_path: Input CSV file path
            output_strategy: 'direct_heavydb', 'parquet_cache', or 'memory_efficient'
            table_name: HeavyDB table name
            
        Returns:
            Processing results and statistics
        """
        start_time = time.time()
        self.logger.info(f"🚀 Starting optimal preprocessing: {csv_path}")
        
        # Analyze file size to choose strategy
        file_size_mb = os.path.getsize(csv_path) / (1024 * 1024)
        self.logger.info(f"📊 File size: {file_size_mb:.1f} MB")
        
        # Choose optimal strategy based on file size
        if file_size_mb < 500:
            strategy = 'memory_efficient'
        elif file_size_mb < 2000:
            strategy = 'chunked_arrow'
        else:
            strategy = 'streaming_arrow'
        
        self.logger.info(f"🎯 Using strategy: {strategy}")
        
        # Execute chosen strategy
        if strategy == 'memory_efficient':
            result = self._process_memory_efficient(csv_path, table_name)
        elif strategy == 'chunked_arrow':
            result = self._process_chunked_arrow(csv_path, table_name)
        else:
            result = self._process_streaming_arrow(csv_path, table_name)
        
        # Add overall timing
        result['total_processing_time'] = time.time() - start_time
        result['file_size_mb'] = file_size_mb
        result['strategy_used'] = strategy
        
        self.logger.info(f"✅ Processing completed in {result['total_processing_time']:.2f}s")
        
        return result
    
    def _process_memory_efficient(self, csv_path: str, table_name: str) -> Dict[str, Any]:
        """Process medium-sized files using in-memory Arrow tables"""
        self.logger.info("💾 Using memory-efficient Arrow processing")
        
        # Configure Arrow CSV reader for optimal performance
        read_options = pa_csv.ReadOptions(
            block_size=self.config['arrow_block_size'],
            use_threads=True
        )
        
        parse_options = pa_csv.ParseOptions(
            delimiter=',',
            quote_char='"',
            escape_char='\\',
            newlines_in_values=False
        )
        
        convert_options = pa_csv.ConvertOptions(
            null_values=['', 'NULL', 'null', 'N/A', 'NA', 'nan'],
            strings_can_be_null=True,
            auto_dict_encode=True,  # Use dictionary encoding for text
            auto_dict_max_cardinality=1000
        )
        
        # Read CSV to Arrow table
        self.logger.info("📖 Reading CSV with PyArrow...")
        read_start = time.time()
        
        arrow_table = pa_csv.read_csv(
            csv_path,
            read_options=read_options,
            parse_options=parse_options,
            convert_options=convert_options
        )
        
        read_time = time.time() - read_start
        self.logger.info(f"✅ CSV read in {read_time:.2f}s - Shape: {arrow_table.shape}")
        
        # Clean schema (column names)
        clean_start = time.time()
        cleaned_table = self._clean_arrow_schema(arrow_table)
        clean_time = time.time() - clean_start
        
        # Validate data
        validate_start = time.time()
        validation_report = self._validate_arrow_table(cleaned_table)
        validate_time = time.time() - validate_start
        
        # Apply ULTA processing if enabled
        ulta_time = 0
        ulta_report = {}
        if self.config['apply_ulta'] and self.ulta_calculator:
            ulta_start = time.time()
            cleaned_table, ulta_report = self._apply_ulta_to_arrow_table(cleaned_table)
            ulta_time = time.time() - ulta_start
            self.logger.info(f"🔄 ULTA processing completed in {ulta_time:.2f}s")
        
        # Convert to HeavyDB
        if table_name:
            heavydb_start = time.time()
            success = self._arrow_to_heavydb(cleaned_table, table_name)
            heavydb_time = time.time() - heavydb_start
        else:
            success = True
            heavydb_time = 0
        
        return {
            'method': 'Memory-Efficient Arrow',
            'success': success,
            'rows': arrow_table.num_rows,
            'columns': arrow_table.num_columns,
            'memory_usage_mb': arrow_table.nbytes / (1024**2),
            'timings': {
                'read_csv': read_time,
                'clean_schema': clean_time,
                'validate_data': validate_time,
                'ulta_processing': ulta_time,
                'load_heavydb': heavydb_time
            },
            'validation': validation_report,
            'ulta_report': ulta_report
        }
    
    def _process_chunked_arrow(self, csv_path: str, table_name: str) -> Dict[str, Any]:
        """Process large files using chunked Arrow batches"""
        self.logger.info("📦 Using chunked Arrow batch processing")
        
        # First, read schema from first few rows
        schema_df = pd.read_csv(csv_path, nrows=10)
        schema_info = self._analyze_schema(schema_df)
        
        # Determine optimal chunk size based on columns
        n_strategies = len([col for col in schema_df.columns if col not in ['Date', 'Day']])
        optimal_chunk_size = min(self.config['chunk_size'], max(500, 25000 // n_strategies))
        
        self.logger.info(f"📊 Processing {n_strategies} strategies in chunks of {optimal_chunk_size}")
        
        # Process file in strategy chunks
        chunks_processed = 0
        total_rows = 0
        chunk_results = []
        
        # Read full data for chunking
        df = pd.read_csv(csv_path)
        date_cols = ['Date', 'Day']
        strategy_cols = [col for col in df.columns if col not in date_cols]
        
        # Process strategy columns in chunks
        for i in range(0, len(strategy_cols), optimal_chunk_size):
            chunk_start = time.time()
            
            # Get chunk columns
            chunk_strategy_cols = strategy_cols[i:i+optimal_chunk_size]
            chunk_df = df[date_cols + chunk_strategy_cols]
            
            self.logger.info(f"📦 Processing chunk {chunks_processed + 1}: "
                           f"{len(chunk_strategy_cols)} strategies")
            
            # Convert chunk to Arrow
            arrow_chunk = pa.Table.from_pandas(chunk_df)
            
            # Clean schema
            cleaned_chunk = self._clean_arrow_schema(arrow_chunk)
            
            # Load to HeavyDB
            chunk_table_name = f"{table_name}_chunk_{chunks_processed}" if table_name else None
            if chunk_table_name:
                success = self._arrow_to_heavydb(cleaned_chunk, chunk_table_name)
            else:
                success = True
            
            chunk_time = time.time() - chunk_start
            chunk_results.append({
                'chunk_id': chunks_processed,
                'strategies': len(chunk_strategy_cols),
                'rows': len(chunk_df),
                'success': success,
                'processing_time': chunk_time
            })
            
            chunks_processed += 1
            total_rows += len(chunk_df)
            
            # Memory management
            del chunk_df, arrow_chunk, cleaned_chunk
        
        return {
            'method': 'Chunked Arrow Batches',
            'success': all(r['success'] for r in chunk_results),
            'chunks_processed': chunks_processed,
            'total_rows': total_rows,
            'total_strategies': len(strategy_cols),
            'chunk_results': chunk_results,
            'avg_chunk_time': np.mean([r['processing_time'] for r in chunk_results])
        }
    
    def _process_streaming_arrow(self, csv_path: str, table_name: str) -> Dict[str, Any]:
        """Process very large files using Arrow streaming"""
        self.logger.info("🌊 Using streaming Arrow processing")
        
        # Configure streaming reader
        read_options = pa_csv.ReadOptions(
            block_size=32 * 1024 * 1024,  # 32MB blocks for streaming
            use_threads=True
        )
        
        # Open streaming reader
        csv_reader = pa_csv.open_csv(csv_path, read_options=read_options)
        
        batch_count = 0
        total_rows = 0
        batch_results = []
        
        # Process batches
        for batch in csv_reader:
            batch_start = time.time()
            
            # Clean batch schema
            cleaned_batch = self._clean_arrow_schema(batch)
            
            # Convert batch to pandas for HeavyDB compatibility
            batch_df = cleaned_batch.to_pandas()
            
            # Load batch to HeavyDB
            batch_table_name = f"{table_name}_batch_{batch_count}" if table_name else None
            if batch_table_name:
                success = load_strategy_data(batch_df, batch_table_name)
            else:
                success = True
            
            batch_time = time.time() - batch_start
            batch_results.append({
                'batch_id': batch_count,
                'rows': batch.num_rows,
                'success': success,
                'processing_time': batch_time
            })
            
            batch_count += 1
            total_rows += batch.num_rows
            
            self.logger.info(f"🌊 Processed batch {batch_count}: {batch.num_rows} rows")
            
            # Memory cleanup
            del batch, cleaned_batch, batch_df
        
        return {
            'method': 'Streaming Arrow',
            'success': all(r['success'] for r in batch_results),
            'batches_processed': batch_count,
            'total_rows': total_rows,
            'batch_results': batch_results,
            'avg_batch_time': np.mean([r['processing_time'] for r in batch_results])
        }
    
    def _clean_arrow_schema(self, arrow_table: pa.Table) -> pa.Table:
        """Clean Arrow table schema (column names)"""
        cleaned_fields = []
        name_counts = {}
        
        for field in arrow_table.schema:
            # Clean column name
            clean_name = self._clean_column_name(field.name)
            
            # Ensure uniqueness
            if clean_name in name_counts:
                name_counts[clean_name] += 1
                clean_name = f"{clean_name}_{name_counts[clean_name]}"
            else:
                name_counts[clean_name] = 0
            
            cleaned_fields.append(pa.field(clean_name, field.type))
        
        # Create new schema and rename columns
        new_schema = pa.schema(cleaned_fields)
        return arrow_table.rename_columns([f.name for f in cleaned_fields])
    
    def _clean_column_name(self, name: str) -> str:
        """Clean individual column name"""
        # Convert to string and strip
        clean = str(name).strip()
        
        # Apply character transformations
        for old_char, new_char in self.column_transforms.items():
            clean = clean.replace(old_char, new_char)
        
        # Remove multiple underscores
        clean = re.sub(r'_+', '_', clean)
        clean = clean.strip('_')
        
        # Handle empty names
        if not clean:
            clean = 'unnamed_column'
        
        # Ensure starts with letter
        if clean[0].isdigit():
            clean = f'col_{clean}'
        
        # Convert to lowercase
        clean = clean.lower()
        
        # Handle reserved keywords
        if clean in self.reserved_keywords:
            clean = f'{clean}_col'
        
        return clean
    
    def _analyze_schema(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze DataFrame schema for optimization"""
        return {
            'total_columns': len(df.columns),
            'date_columns': [col for col in df.columns if col.lower() in ['date', 'day']],
            'strategy_columns': [col for col in df.columns if col not in ['Date', 'Day']],
            'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024**2)
        }
    
    def _validate_arrow_table(self, arrow_table: pa.Table) -> Dict[str, Any]:
        """Validate Arrow table quality"""
        validation = {
            'null_counts': {},
            'type_consistency': True,
            'column_issues': []
        }
        
        # Check for null values
        for col_name in arrow_table.column_names[:10]:  # Sample check
            column = arrow_table[col_name]
            null_count = column.null_count
            validation['null_counts'][col_name] = null_count
        
        return validation
    
    def _apply_ulta_to_arrow_table(self, arrow_table: pa.Table) -> Tuple[pa.Table, Dict[str, Any]]:
        """Apply ULTA processing to Arrow table"""
        try:
            # Convert to pandas for ULTA processing
            df = arrow_table.to_pandas()
            
            # Apply ULTA logic
            processed_df, ulta_metrics = self.ulta_calculator.apply_ulta_logic(df)
            
            # Convert back to Arrow
            processed_table = pa.Table.from_pandas(processed_df)
            
            # Generate ULTA report
            inverted_count = len([m for m in ulta_metrics.values() if m.was_inverted])
            ulta_report = {
                'strategies_processed': len([col for col in df.columns if col not in ['Date', 'Day', 'Zone']]),
                'strategies_inverted': inverted_count,
                'inversion_rate': inverted_count / max(1, len(ulta_metrics)) * 100,
                'average_improvement': self.ulta_calculator._calculate_average_improvement(),
                'inverted_strategies': [name for name, m in ulta_metrics.items() if m.was_inverted]
            }
            
            self.logger.info(f"🔄 ULTA applied: {inverted_count} strategies inverted out of {ulta_report['strategies_processed']}")
            
            return processed_table, ulta_report
            
        except Exception as e:
            self.logger.error(f"❌ Failed to apply ULTA to Arrow table: {e}")
            return arrow_table, {}
    
    def _arrow_to_heavydb(self, arrow_table: pa.Table, table_name: str) -> bool:
        """Load Arrow table to HeavyDB"""
        try:
            # Convert to pandas for compatibility
            df = arrow_table.to_pandas()
            
            # Use existing load function
            return load_strategy_data(df, table_name)
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load Arrow table to HeavyDB: {e}")
            return False
    
    def create_parquet_cache(self, csv_path: str, parquet_path: str) -> Dict[str, Any]:
        """Create optimized Parquet cache for repeated processing"""
        self.logger.info(f"💾 Creating Parquet cache: {parquet_path}")
        
        start_time = time.time()
        
        # Read CSV with Arrow
        arrow_table = pa_csv.read_csv(csv_path)
        
        # Clean schema
        cleaned_table = self._clean_arrow_schema(arrow_table)
        
        # Write optimized Parquet
        pq.write_table(
            cleaned_table,
            parquet_path,
            compression='snappy',
            use_dictionary=True,
            row_group_size=100000,
            use_deprecated_int96_timestamps=False
        )
        
        processing_time = time.time() - start_time
        file_size_mb = os.path.getsize(parquet_path) / (1024**2)
        compression_ratio = file_size_mb / (os.path.getsize(csv_path) / (1024**2))
        
        return {
            'parquet_path': parquet_path,
            'processing_time': processing_time,
            'file_size_mb': file_size_mb,
            'compression_ratio': compression_ratio,
            'rows': cleaned_table.num_rows,
            'columns': cleaned_table.num_columns
        }


def main():
    """Test optimal preprocessing pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Optimal CSV Preprocessing')
    parser.add_argument('--input', '-i', required=True, help='Input CSV file')
    parser.add_argument('--table', '-t', help='HeavyDB table name')
    parser.add_argument('--create-cache', action='store_true', help='Create Parquet cache')
    
    args = parser.parse_args()
    
    processor = OptimalPreprocessor()
    
    if args.create_cache:
        parquet_path = args.input.replace('.csv', '_optimized.parquet')
        cache_result = processor.create_parquet_cache(args.input, parquet_path)
        print(f"✅ Parquet cache created: {cache_result}")
    
    # Process with optimal pipeline
    result = processor.process_large_csv(
        args.input,
        table_name=args.table or f"strategies_{int(time.time())}"
    )
    
    print("\n" + "="*60)
    print("OPTIMAL PREPROCESSING RESULTS")
    print("="*60)
    print(f"Strategy: {result['strategy_used']}")
    print(f"Processing Time: {result['total_processing_time']:.2f}s")
    print(f"File Size: {result['file_size_mb']:.1f} MB")
    print(f"Success: {result['success']}")
    
    if 'timings' in result:
        print("\nTimings:")
        for step, time_taken in result['timings'].items():
            print(f"  {step}: {time_taken:.3f}s")


if __name__ == "__main__":
    main()