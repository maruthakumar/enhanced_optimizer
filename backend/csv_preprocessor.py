#!/usr/bin/env python3
"""
Advanced CSV Preprocessor for Heavy Optimizer Platform
Handles various data quality issues and special characters in column names
"""

import os
import re
import sys
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import unicodedata

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class CSVPreprocessor:
    """
    Comprehensive CSV preprocessing to handle various data quality issues
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Column name cleaning rules
        self.column_replacements = {
            # Special characters to replace
            '%': 'pct',
            '$': 'dollar',
            '&': 'and',
            '@': 'at',
            '#': 'num',
            '!': '',
            '?': '',
            '*': 'star',
            '+': 'plus',
            '=': 'eq',
            '|': 'or',
            '\\': '_',
            '/': '_',
            '<': 'lt',
            '>': 'gt',
            '(': '_',
            ')': '_',
            '[': '_',
            ']': '_',
            '{': '_',
            '}': '_',
            "'": '',
            '"': '',
            '`': '',
            '~': '',
            '^': '',
            ',': '_',
            ';': '_',
            ':': '_',
            '.': '_',
            ' ': '_',
            '-': '_',
            '\t': '_',
            '\n': '_',
            '\r': '_'
        }
        
        # SQL reserved keywords that need special handling
        self.sql_reserved_keywords = {
            'date', 'time', 'timestamp', 'day', 'month', 'year', 'hour', 'minute', 'second',
            'user', 'table', 'column', 'select', 'from', 'where', 'group', 'order', 'by',
            'insert', 'update', 'delete', 'create', 'drop', 'alter', 'index', 'key',
            'primary', 'foreign', 'references', 'constraint', 'unique', 'default', 'null',
            'not', 'and', 'or', 'in', 'exists', 'between', 'like', 'limit', 'offset',
            'union', 'all', 'distinct', 'having', 'join', 'inner', 'outer', 'left', 'right',
            'full', 'cross', 'using', 'on', 'as', 'case', 'when', 'then', 'else', 'end',
            'count', 'sum', 'avg', 'min', 'max', 'row', 'rows', 'value', 'values',
            'into', 'set', 'begin', 'commit', 'rollback', 'transaction', 'trigger',
            'procedure', 'function', 'return', 'returns', 'declare', 'cursor', 'for',
            'while', 'loop', 'if', 'elseif', 'endif', 'repeat', 'until', 'goto', 'exit'
        }
        
        # Data type detection patterns
        self.date_patterns = [
            r'^\d{4}-\d{2}-\d{2}$',  # YYYY-MM-DD
            r'^\d{2}/\d{2}/\d{4}$',  # MM/DD/YYYY
            r'^\d{2}-\d{2}-\d{4}$',  # DD-MM-YYYY
            r'^\d{1,2}/\d{1,2}/\d{2,4}$'  # M/D/YY or MM/DD/YYYY
        ]
        
        # Numeric cleaning patterns
        self.numeric_cleaning_patterns = [
            (r'[\$,]', ''),  # Remove currency symbols and commas
            (r'[^\d.-]', ''),  # Keep only digits, dots, and minus
            (r'^-\.$', '0'),  # Convert lone minus-dot to zero
            (r'^\.$', '0'),  # Convert lone dot to zero
        ]
    
    def preprocess_csv(self, file_path: str, 
                      output_path: Optional[str] = None,
                      validate_data: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Comprehensive CSV preprocessing
        
        Args:
            file_path: Input CSV file path
            output_path: Optional output path for cleaned CSV
            validate_data: Whether to perform data validation
            
        Returns:
            Tuple of (cleaned DataFrame, preprocessing report)
        """
        self.logger.info(f"📊 Starting CSV preprocessing: {file_path}")
        
        # Initialize report
        report = {
            'original_file': file_path,
            'preprocessing_time': None,
            'original_columns': 0,
            'cleaned_columns': 0,
            'renamed_columns': {},
            'data_issues': [],
            'numeric_conversions': 0,
            'null_replacements': 0,
            'duplicate_columns': [],
            'validation_errors': []
        }
        
        start_time = datetime.now()
        
        try:
            # Step 1: Read CSV with various encoding attempts
            df = self._read_csv_safely(file_path)
            report['original_columns'] = len(df.columns)
            report['original_shape'] = df.shape
            
            # Step 2: Clean column names
            df, column_mapping = self._clean_column_names(df)
            report['renamed_columns'] = column_mapping
            report['cleaned_columns'] = len(df.columns)
            
            # Step 3: Handle duplicate columns
            df, duplicates = self._handle_duplicate_columns(df)
            report['duplicate_columns'] = duplicates
            
            # Step 4: Clean data values
            df, data_report = self._clean_data_values(df)
            report.update(data_report)
            
            # Step 5: Data type optimization
            df = self._optimize_data_types(df)
            
            # Step 6: Validate data if requested
            if validate_data:
                validation_report = self._validate_data(df)
                report['validation_errors'] = validation_report
            
            # Step 7: Sort columns for consistency
            df = self._sort_columns(df)
            
            # Save cleaned CSV if output path provided
            if output_path:
                df.to_csv(output_path, index=False)
                report['output_file'] = output_path
                self.logger.info(f"✅ Saved cleaned CSV to: {output_path}")
            
            # Final report
            report['preprocessing_time'] = (datetime.now() - start_time).total_seconds()
            report['final_shape'] = df.shape
            
            self.logger.info(f"✅ Preprocessing completed in {report['preprocessing_time']:.2f}s")
            
            return df, report
            
        except Exception as e:
            self.logger.error(f"❌ Preprocessing failed: {e}")
            report['error'] = str(e)
            raise
    
    def _read_csv_safely(self, file_path: str) -> pd.DataFrame:
        """Read CSV with multiple encoding attempts"""
        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'utf-16']
        
        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                self.logger.info(f"✅ Successfully read CSV with {encoding} encoding")
                return df
            except UnicodeDecodeError:
                continue
            except Exception as e:
                if encoding == encodings[-1]:
                    raise Exception(f"Failed to read CSV with any encoding: {e}")
        
        raise Exception("Unable to read CSV file with any supported encoding")
    
    def _clean_column_names(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """Clean column names for SQL compatibility"""
        column_mapping = {}
        new_columns = []
        
        for col in df.columns:
            # Convert to string and strip whitespace
            original = str(col).strip()
            
            # Remove Unicode characters
            cleaned = unicodedata.normalize('NFKD', original)
            cleaned = cleaned.encode('ascii', 'ignore').decode('ascii')
            
            # Apply character replacements
            for old_char, new_char in self.column_replacements.items():
                cleaned = cleaned.replace(old_char, new_char)
            
            # Remove multiple underscores
            cleaned = re.sub(r'_+', '_', cleaned)
            
            # Remove leading/trailing underscores
            cleaned = cleaned.strip('_')
            
            # Handle empty column names
            if not cleaned:
                cleaned = f'column_{len(new_columns)}'
            
            # Ensure column starts with letter or underscore
            if cleaned[0].isdigit():
                cleaned = f'col_{cleaned}'
            
            # Convert to lowercase for consistency
            cleaned = cleaned.lower()
            
            # Handle SQL reserved keywords
            if cleaned in self.sql_reserved_keywords:
                cleaned = f'{cleaned}_col'
            
            # Ensure uniqueness
            if cleaned in new_columns:
                counter = 1
                while f'{cleaned}_{counter}' in new_columns:
                    counter += 1
                cleaned = f'{cleaned}_{counter}'
            
            new_columns.append(cleaned)
            
            if original != cleaned:
                column_mapping[original] = cleaned
        
        df.columns = new_columns
        
        self.logger.info(f"📝 Renamed {len(column_mapping)} columns")
        
        return df, column_mapping
    
    def _handle_duplicate_columns(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Handle duplicate column names"""
        duplicates = []
        columns_seen = {}
        
        for col in df.columns:
            if col in columns_seen:
                duplicates.append(col)
            else:
                columns_seen[col] = True
        
        if duplicates:
            self.logger.warning(f"⚠️ Found {len(duplicates)} duplicate columns: {duplicates[:5]}...")
            
            # Remove exact duplicates (same name and values)
            df = df.loc[:, ~df.columns.duplicated()]
        
        return df, duplicates
    
    def _clean_data_values(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Clean data values in all columns"""
        report = {
            'numeric_conversions': 0,
            'null_replacements': 0,
            'trimmed_strings': 0,
            'invalid_values': []
        }
        
        for col in df.columns:
            # Skip date columns
            if col.lower() in ['date', 'day', 'date_col', 'day_col']:
                continue
            
            # Try to convert to numeric
            if df[col].dtype == 'object':
                # Clean numeric values
                original_nulls = df[col].isna().sum()
                
                # Try numeric conversion
                try:
                    # Remove common non-numeric characters
                    cleaned_col = df[col].astype(str)
                    for pattern, replacement in self.numeric_cleaning_patterns:
                        cleaned_col = cleaned_col.str.replace(pattern, replacement, regex=True)
                    
                    # Convert to numeric
                    numeric_col = pd.to_numeric(cleaned_col, errors='coerce')
                    
                    # Check if conversion is meaningful (>50% success rate)
                    success_rate = numeric_col.notna().sum() / len(numeric_col)
                    
                    if success_rate > 0.5:
                        df[col] = numeric_col
                        report['numeric_conversions'] += 1
                        
                        # Fill NaN values with 0 for numeric columns
                        null_count = df[col].isna().sum() - original_nulls
                        if null_count > 0:
                            df[col] = df[col].fillna(0)
                            report['null_replacements'] += null_count
                    else:
                        # Keep as string but clean it
                        df[col] = df[col].astype(str).str.strip()
                        report['trimmed_strings'] += 1
                        
                except Exception as e:
                    self.logger.debug(f"Could not convert column {col}: {e}")
                    report['invalid_values'].append(col)
            
            # Handle existing numeric columns
            elif pd.api.types.is_numeric_dtype(df[col]):
                # Replace infinities with NaN then with 0
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)
                null_count = df[col].isna().sum()
                if null_count > 0:
                    df[col] = df[col].fillna(0)
                    report['null_replacements'] += null_count
        
        return df, report
    
    def _optimize_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize data types for memory efficiency"""
        for col in df.columns:
            col_type = df[col].dtype
            
            # Optimize numeric types
            if pd.api.types.is_numeric_dtype(df[col]):
                # Check if integer
                if df[col].dtype in ['float64', 'float32']:
                    if df[col].fillna(0).apply(lambda x: x.is_integer() if pd.notna(x) else True).all():
                        try:
                            df[col] = df[col].fillna(0).astype('int64')
                        except:
                            pass
                
                # Downcast numeric types
                if df[col].dtype == 'int64':
                    try:
                        df[col] = pd.to_numeric(df[col], downcast='integer')
                    except:
                        pass
                elif df[col].dtype == 'float64':
                    try:
                        df[col] = pd.to_numeric(df[col], downcast='float')
                    except:
                        pass
        
        return df
    
    def _validate_data(self, df: pd.DataFrame) -> List[str]:
        """Validate data quality"""
        errors = []
        
        # Check for empty dataframe
        if df.empty:
            errors.append("DataFrame is empty")
        
        # Check for all-null columns
        null_columns = df.columns[df.isna().all()].tolist()
        if null_columns:
            errors.append(f"All-null columns found: {null_columns}")
        
        # Check for constant columns (excluding date/day)
        non_date_cols = [col for col in df.columns if col.lower() not in ['date', 'day', 'date_col', 'day_col']]
        constant_cols = []
        for col in non_date_cols:
            if df[col].nunique() <= 1:
                constant_cols.append(col)
        
        if constant_cols:
            errors.append(f"Constant value columns found: {constant_cols[:10]}...")
        
        # Check for extreme values in numeric columns
        for col in df.select_dtypes(include=[np.number]).columns:
            col_std = df[col].std()
            col_mean = df[col].mean()
            
            if col_std > 0:
                # Check for outliers (> 10 standard deviations)
                outliers = df[abs(df[col] - col_mean) > 10 * col_std]
                if len(outliers) > 0:
                    errors.append(f"Extreme outliers in {col}: {len(outliers)} values")
        
        return errors
    
    def _sort_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sort columns for consistency"""
        # Separate date/day columns and strategy columns
        date_cols = []
        strategy_cols = []
        other_cols = []
        
        for col in df.columns:
            if col.lower() in ['date', 'day', 'date_col', 'day_col']:
                date_cols.append(col)
            elif 'strategy' in col.lower() or 'sensex' in col.lower():
                strategy_cols.append(col)
            else:
                other_cols.append(col)
        
        # Sort strategy columns
        strategy_cols.sort()
        other_cols.sort()
        
        # Reorder columns: date/day first, then strategies, then others
        new_order = date_cols + strategy_cols + other_cols
        
        return df[new_order]
    
    def create_heavydb_schema(self, df: pd.DataFrame, 
                            table_name: str = "strategies") -> str:
        """
        Create HeavyDB-compatible CREATE TABLE statement
        
        Args:
            df: Preprocessed DataFrame
            table_name: Name for the table
            
        Returns:
            CREATE TABLE SQL statement
        """
        columns = []
        
        for col in df.columns:
            dtype = df[col].dtype
            
            # Determine SQL type
            if pd.api.types.is_integer_dtype(dtype):
                sql_type = "BIGINT"
            elif pd.api.types.is_float_dtype(dtype):
                sql_type = "DOUBLE"
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                sql_type = "DATE"
            elif col.lower() in ['date', 'date_col']:
                sql_type = "DATE"
            else:
                # Use dictionary encoding for text
                sql_type = "TEXT ENCODING DICT(32)"
            
            # Quote column name if needed
            if col.lower() in self.sql_reserved_keywords:
                col_sql = f'"{col}"'
            else:
                col_sql = col
            
            columns.append(f"    {col_sql} {sql_type}")
        
        # Create table statement with GPU optimization
        columns_str = ',\n'.join(columns)
        create_sql = f"""
CREATE TABLE {table_name} (
{columns_str}
) WITH (
    fragment_size = 75000000,
    max_chunk_size = 1000000000
);
"""
        
        return create_sql
    
    def generate_preprocessing_report(self, report: Dict[str, Any]) -> str:
        """Generate a formatted preprocessing report"""
        lines = [
            "=" * 80,
            "CSV PREPROCESSING REPORT",
            "=" * 80,
            "",
            f"Original File: {report.get('original_file', 'N/A')}",
            f"Processing Time: {report.get('preprocessing_time', 0):.2f} seconds",
            f"Original Shape: {report.get('original_shape', 'N/A')}",
            f"Final Shape: {report.get('final_shape', 'N/A')}",
            "",
            "COLUMN CLEANING:",
            f"  - Original Columns: {report.get('original_columns', 0)}",
            f"  - Cleaned Columns: {report.get('cleaned_columns', 0)}",
            f"  - Renamed Columns: {len(report.get('renamed_columns', {}))}",
            ""
        ]
        
        # Show sample of renamed columns
        renamed = report.get('renamed_columns', {})
        if renamed:
            lines.append("  Sample Renamed Columns:")
            for i, (old, new) in enumerate(list(renamed.items())[:5]):
                lines.append(f"    - '{old}' -> '{new}'")
            if len(renamed) > 5:
                lines.append(f"    ... and {len(renamed) - 5} more")
            lines.append("")
        
        lines.extend([
            "DATA CLEANING:",
            f"  - Numeric Conversions: {report.get('numeric_conversions', 0)}",
            f"  - Null Replacements: {report.get('null_replacements', 0)}",
            f"  - Trimmed String Columns: {report.get('trimmed_strings', 0)}",
            ""
        ])
        
        # Validation errors
        errors = report.get('validation_errors', [])
        if errors:
            lines.append("VALIDATION ISSUES:")
            for error in errors:
                lines.append(f"  - {error}")
            lines.append("")
        
        # Duplicate columns
        duplicates = report.get('duplicate_columns', [])
        if duplicates:
            lines.append(f"DUPLICATE COLUMNS FOUND: {len(duplicates)}")
            for dup in duplicates[:5]:
                lines.append(f"  - {dup}")
            if len(duplicates) > 5:
                lines.append(f"  ... and {len(duplicates) - 5} more")
            lines.append("")
        
        lines.append("=" * 80)
        
        return "\n".join(lines)


def test_preprocessor():
    """Test the CSV preprocessor"""
    print("🧪 Testing CSV Preprocessor")
    print("=" * 60)
    
    preprocessor = CSVPreprocessor()
    
    # Test with production data
    input_file = "/mnt/optimizer_share/input/Python_Multi_Consolidated_20250726_161921.csv"
    output_file = "/mnt/optimizer_share/output/preprocessed_data.csv"
    
    try:
        # Run preprocessing
        df, report = preprocessor.preprocess_csv(
            input_file,
            output_file,
            validate_data=True
        )
        
        # Print report
        print(preprocessor.generate_preprocessing_report(report))
        
        # Show sample of cleaned data
        print("\nSAMPLE CLEANED DATA:")
        print(df.head())
        
        # Generate and show CREATE TABLE statement
        create_sql = preprocessor.create_heavydb_schema(df)
        print("\nHEAVYDB CREATE TABLE STATEMENT:")
        print(create_sql[:500] + "..." if len(create_sql) > 500 else create_sql)
        
        print("\n✅ Preprocessing test completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_preprocessor()