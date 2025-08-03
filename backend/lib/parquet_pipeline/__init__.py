"""
Parquet Pipeline Module
Handles CSV to Parquet conversion and storage optimization
"""

from .csv_to_parquet import (
    csv_to_parquet,
    detect_csv_schema,
    validate_parquet_file,
    get_parquet_metadata,
    optimize_parquet_storage
)

__all__ = [
    'csv_to_parquet',
    'detect_csv_schema',
    'validate_parquet_file',
    'get_parquet_metadata',
    'optimize_parquet_storage'
]