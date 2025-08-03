"""
Data Access Layer (DAL) for Heavy Optimizer Platform

This module provides a clean abstraction for data operations,
supporting CSV data processing with GPU acceleration via Parquet/Arrow/cuDF.
"""

from .base_dal import BaseDAL
from .csv_dal import CSVDAL
from .dal_factory import DALFactory, get_dal

__all__ = ['BaseDAL', 'CSVDAL', 'DALFactory', 'get_dal']