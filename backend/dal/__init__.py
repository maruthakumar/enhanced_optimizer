"""
Data Access Layer (DAL) for Heavy Optimizer Platform

This module provides a clean abstraction for all database operations,
supporting both HeavyDB GPU-accelerated operations and CSV fallback mode.
"""

from .base_dal import BaseDAL
from .heavydb_dal import HeavyDBDAL
from .csv_dal import CSVDAL
from .dal_factory import DALFactory, get_dal

__all__ = ['BaseDAL', 'HeavyDBDAL', 'CSVDAL', 'DALFactory', 'get_dal']