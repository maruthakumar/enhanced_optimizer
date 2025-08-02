"""
DAL Factory for creating appropriate Data Access Layer instances

This factory handles the creation of DAL instances based on configuration
and availability of HeavyDB.
"""

import os
import logging
import configparser
from typing import Optional, Dict, Any

from .base_dal import BaseDAL
from .heavydb_dal import HeavyDBDAL
from .csv_dal import CSVDAL


class DALFactory:
    """
    Factory class for creating DAL instances
    
    Automatically selects the appropriate DAL implementation based on
    configuration and system capabilities.
    """
    
    @staticmethod
    def create_dal(dal_type: Optional[str] = None, 
                   config_path: Optional[str] = None) -> BaseDAL:
        """
        Create a DAL instance
        
        Args:
            dal_type: Type of DAL to create ('heavydb', 'csv', or None for auto)
            config_path: Path to configuration file (optional)
            
        Returns:
            BaseDAL: Appropriate DAL implementation
        """
        logger = logging.getLogger(__name__)
        
        # Load configuration
        config = DALFactory._load_config(config_path)
        
        # Determine DAL type
        if dal_type is None:
            dal_type = config.get('system', {}).get('dal_type', 'auto')
        
        if dal_type == 'auto':
            # Try HeavyDB first, fall back to CSV
            try:
                logger.info("Attempting to create HeavyDB DAL...")
                dal = HeavyDBDAL(config)
                if dal.connect():
                    logger.info("Successfully created HeavyDB DAL")
                    return dal
                else:
                    logger.warning("HeavyDB connection failed, falling back to CSV DAL")
            except Exception as e:
                logger.warning(f"HeavyDB DAL creation failed: {str(e)}, falling back to CSV DAL")
            
            # Fall back to CSV
            dal_type = 'csv'
        
        # Create requested DAL type
        if dal_type == 'heavydb':
            logger.info("Creating HeavyDB DAL")
            return HeavyDBDAL(config)
        elif dal_type == 'csv':
            logger.info("Creating CSV DAL")
            return CSVDAL(config)
        else:
            raise ValueError(f"Unknown DAL type: {dal_type}")
    
    @staticmethod
    def _load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Load configuration from file
        
        Args:
            config_path: Path to config file, or None to use default
            
        Returns:
            Dict: Configuration dictionary
        """
        config = {
            'system': {},
            'database': {},
            'performance': {},
            'algorithms': {}
        }
        
        # Use default path if not specified
        if config_path is None:
            config_path = '/mnt/optimizer_share/config/production_config.ini'
        
        if os.path.exists(config_path):
            parser = configparser.ConfigParser()
            parser.read(config_path)
            
            # System settings
            if 'system' in parser:
                config['system'] = dict(parser['system'])
            
            # Database settings
            if 'database' in parser:
                config['database'] = dict(parser['database'])
            
            # Performance settings
            if 'performance' in parser:
                config['performance'] = {
                    'batch_size': parser.getint('performance', 'batch_size', fallback=10000),
                    'gpu_memory_limit': parser.getint('performance', 'gpu_memory_limit', 
                                                     fallback=8 * 1024 * 1024 * 1024)
                }
            
            # Algorithm settings
            if 'algorithms' in parser:
                config['algorithms'] = dict(parser['algorithms'])
        
        return config


def get_dal(dal_type: Optional[str] = None) -> BaseDAL:
    """
    Convenience function to get a DAL instance
    
    Args:
        dal_type: Type of DAL ('heavydb', 'csv', or None for auto)
        
    Returns:
        BaseDAL: DAL instance
    """
    return DALFactory.create_dal(dal_type)