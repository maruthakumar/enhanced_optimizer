"""
Configuration Reader for GPU Settings
"""

import os
import configparser
from typing import Dict, Any, Optional


class GPUConfigReader:
    """Read GPU configuration from production config"""
    
    def __init__(self, config_file: Optional[str] = None):
        """Initialize configuration reader"""
        if config_file is None:
            config_file = "/mnt/optimizer_share/config/production_config.ini"
        
        self.config = configparser.ConfigParser()
        self.config.read(config_file)
    
    def get_gpu_config(self) -> Dict[str, Any]:
        """Get GPU configuration settings"""
        gpu_config = {}
        
        if self.config.has_section('GPU'):
            # Basic settings
            gpu_config['enabled'] = self.config.getboolean('GPU', 'heavydb_enabled', fallback=True)
            gpu_config['acceleration'] = self.config.get('GPU', 'gpu_acceleration', fallback='optional')
            gpu_config['cpu_fallback_allowed'] = self.config.getboolean('GPU', 'cpu_fallback_allowed', fallback=True)
            gpu_config['force_gpu_mode'] = self.config.getboolean('GPU', 'force_gpu_mode', fallback=False)
            
            # Connection settings
            gpu_config['heavydb_host'] = self.config.get('GPU', 'heavydb_host', fallback='127.0.0.1')
            gpu_config['heavydb_port'] = self.config.getint('GPU', 'heavydb_port', fallback=6274)
            gpu_config['heavydb_database'] = self.config.get('GPU', 'heavydb_database', fallback='portfolio_optimizer')
            gpu_config['heavydb_user'] = self.config.get('GPU', 'heavydb_user', fallback='admin')
            gpu_config['heavydb_password'] = self.config.get('GPU', 'heavydb_password', fallback='')
            gpu_config['heavydb_protocol'] = self.config.get('GPU', 'heavydb_protocol', fallback='binary')
            
            # Processing settings
            gpu_config['gpu_correlation_calculation'] = self.config.getboolean('GPU', 'gpu_correlation_calculation', fallback=True)
            gpu_config['min_strategies_for_gpu'] = self.config.getint('GPU', 'min_strategies_for_gpu', fallback=10)
            
            # Error handling
            gpu_config['fail_on_gpu_error'] = self.config.getboolean('GPU', 'fail_on_gpu_error', fallback=False)
            gpu_config['retry_on_connection_failure'] = self.config.getboolean('GPU', 'retry_on_connection_failure', fallback=True)
            gpu_config['max_connection_retries'] = self.config.getint('GPU', 'max_connection_retries', fallback=3)
        else:
            # Default configuration if no GPU section
            gpu_config = {
                'enabled': True,
                'acceleration': 'optional',
                'cpu_fallback_allowed': True,
                'force_gpu_mode': False,
                'heavydb_host': '127.0.0.1',
                'heavydb_port': 6274,
                'heavydb_database': 'portfolio_optimizer',
                'heavydb_user': 'admin',
                'heavydb_password': '',
                'heavydb_protocol': 'binary',
                'gpu_correlation_calculation': True,
                'min_strategies_for_gpu': 10,
                'fail_on_gpu_error': False,
                'retry_on_connection_failure': True,
                'max_connection_retries': 3
            }
        
        # Override with environment variables if present
        env_mappings = {
            'HEAVYDB_HOST': 'heavydb_host',
            'HEAVYDB_PORT': ('heavydb_port', int),
            'HEAVYDB_USER': 'heavydb_user',
            'HEAVYDB_PASSWORD': 'heavydb_password',
            'HEAVYDB_DATABASE': 'heavydb_database',
            'GPU_FALLBACK_ALLOWED': ('cpu_fallback_allowed', lambda x: x.lower() == 'true'),
            'FORCE_GPU_MODE': ('force_gpu_mode', lambda x: x.lower() == 'true')
        }
        
        for env_var, config_key in env_mappings.items():
            if env_var in os.environ:
                if isinstance(config_key, tuple):
                    key, converter = config_key
                    gpu_config[key] = converter(os.environ[env_var])
                else:
                    gpu_config[config_key] = os.environ[env_var]
        
        return gpu_config
    
    def is_gpu_required(self) -> bool:
        """Check if GPU processing is required (no CPU fallback)"""
        config = self.get_gpu_config()
        return (config['acceleration'] == 'required' or 
                config['force_gpu_mode'] or 
                not config['cpu_fallback_allowed'])
    
    def should_use_gpu(self, num_strategies: int) -> bool:
        """Determine if GPU should be used based on data size"""
        config = self.get_gpu_config()
        
        if not config['enabled']:
            return False
        
        if self.is_gpu_required():
            return True
        
        return num_strategies >= config['min_strategies_for_gpu']


# Singleton instance
_config_reader = None

def get_gpu_config() -> Dict[str, Any]:
    """Get GPU configuration (singleton)"""
    global _config_reader
    if _config_reader is None:
        _config_reader = GPUConfigReader()
    return _config_reader.get_gpu_config()

def is_gpu_required() -> bool:
    """Check if GPU is required"""
    global _config_reader
    if _config_reader is None:
        _config_reader = GPUConfigReader()
    return _config_reader.is_gpu_required()

def should_use_gpu(num_strategies: int) -> bool:
    """Check if GPU should be used"""
    global _config_reader
    if _config_reader is None:
        _config_reader = GPUConfigReader()
    return _config_reader.should_use_gpu(num_strategies)