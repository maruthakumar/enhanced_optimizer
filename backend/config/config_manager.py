"""
Configuration Manager for algorithms
"""

import configparser
import os
from typing import Any

class ConfigManager:
    """Manages configuration for algorithms"""
    
    def __init__(self, config_file: str = None):
        """Initialize configuration manager"""
        if config_file is None:
            config_file = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                'config',
                'production_config.ini'
            )
        
        self.config = configparser.ConfigParser()
        self.config.read(config_file)
    
    def get(self, section: str, option: str, fallback: Any = None) -> str:
        """Get configuration value"""
        try:
            return self.config.get(section, option)
        except:
            return fallback
    
    def getint(self, section: str, option: str, fallback: int = 0) -> int:
        """Get integer configuration value"""
        try:
            return self.config.getint(section, option)
        except:
            return fallback
    
    def getfloat(self, section: str, option: str, fallback: float = 0.0) -> float:
        """Get float configuration value"""
        try:
            return self.config.getfloat(section, option)
        except:
            return fallback
    
    def getboolean(self, section: str, option: str, fallback: bool = False) -> bool:
        """Get boolean configuration value"""
        try:
            return self.config.getboolean(section, option)
        except:
            return fallback

# Singleton instance
_config_manager = None

def get_config_manager():
    """Get config manager instance"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager