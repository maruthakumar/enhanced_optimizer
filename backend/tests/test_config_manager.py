#!/usr/bin/env python3
"""
Unit Tests for Configuration Manager

Tests the comprehensive configuration management system.
"""

import unittest
import tempfile
import os
import shutil
import configparser
from pathlib import Path
import sys

# Add backend to path
sys.path.append(str(Path(__file__).parent.parent))

from config.config_manager import ConfigurationManager, ConfigValidationResult, get_config_manager


class TestConfigurationManager(unittest.TestCase):
    """Test cases for Configuration Manager"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_config_path = os.path.join(self.temp_dir, 'test_config.ini')
        
        # Create a test configuration file
        self._create_test_config()
        
        # Create manager instance
        self.config_manager = ConfigurationManager(self.test_config_path)
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir)
    
    def _create_test_config(self):
        """Create a test configuration file"""
        config = configparser.ConfigParser()
        
        # Add test sections
        config['SYSTEM'] = {
            'platform_name': 'Test Platform',
            'version': '1.0'
        }
        
        config['ALGORITHM_PARAMETERS'] = {
            'ga_population_size': '50',
            'ga_mutation_rate': '0.2',
            'pso_swarm_size': '30',
            'sa_initial_temperature': '2000'
        }
        
        config['PORTFOLIO_OPTIMIZATION'] = {
            'min_portfolio_size': '5',
            'max_portfolio_size': '50',
            'default_portfolio_size': '20'
        }
        
        config['ZONE'] = {
            'enabled': 'true',
            'zone_count': '4',
            'zone_weights': '0.25,0.25,0.25,0.25'
        }
        
        config['DATABASE'] = {
            'host': 'testhost',
            'port': '5432',
            'user': 'testuser',
            'password': 'testpass'
        }
        
        with open(self.test_config_path, 'w') as f:
            config.write(f)
    
    def test_load_base_config(self):
        """Test loading base configuration"""
        self.assertTrue(self.config_manager.config.has_section('SYSTEM'))
        self.assertEqual(
            self.config_manager.get('SYSTEM', 'platform_name'),
            'Test Platform'
        )
    
    def test_get_with_defaults(self):
        """Test getting values with defaults"""
        # Existing value
        value = self.config_manager.get('ALGORITHM_PARAMETERS', 'ga_population_size')
        self.assertEqual(value, '50')
        
        # Missing value with default
        value = self.config_manager.get('ALGORITHM_PARAMETERS', 'missing_param', 'default')
        self.assertEqual(value, 'default')
        
        # Missing section with defaults
        value = self.config_manager.get('MISSING_SECTION', 'param', 'default')
        self.assertEqual(value, 'default')
    
    def test_typed_getters(self):
        """Test typed getter methods"""
        # Integer
        value = self.config_manager.getint('ALGORITHM_PARAMETERS', 'ga_population_size')
        self.assertEqual(value, 50)
        self.assertIsInstance(value, int)
        
        # Float
        value = self.config_manager.getfloat('ALGORITHM_PARAMETERS', 'ga_mutation_rate')
        self.assertEqual(value, 0.2)
        self.assertIsInstance(value, float)
        
        # Boolean
        value = self.config_manager.getboolean('ZONE', 'enabled')
        self.assertTrue(value)
        self.assertIsInstance(value, bool)
        
        # List
        value = self.config_manager.getlist('ZONE', 'zone_weights')
        self.assertEqual(value, ['0.25', '0.25', '0.25', '0.25'])
        self.assertIsInstance(value, list)
    
    def test_validation_success(self):
        """Test successful configuration validation"""
        result = self.config_manager.validate_config()
        
        # Should have warnings but no errors for test config
        self.assertTrue(len(result.errors) < len(result.warnings))
    
    def test_validation_missing_sections(self):
        """Test validation with missing required sections"""
        # Create minimal config
        minimal_config = ConfigurationManager()
        minimal_config.config = configparser.ConfigParser()
        
        result = minimal_config.validate_config()
        
        self.assertFalse(result.is_valid)
        self.assertGreater(len(result.missing_sections), 0)
        self.assertIn('ALGORITHMS', result.missing_sections)
    
    def test_algorithm_config(self):
        """Test getting algorithm-specific configuration"""
        ga_config = self.config_manager.get_algorithm_config('ga')
        
        self.assertIn('population_size', ga_config)
        self.assertIn('mutation_rate', ga_config)
        self.assertEqual(ga_config['population_size'], '50')
    
    def test_zone_config(self):
        """Test getting zone configuration"""
        zone_config = self.config_manager.get_zone_config()
        
        self.assertTrue(zone_config['enabled'])
        self.assertEqual(zone_config['zone_count'], 4)
        self.assertEqual(len(zone_config['zone_weights']), 4)
        self.assertAlmostEqual(sum(zone_config['zone_weights']), 1.0)
    
    def test_database_config(self):
        """Test getting database configuration"""
        db_config = self.config_manager.get_database_config()
        
        self.assertEqual(db_config['host'], 'testhost')
        self.assertEqual(db_config['port'], 5432)
        self.assertEqual(db_config['user'], 'testuser')
        self.assertEqual(db_config['password'], 'testpass')
    
    def test_job_config_override(self):
        """Test job-specific configuration override"""
        # Create job config
        job_config_path = os.path.join(self.temp_dir, 'job_config.ini')
        job_config = configparser.ConfigParser()
        job_config['ALGORITHM_PARAMETERS'] = {
            'ga_population_size': '100',  # Override
            'ga_elitism_rate': '0.2'      # New parameter
        }
        
        with open(job_config_path, 'w') as f:
            job_config.write(f)
        
        # Load job config
        self.config_manager.load_job_config(job_config_path)
        
        # Check override
        self.assertEqual(
            self.config_manager.getint('ALGORITHM_PARAMETERS', 'ga_population_size'),
            100
        )
        
        # Check new parameter
        self.assertEqual(
            self.config_manager.getfloat('ALGORITHM_PARAMETERS', 'ga_elitism_rate'),
            0.2
        )
    
    def test_save_config(self):
        """Test saving configuration"""
        # Modify config
        self.config_manager.config.set('SYSTEM', 'test_param', 'test_value')
        
        # Save to new file
        save_path = os.path.join(self.temp_dir, 'saved_config.ini')
        self.config_manager.save_config(save_path)
        
        # Load and verify
        saved_config = configparser.ConfigParser()
        saved_config.read(save_path)
        
        self.assertEqual(saved_config.get('SYSTEM', 'test_param'), 'test_value')
    
    def test_export_json(self):
        """Test exporting configuration as JSON"""
        json_path = os.path.join(self.temp_dir, 'config.json')
        self.config_manager.export_json(json_path)
        
        # Verify file exists
        self.assertTrue(os.path.exists(json_path))
        
        # Load and check
        import json
        with open(json_path, 'r') as f:
            json_config = json.load(f)
        
        self.assertIn('SYSTEM', json_config)
        self.assertIn('ALGORITHM_PARAMETERS', json_config)
    
    def test_validation_report(self):
        """Test validation report generation"""
        report = self.config_manager.get_validation_report()
        
        self.assertIn('Configuration Validation Report', report)
        self.assertIn('Status:', report)
    
    def test_default_values(self):
        """Test default value handling"""
        # Create empty config manager
        empty_manager = ConfigurationManager()
        empty_manager.config = configparser.ConfigParser()
        
        # Should return defaults
        self.assertEqual(
            empty_manager.getint('ALGORITHM_PARAMETERS', 'ga_population_size'),
            30  # Default value
        )
        
        self.assertEqual(
            empty_manager.getfloat('ALGORITHM_PARAMETERS', 'pso_inertia_weight'),
            0.9  # Default value
        )
    
    def test_cache_functionality(self):
        """Test configuration caching"""
        # First access
        value1 = self.config_manager.get('ALGORITHM_PARAMETERS', 'ga_population_size')
        
        # Modify underlying config (bypass cache)
        self.config_manager.config.set('ALGORITHM_PARAMETERS', 'ga_population_size', '999')
        
        # Second access should return cached value
        value2 = self.config_manager.get('ALGORITHM_PARAMETERS', 'ga_population_size')
        
        self.assertEqual(value1, value2)  # Should be same (cached)
    
    def test_singleton_pattern(self):
        """Test singleton configuration manager"""
        manager1 = get_config_manager()
        manager2 = get_config_manager()
        
        self.assertIs(manager1, manager2)  # Should be same instance


class TestConfigValidation(unittest.TestCase):
    """Test configuration validation logic"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config_manager = ConfigurationManager()
        self.config_manager.config = configparser.ConfigParser()
    
    def test_validate_portfolio_sizes(self):
        """Test portfolio size validation"""
        # Add invalid portfolio sizes
        self.config_manager.config.add_section('PORTFOLIO_OPTIMIZATION')
        self.config_manager.config.set('PORTFOLIO_OPTIMIZATION', 'min_portfolio_size', '0')
        self.config_manager.config.set('PORTFOLIO_OPTIMIZATION', 'max_portfolio_size', '0')
        self.config_manager.config.set('PORTFOLIO_OPTIMIZATION', 'default_portfolio_size', '100')
        
        result = self.config_manager.validate_config()
        
        self.assertFalse(result.is_valid)
        self.assertGreater(len(result.errors), 0)
        
        # Check for specific errors
        error_messages = ' '.join(result.errors)
        self.assertIn('min_portfolio_size', error_messages)
        self.assertIn('must be positive', error_messages)
    
    def test_validate_zone_weights(self):
        """Test zone weight validation"""
        # Add zone config with invalid weights
        self.config_manager.config.add_section('ZONE')
        self.config_manager.config.set('ZONE', 'zone_count', '3')
        self.config_manager.config.set('ZONE', 'zone_weights', '0.5,0.5')  # Wrong count
        
        result = self.config_manager.validate_config()
        
        self.assertFalse(result.is_valid)
        error_messages = ' '.join(result.errors)
        self.assertIn('zone weights', error_messages)
    
    def test_validate_numeric_parameters(self):
        """Test numeric parameter validation"""
        # Add non-numeric values
        self.config_manager.config.add_section('ALGORITHM_PARAMETERS')
        self.config_manager.config.set('ALGORITHM_PARAMETERS', 'ga_population_size', 'not_a_number')
        
        result = self.config_manager.validate_config()
        
        self.assertFalse(result.is_valid)
        error_messages = ' '.join(result.errors)
        self.assertIn('must be numeric', error_messages)


class TestConfigurationIntegration(unittest.TestCase):
    """Integration tests for configuration with algorithms"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, 'integration_config.ini')
        
        # Create comprehensive test config
        self._create_integration_config()
    
    def tearDown(self):
        """Clean up"""
        shutil.rmtree(self.temp_dir)
    
    def _create_integration_config(self):
        """Create integration test configuration"""
        config = configparser.ConfigParser()
        
        # All required sections
        for section in ConfigurationManager.REQUIRED_SECTIONS:
            config.add_section(section)
        
        # Add all algorithm parameters
        config['ALGORITHM_PARAMETERS'] = {
            'ga_population_size': '40',
            'ga_generations': '150',
            'pso_swarm_size': '35',
            'pso_iterations': '100',
            'sa_initial_temperature': '1500',
            'sa_cooling_rate': '0.97'
        }
        
        # Add other required parameters
        config['PORTFOLIO_OPTIMIZATION'] = {
            'min_portfolio_size': '10',
            'max_portfolio_size': '100',
            'default_portfolio_size': '35'
        }
        
        config['ZONE'] = {
            'enabled': 'true',
            'zone_count': '4',
            'zone_weights': '0.25,0.25,0.25,0.25'
        }
        
        config['ULTA'] = {
            'enabled': 'true',
            'roi_threshold': '0.0'
        }
        
        config['DATABASE'] = {
            'host': 'localhost',
            'port': '6274'
        }
        
        with open(self.config_path, 'w') as f:
            config.write(f)
    
    def test_algorithm_integration(self):
        """Test configuration integration with algorithms"""
        from algorithms import create_algorithm
        
        # Create algorithm with config
        ga = create_algorithm('ga', self.config_path)
        
        # Verify it loaded configuration
        self.assertEqual(ga.population_size, 40)
        self.assertEqual(ga.generations, 150)
    
    def test_full_validation(self):
        """Test full configuration validation"""
        manager = ConfigurationManager(self.config_path)
        result = manager.validate_config()
        
        # Should be mostly valid (may have warnings)
        self.assertEqual(len(result.missing_sections), 0)


if __name__ == '__main__':
    unittest.main()