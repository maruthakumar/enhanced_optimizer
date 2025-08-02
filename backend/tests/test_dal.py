"""
Unit tests for Data Access Layer (DAL) implementations

Tests both HeavyDB and CSV DAL implementations to ensure they
conform to the BaseDAL interface.
"""

import unittest
import os
import sys
import tempfile
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dal import get_dal, CSVDAL, HeavyDBDAL
from dal.base_dal import BaseDAL


class TestCSVDAL(unittest.TestCase):
    """Test CSV DAL implementation"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.dal = CSVDAL()
        self.dal.connect()
        
        # Create test data
        self.test_data = pd.DataFrame({
            'strategy_1': np.random.random(100),
            'strategy_2': np.random.random(100),
            'strategy_3': np.random.random(100),
            'strategy_4': np.random.random(100),
            'strategy_5': np.random.random(100)
        })
        
        # Create temporary CSV file
        self.temp_csv = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        self.test_data.to_csv(self.temp_csv.name, index=False)
        self.temp_csv.close()
    
    def tearDown(self):
        """Clean up test fixtures"""
        self.dal.disconnect()
        if os.path.exists(self.temp_csv.name):
            os.unlink(self.temp_csv.name)
    
    def test_connection(self):
        """Test connection functionality"""
        self.assertTrue(self.dal.is_connected)
        self.assertFalse(self.dal.supports_gpu)
    
    def test_load_csv(self):
        """Test CSV loading"""
        success = self.dal.load_csv_to_heavydb(self.temp_csv.name, 'test_table')
        self.assertTrue(success)
        
        # Verify table exists
        self.assertIn('test_table', self.dal.list_tables())
        
        # Verify row count
        row_count = self.dal.get_table_row_count('test_table')
        self.assertEqual(row_count, 100)
    
    def test_table_schema(self):
        """Test schema retrieval"""
        self.dal.load_csv_to_heavydb(self.temp_csv.name, 'test_table')
        
        schema = self.dal.get_table_schema('test_table')
        self.assertIsNotNone(schema)
        self.assertEqual(len(schema), 5)
        self.assertIn('strategy_1', schema)
        self.assertEqual(schema['strategy_1'], 'DOUBLE')
    
    def test_ulta_transformation(self):
        """Test ULTA transformation"""
        self.dal.load_csv_to_heavydb(self.temp_csv.name, 'test_table')
        
        success = self.dal.apply_ulta_transformation('test_table')
        self.assertTrue(success)
        
        # Verify ULTA table created
        self.assertIn('test_table_ulta', self.dal.list_tables())
    
    def test_correlation_matrix(self):
        """Test correlation matrix computation"""
        self.dal.load_csv_to_heavydb(self.temp_csv.name, 'test_table')
        
        corr_matrix = self.dal.compute_correlation_matrix('test_table')
        self.assertIsNotNone(corr_matrix)
        self.assertEqual(corr_matrix.shape, (5, 5))
        
        # Check diagonal is 1.0
        np.testing.assert_array_almost_equal(np.diag(corr_matrix), np.ones(5))
    
    def test_strategy_subset(self):
        """Test strategy subset retrieval"""
        self.dal.load_csv_to_heavydb(self.temp_csv.name, 'test_table')
        
        subset = self.dal.get_strategy_subset('test_table', [0, 2, 4])
        self.assertIsNotNone(subset)
        self.assertEqual(len(subset.columns), 3)
        self.assertIn('strategy_1', subset.columns)
        self.assertIn('strategy_3', subset.columns)
        self.assertIn('strategy_5', subset.columns)
    
    def test_create_dynamic_table(self):
        """Test dynamic table creation"""
        new_data = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c'],
            'col3': [1.1, 2.2, 3.3]
        })
        
        success = self.dal.create_dynamic_table('dynamic_table', new_data)
        self.assertTrue(success)
        
        # Verify table exists and has correct schema
        schema = self.dal.get_table_schema('dynamic_table')
        self.assertEqual(len(schema), 3)
        self.assertEqual(schema['col1'], 'INTEGER')
        self.assertEqual(schema['col2'], 'TEXT')
        self.assertEqual(schema['col3'], 'DOUBLE')
    
    def test_drop_table(self):
        """Test table dropping"""
        self.dal.load_csv_to_heavydb(self.temp_csv.name, 'test_table')
        
        success = self.dal.drop_table('test_table')
        self.assertTrue(success)
        
        # Verify table no longer exists
        self.assertNotIn('test_table', self.dal.list_tables())
    
    def test_save_table_to_csv(self):
        """Test saving table to CSV"""
        self.dal.load_csv_to_heavydb(self.temp_csv.name, 'test_table')
        
        output_file = tempfile.NamedTemporaryFile(suffix='.csv', delete=False)
        output_file.close()
        
        try:
            success = self.dal.save_table_to_csv('test_table', output_file.name)
            self.assertTrue(success)
            
            # Verify file exists and has correct data
            loaded_data = pd.read_csv(output_file.name)
            self.assertEqual(len(loaded_data), 100)
            self.assertEqual(len(loaded_data.columns), 5)
        finally:
            if os.path.exists(output_file.name):
                os.unlink(output_file.name)


class TestDALFactory(unittest.TestCase):
    """Test DAL Factory functionality"""
    
    def test_create_csv_dal(self):
        """Test creating CSV DAL explicitly"""
        dal = get_dal('csv')
        self.assertIsInstance(dal, CSVDAL)
        self.assertIsInstance(dal, BaseDAL)
    
    def test_auto_dal_selection(self):
        """Test automatic DAL selection"""
        dal = get_dal()  # Should fall back to CSV
        self.assertIsInstance(dal, BaseDAL)
        # In test environment, should get CSV DAL
        self.assertFalse(dal.supports_gpu)


class TestDALInterface(unittest.TestCase):
    """Test that all DAL implementations conform to interface"""
    
    def test_interface_methods(self):
        """Verify all required methods are present"""
        required_methods = [
            'connect', 'disconnect', 'load_csv_to_heavydb',
            'apply_ulta_transformation', 'compute_correlation_matrix',
            'get_strategy_subset', 'execute_gpu_query', 'get_table_schema',
            'create_dynamic_table', 'get_table_row_count', 'drop_table'
        ]
        
        # Test CSV DAL
        csv_dal = CSVDAL()
        for method in required_methods:
            self.assertTrue(hasattr(csv_dal, method))
            self.assertTrue(callable(getattr(csv_dal, method)))
        
        # Would test HeavyDB DAL here if HeavyDB was available


if __name__ == '__main__':
    unittest.main()