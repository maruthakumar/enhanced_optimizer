"""
Unit tests for Dynamic Table Manager
"""

import unittest
import os
import sys
import tempfile
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dal.dynamic_table_manager import DynamicTableManager, ColumnSchema, TableSchema


class TestDynamicTableManager(unittest.TestCase):
    """Test Dynamic Table Manager functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.manager = DynamicTableManager()
        
        # Create test data with various column types
        self.test_data = pd.DataFrame({
            'Date': pd.date_range('2024-01-01', periods=100),
            'Zone': ['Zone_A', 'Zone_B', 'Zone_C', 'Zone_D'] * 25,
            'Index': range(100),
            'Strategy_1': np.random.randn(100).cumsum() + 100,
            'Strategy_2': np.random.randn(100).cumsum() + 100,
            'Strategy_3': np.random.randn(100).cumsum() + 100,
            'Text_Column': ['Category_' + str(i % 5) for i in range(100)],
            'Boolean_Flag': [True, False] * 50,
            'Integer_Values': np.random.randint(0, 1000, 100),
            'Nullable_Column': [None if i % 10 == 0 else i for i in range(100)]
        })
        
        # Create temporary CSV file
        self.temp_csv = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        self.test_data.to_csv(self.temp_csv.name, index=False)
        self.temp_csv.close()
    
    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.temp_csv.name):
            os.unlink(self.temp_csv.name)
    
    def test_schema_detection(self):
        """Test schema detection from CSV"""
        schema = self.manager.detect_schema_from_csv(self.temp_csv.name)
        
        self.assertIsNotNone(schema)
        self.assertEqual(schema.column_count, 12)  # 10 original + 2 metadata
        self.assertEqual(schema.strategy_column_count, 3)
        
        # Check table name sanitization
        self.assertIsInstance(schema.table_name, str)
        self.assertRegex(schema.table_name, r'^[a-z][a-z0-9_]*$')
    
    def test_column_analysis(self):
        """Test individual column analysis"""
        # Test date column
        date_series = self.test_data['Date']
        date_schema = self.manager._analyze_column('Date', date_series)
        self.assertEqual(date_schema.sql_type, 'TIMESTAMP')
        self.assertTrue(date_schema.has_special_handling)
        
        # Test strategy column
        strategy_series = self.test_data['Strategy_1']
        strategy_schema = self.manager._analyze_column('Strategy_1', strategy_series)
        self.assertEqual(strategy_schema.sql_type, 'DOUBLE')
        self.assertTrue(strategy_schema.metadata['is_strategy'])
        
        # Test text column
        text_series = self.test_data['Text_Column']
        text_schema = self.manager._analyze_column('Text_Column', text_series)
        self.assertIn('TEXT', text_schema.sql_type)
        
        # Test nullable column
        null_series = self.test_data['Nullable_Column']
        null_schema = self.manager._analyze_column('Nullable_Column', null_series)
        self.assertTrue(null_schema.is_nullable)
        self.assertEqual(null_schema.metadata['null_count'], 10)
    
    def test_column_name_sanitization(self):
        """Test column name sanitization"""
        test_cases = [
            ('Normal Column', 'normal_column'),
            ('Column With Spaces', 'column_with_spaces'),
            ('Column-With-Dashes', 'column_with_dashes'),
            ('Column@#$Special!', 'column_special'),
            ('123_Starting_With_Number', 'col_123_starting_with_number'),
            ('__Multiple___Underscores__', 'multiple_underscores'),
            ('', 'column'),
            ('A' * 100, 'a' * 63)  # Test truncation
        ]
        
        for original, expected in test_cases:
            sanitized = self.manager._sanitize_column_name(original)
            self.assertEqual(sanitized, expected, 
                           f"Failed for '{original}' -> expected '{expected}', got '{sanitized}'")
    
    def test_data_type_mapping(self):
        """Test data type mapping"""
        # Create series with specific types
        test_series = {
            'int8': pd.Series([1, 2, 3], dtype='int8'),
            'int64': pd.Series([1, 2, 3], dtype='int64'),
            'float32': pd.Series([1.1, 2.2, 3.3], dtype='float32'),
            'float64': pd.Series([1.1, 2.2, 3.3], dtype='float64'),
            'bool': pd.Series([True, False, True]),
            'datetime': pd.Series(pd.date_range('2024-01-01', periods=3)),
            'string': pd.Series(['a', 'b', 'c'])
        }
        
        expected_types = {
            'int8': 'SMALLINT',
            'int64': 'BIGINT',
            'float32': 'FLOAT',
            'float64': 'DOUBLE',
            'bool': 'BOOLEAN',
            'datetime': 'TIMESTAMP',
            'string': 'TEXT'
        }
        
        for dtype_name, series in test_series.items():
            sql_type = self.manager._map_to_sql_type(str(series.dtype), series)
            
            # For string types, might be VARCHAR or TEXT
            if dtype_name == 'string':
                self.assertIn(sql_type, ['TEXT', 'VARCHAR(255)'])
            else:
                self.assertEqual(sql_type, expected_types[dtype_name],
                               f"Failed for {dtype_name}: expected {expected_types[dtype_name]}, got {sql_type}")
    
    def test_index_determination(self):
        """Test index determination logic"""
        schema = self.manager.detect_schema_from_csv(self.temp_csv.name)
        
        # Should have indexes on date, zone, and index columns
        self.assertIn('date', schema.indexes)
        self.assertIn('index', schema.indexes)
        
        # Should not index high-cardinality columns
        for idx in schema.indexes:
            self.assertNotIn('strategy', idx)
    
    def test_create_table_sql_generation(self):
        """Test SQL generation"""
        schema = self.manager.detect_schema_from_csv(self.temp_csv.name)
        
        # Generate CREATE TABLE SQL
        create_sql = self.manager.generate_create_table_sql(schema)
        
        self.assertIn('CREATE TABLE IF NOT EXISTS', create_sql)
        self.assertIn('date timestamp', create_sql.lower())
        self.assertIn('strategy_1 double', create_sql.lower())
        self.assertIn('zone text', create_sql.lower())
        
        # Generate index SQL
        index_sqls = self.manager.generate_index_sql(schema)
        self.assertTrue(len(index_sqls) > 0)
        
        for idx_sql in index_sqls:
            self.assertIn('CREATE INDEX IF NOT EXISTS', idx_sql)
    
    def test_wide_table_detection(self):
        """Test wide table detection and optimization"""
        # Create wide table with many columns
        wide_data = {'Date': pd.date_range('2024-01-01', periods=10)}
        
        # Add 2000 strategy columns
        for i in range(2000):
            wide_data[f'Strategy_{i+1}'] = np.random.randn(10)
        
        wide_df = pd.DataFrame(wide_data)
        
        # Save to CSV
        wide_csv = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        wide_df.to_csv(wide_csv.name, index=False)
        wide_csv.close()
        
        try:
            # Detect schema
            schema = self.manager.detect_schema_from_csv(wide_csv.name)
            
            self.assertTrue(schema.metadata['is_wide_table'])
            self.assertEqual(schema.strategy_column_count, 2000)
            
            # Test optimization
            optimized = self.manager.optimize_for_wide_table(schema)
            
            # Should have fewer indexes
            self.assertLess(len(optimized.indexes), 4)
            
            # Check SQL generation includes optimization
            create_sql = self.manager.generate_create_table_sql(optimized)
            self.assertIn('WITH', create_sql)
            
        finally:
            os.unlink(wide_csv.name)
    
    def test_schema_validation(self):
        """Test schema validation"""
        schema = self.manager.detect_schema_from_csv(self.temp_csv.name)
        
        # Validate with matching data
        is_valid, issues = self.manager.validate_schema_implementation(schema, self.test_data)
        self.assertTrue(is_valid)
        self.assertEqual(len(issues), 0)
        
        # Test with missing column
        incomplete_data = self.test_data.drop(columns=['Zone'])
        is_valid, issues = self.manager.validate_schema_implementation(schema, incomplete_data)
        self.assertFalse(is_valid)
        self.assertTrue(any('Missing columns' in issue for issue in issues))
        
        # Test with wrong data type
        wrong_type_data = self.test_data.copy()
        wrong_type_data['Integer_Values'] = ['text'] * len(wrong_type_data)
        is_valid, issues = self.manager.validate_schema_implementation(schema, wrong_type_data)
        self.assertFalse(is_valid)
        self.assertTrue(any('non-numeric' in issue for issue in issues))
    
    def test_special_column_detection(self):
        """Test special column pattern detection"""
        special_names = {
            'trade_date': 'date',
            'DateTime': 'date',
            'Region_1': 'zone',
            'Area_Code': 'zone',
            'row_index': 'index',
            'ID': 'index',
            'Strategy_100': 'strategy',
            'strategy500': 'strategy'
        }
        
        for col_name, expected_type in special_names.items():
            series = pd.Series([1, 2, 3])
            col_schema = self.manager._analyze_column(col_name, series)
            
            if expected_type == 'strategy':
                self.assertTrue(col_schema.metadata['is_strategy'])
            elif expected_type == 'index':
                self.assertTrue(col_schema.is_index)
            
            self.assertTrue(col_schema.has_special_handling)
    
    def test_metadata_columns(self):
        """Test metadata column creation"""
        metadata_cols = self.manager._create_metadata_columns()
        
        self.assertEqual(len(metadata_cols), 2)
        
        # Check load_timestamp
        timestamp_col = next(c for c in metadata_cols if c.name == 'load_timestamp')
        self.assertEqual(timestamp_col.sql_type, 'TIMESTAMP')
        self.assertFalse(timestamp_col.is_nullable)
        
        # Check row_hash
        hash_col = next(c for c in metadata_cols if c.name == 'row_hash')
        self.assertEqual(hash_col.sql_type, 'VARCHAR(64)')
        self.assertTrue(hash_col.is_nullable)


class TestTableSchemaIntegration(unittest.TestCase):
    """Test integration with TableSchema dataclass"""
    
    def test_schema_properties(self):
        """Test TableSchema properties"""
        columns = [
            ColumnSchema('col1', 'Col 1', 'int64', 'BIGINT'),
            ColumnSchema('strategy_1', 'Strategy 1', 'float64', 'DOUBLE',
                        metadata={'is_strategy': True}),
            ColumnSchema('strategy_2', 'Strategy 2', 'float64', 'DOUBLE',
                        metadata={'is_strategy': True}),
        ]
        
        schema = TableSchema(
            table_name='test_table',
            columns=columns,
            indexes=['col1'],
            partitions=[],
            metadata={}
        )
        
        self.assertEqual(schema.column_count, 3)
        self.assertEqual(schema.strategy_column_count, 2)


if __name__ == '__main__':
    unittest.main()