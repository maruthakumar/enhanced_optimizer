#!/usr/bin/env python3
"""
Example usage of Dynamic Table Manager

Demonstrates automatic schema detection, wide table handling,
and integration with the DAL.
"""

import sys
import os
import logging
import pandas as pd
import numpy as np
import tempfile

# Add backend to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dal import get_dal
from dal.dynamic_table_manager import DynamicTableManager

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def demonstrate_schema_detection():
    """Demonstrate automatic schema detection"""
    logger.info("\n" + "="*60)
    logger.info("Demonstrating Schema Detection")
    logger.info("="*60)
    
    # Create sample data with mixed types
    sample_data = pd.DataFrame({
        'Date': pd.date_range('2024-01-01', periods=100),
        'Zone': ['Zone_1', 'Zone_2', 'Zone_3', 'Zone_4'] * 25,
        'Day': range(100),
        'Strategy_1': np.random.randn(100).cumsum() + 1000,
        'Strategy_2': np.random.randn(100).cumsum() + 1000,
        'Strategy_3': np.random.randn(100).cumsum() + 1000,
        'Strategy_4': np.random.randn(100).cumsum() + 1000,
        'Strategy_5': np.random.randn(100).cumsum() + 1000,
        'Text_Field': ['Type_' + str(i % 3) for i in range(100)],
        'Boolean_Flag': [True, False] * 50,
        'Integer_Count': np.random.randint(1, 100, 100),
        'Nullable_Value': [None if i % 20 == 0 else float(i) for i in range(100)]
    })
    
    # Save to CSV
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    sample_data.to_csv(temp_file.name, index=False)
    temp_file.close()
    
    try:
        # Create table manager
        manager = DynamicTableManager()
        
        # Detect schema
        logger.info(f"\nDetecting schema from: {temp_file.name}")
        schema = manager.detect_schema_from_csv(temp_file.name)
        
        logger.info(f"\nDetected Schema:")
        logger.info(f"  Table name: {schema.table_name}")
        logger.info(f"  Total columns: {schema.column_count}")
        logger.info(f"  Strategy columns: {schema.strategy_column_count}")
        logger.info(f"  Indexes: {', '.join(schema.indexes)}")
        logger.info(f"  Wide table: {schema.metadata.get('is_wide_table', False)}")
        
        # Show column details
        logger.info("\nColumn Details:")
        for col in schema.columns[:5]:  # Show first 5
            logger.info(f"  {col.name}: {col.sql_type} (nullable: {col.is_nullable})")
        
        # Generate SQL
        create_sql = manager.generate_create_table_sql(schema)
        logger.info(f"\nGenerated CREATE TABLE SQL:")
        logger.info(create_sql[:500] + "..." if len(create_sql) > 500 else create_sql)
        
    finally:
        os.unlink(temp_file.name)


def demonstrate_wide_table_handling():
    """Demonstrate handling of wide tables with many columns"""
    logger.info("\n" + "="*60)
    logger.info("Demonstrating Wide Table Handling")
    logger.info("="*60)
    
    # Create wide table similar to production data
    n_strategies = 25546  # Production-like column count
    n_rows = 82  # Production-like row count (days)
    
    logger.info(f"\nCreating wide table with {n_strategies} strategy columns...")
    
    # Build data dictionary
    data = {
        'Date': pd.date_range('2024-01-01', periods=n_rows),
        'Zone': ['Zone_' + str(i % 4 + 1) for i in range(n_rows)],
        'Day': range(n_rows)
    }
    
    # Add strategy columns (limited for demo)
    demo_strategies = min(n_strategies, 100)  # Limit for demo
    for i in range(demo_strategies):
        data[f'Strategy_{i+1}'] = np.random.randn(n_rows).cumsum() + 1000
    
    wide_df = pd.DataFrame(data)
    
    # Save to CSV
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    wide_df.to_csv(temp_file.name, index=False)
    temp_file.close()
    
    try:
        # Create table manager
        manager = DynamicTableManager()
        
        # Detect schema
        logger.info(f"\nDetecting schema for wide table...")
        schema = manager.detect_schema_from_csv(temp_file.name)
        
        logger.info(f"\nWide Table Schema:")
        logger.info(f"  Detected as wide table: {schema.metadata.get('is_wide_table', False)}")
        logger.info(f"  Total columns: {schema.column_count}")
        logger.info(f"  Strategy columns: {schema.strategy_column_count}")
        
        # Apply optimizations
        logger.info("\nApplying wide table optimizations...")
        optimized_schema = manager.optimize_for_wide_table(schema)
        
        logger.info(f"  Original indexes: {len(schema.indexes)}")
        logger.info(f"  Optimized indexes: {len(optimized_schema.indexes)}")
        logger.info(f"  Optimizations applied: {optimized_schema.metadata.get('optimizations_applied', [])}")
        
        # Show create table SQL with optimizations
        create_sql = manager.generate_create_table_sql(optimized_schema)
        if 'WITH' in create_sql:
            logger.info("\n  Table options applied for wide table optimization")
        
    finally:
        os.unlink(temp_file.name)


def demonstrate_dal_integration():
    """Demonstrate integration with DAL"""
    logger.info("\n" + "="*60)
    logger.info("Demonstrating DAL Integration")
    logger.info("="*60)
    
    # Create production-like data
    data = {
        'Date': pd.date_range('2024-01-01', periods=100),
        'Zone': ['Zone_1', 'Zone_2', 'Zone_3', 'Zone_4'] * 25,
        'Day': range(100)
    }
    
    # Add 50 strategy columns
    for i in range(50):
        data[f'Strategy_{i+1}'] = np.random.randn(100).cumsum() + 1000
    
    df = pd.DataFrame(data)
    
    # Save to CSV
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    df.to_csv(temp_file.name, index=False)
    temp_file.close()
    
    try:
        # Use DAL with dynamic table creation
        with get_dal('csv') as dal:
            logger.info("\nLoading CSV with automatic schema detection...")
            
            # Load with dynamic schema
            success = dal.load_csv_to_heavydb(temp_file.name, 'production_strategies')
            logger.info(f"Load success: {success}")
            
            if success:
                # Check loaded data
                row_count = dal.get_table_row_count('production_strategies')
                logger.info(f"Rows loaded: {row_count}")
                
                # Get schema info (for CSV DAL, stored as metadata)
                if hasattr(dal, 'tables') and '_schema_production_strategies' in dal.tables:
                    schema_info = dal.tables['_schema_production_strategies'].iloc[0]
                    logger.info(f"Strategy columns: {schema_info['strategy_count']}")
                    logger.info(f"Wide table: {schema_info['is_wide_table']}")
                    logger.info(f"Indexes: {schema_info['indexes']}")
        
    finally:
        os.unlink(temp_file.name)


def demonstrate_validation():
    """Demonstrate schema validation"""
    logger.info("\n" + "="*60)
    logger.info("Demonstrating Schema Validation")
    logger.info("="*60)
    
    # Create data
    data = pd.DataFrame({
        'Date': pd.date_range('2024-01-01', periods=10),
        'Strategy_1': np.random.randn(10),
        'Strategy_2': np.random.randn(10),
        'Invalid_Column': ['text'] * 10  # Will cause validation issue
    })
    
    # Save to CSV
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    data.to_csv(temp_file.name, index=False)
    temp_file.close()
    
    try:
        manager = DynamicTableManager()
        
        # Detect schema
        schema = manager.detect_schema_from_csv(temp_file.name)
        
        # Modify data to cause validation issues
        invalid_data = data.copy()
        invalid_data['Strategy_1'] = ['not_a_number'] * 10
        
        # Validate
        is_valid, issues = manager.validate_schema_implementation(schema, invalid_data)
        
        logger.info(f"\nValidation result: {'PASS' if is_valid else 'FAIL'}")
        if issues:
            logger.info("Issues found:")
            for issue in issues:
                logger.info(f"  - {issue}")
        
    finally:
        os.unlink(temp_file.name)


if __name__ == '__main__':
    # Run all demonstrations
    demonstrate_schema_detection()
    demonstrate_wide_table_handling()
    demonstrate_dal_integration()
    demonstrate_validation()
    
    logger.info("\n" + "="*60)
    logger.info("All demonstrations completed!")
    logger.info("="*60)