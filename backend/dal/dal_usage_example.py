#!/usr/bin/env python3
"""
Example usage of the Data Access Layer (DAL)

This demonstrates how to use the DAL in the Heavy Optimizer Platform
for both HeavyDB and CSV modes.
"""

import sys
import os
import logging
import numpy as np
import pandas as pd

# Add backend to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dal import get_dal

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def demonstrate_dal_usage():
    """Demonstrate basic DAL usage"""
    
    # Create sample data
    logger.info("Creating sample data...")
    sample_data = pd.DataFrame({
        'Date': pd.date_range('2024-01-01', periods=100),
        'Time': pd.date_range('09:30:00', periods=100, freq='1min').time,
        'Index': range(100),
        'Strategy_1': np.random.randn(100).cumsum() + 100,
        'Strategy_2': np.random.randn(100).cumsum() + 100,
        'Strategy_3': np.random.randn(100).cumsum() + 100,
        'Strategy_4': np.random.randn(100).cumsum() + 100,
        'Strategy_5': np.random.randn(100).cumsum() + 100,
    })
    
    # Save sample data
    sample_file = '/tmp/sample_strategies.csv'
    sample_data.to_csv(sample_file, index=False)
    logger.info(f"Sample data saved to {sample_file}")
    
    # Get DAL instance (auto-selects based on availability)
    logger.info("\n1. Creating DAL instance...")
    with get_dal() as dal:
        logger.info(f"DAL type: {type(dal).__name__}")
        logger.info(f"GPU support: {dal.supports_gpu}")
        
        # Load CSV data
        logger.info("\n2. Loading CSV data...")
        success = dal.load_csv_to_heavydb(sample_file, 'strategies')
        logger.info(f"Load success: {success}")
        
        if success:
            # Get table info
            row_count = dal.get_table_row_count('strategies')
            logger.info(f"Rows loaded: {row_count}")
            
            schema = dal.get_table_schema('strategies')
            logger.info(f"Table schema: {list(schema.keys())[:5]}...")
            
            # Apply ULTA transformation
            logger.info("\n3. Applying ULTA transformation...")
            ulta_success = dal.apply_ulta_transformation('strategies')
            logger.info(f"ULTA transformation success: {ulta_success}")
            
            # Compute correlation matrix
            logger.info("\n4. Computing correlation matrix...")
            corr_matrix = dal.compute_correlation_matrix('strategies')
            if corr_matrix is not None:
                logger.info(f"Correlation matrix shape: {corr_matrix.shape}")
                logger.info(f"Sample correlations: {corr_matrix[0, 1]:.3f}, {corr_matrix[0, 2]:.3f}")
            
            # Get strategy subset
            logger.info("\n5. Getting strategy subset...")
            subset = dal.get_strategy_subset('strategies', [3, 4, 5])
            if subset is not None:
                logger.info(f"Subset shape: {subset.shape}")
                logger.info(f"Subset columns: {list(subset.columns)}")
            
            # Execute custom query (CSV DAL has limited SQL support)
            logger.info("\n6. Executing custom query...")
            if dal.supports_gpu:
                # HeavyDB supports full SQL
                result = dal.execute_gpu_query("SELECT COUNT(*) FROM strategies")
                logger.info(f"Query result: {result}")
            else:
                # CSV DAL has limited SQL support
                logger.info("CSV DAL has limited SQL support")
            
            # Create dynamic table
            logger.info("\n7. Creating dynamic table...")
            new_data = pd.DataFrame({
                'metric': ['ROI', 'Sharpe', 'MaxDD'],
                'value': [0.15, 1.2, -0.08]
            })
            dynamic_success = dal.create_dynamic_table('metrics', new_data)
            logger.info(f"Dynamic table creation success: {dynamic_success}")
            
            # Clean up
            logger.info("\n8. Cleaning up...")
            dal.drop_table('strategies')
            dal.drop_table('strategies_ulta')
            dal.drop_table('metrics')
            logger.info("Tables dropped")
    
    # Clean up sample file
    os.unlink(sample_file)
    logger.info("\nDemo completed successfully!")


def demonstrate_fallback_behavior():
    """Demonstrate DAL fallback from HeavyDB to CSV"""
    
    logger.info("\n" + "="*60)
    logger.info("Demonstrating DAL fallback behavior")
    logger.info("="*60)
    
    # Try to create HeavyDB DAL explicitly
    logger.info("\nAttempting to create HeavyDB DAL...")
    try:
        heavydb_dal = get_dal('heavydb')
        if heavydb_dal.connect():
            logger.info("HeavyDB connection successful!")
            heavydb_dal.disconnect()
        else:
            logger.info("HeavyDB connection failed, would use CSV fallback")
    except Exception as e:
        logger.info(f"HeavyDB not available: {e}")
    
    # Force CSV DAL
    logger.info("\nForcing CSV DAL...")
    csv_dal = get_dal('csv')
    csv_dal.connect()
    logger.info(f"CSV DAL connected: {csv_dal.is_connected}")
    logger.info(f"GPU support: {csv_dal.supports_gpu}")
    csv_dal.disconnect()


def demonstrate_advanced_features():
    """Demonstrate advanced DAL features"""
    
    logger.info("\n" + "="*60)
    logger.info("Demonstrating advanced DAL features")
    logger.info("="*60)
    
    with get_dal('csv') as dal:  # Force CSV for demo
        # Create larger dataset
        logger.info("\nCreating larger dataset...")
        n_strategies = 50
        n_days = 250
        
        large_data = pd.DataFrame({
            'Date': pd.date_range('2024-01-01', periods=n_days),
            'Time': '09:30:00',
            'Index': range(n_days)
        })
        
        # Add strategy columns
        for i in range(n_strategies):
            large_data[f'Strategy_{i+1}'] = np.random.randn(n_days).cumsum() + 100
        
        # Load data
        dal.create_dynamic_table('large_strategies', large_data)
        logger.info(f"Created table with {n_strategies} strategies and {n_days} days")
        
        # Test correlation on larger dataset
        logger.info("\nComputing correlation on larger dataset...")
        corr_matrix = dal.compute_correlation_matrix('large_strategies')
        if corr_matrix is not None:
            logger.info(f"Large correlation matrix shape: {corr_matrix.shape}")
            
            # Find highly correlated pairs
            high_corr_pairs = []
            for i in range(n_strategies):
                for j in range(i+1, n_strategies):
                    if abs(corr_matrix[i, j]) > 0.7:
                        high_corr_pairs.append((i, j, corr_matrix[i, j]))
            
            logger.info(f"Found {len(high_corr_pairs)} highly correlated pairs (|r| > 0.7)")
        
        # Test CSV-specific features
        if hasattr(dal, 'save_table_to_csv'):
            logger.info("\nTesting CSV-specific features...")
            output_file = '/tmp/exported_strategies.csv'
            dal.save_table_to_csv('large_strategies', output_file)
            logger.info(f"Exported table to {output_file}")
            
            # Verify export
            if os.path.exists(output_file):
                exported_df = pd.read_csv(output_file)
                logger.info(f"Exported file has {len(exported_df)} rows and {len(exported_df.columns)} columns")
                os.unlink(output_file)


if __name__ == '__main__':
    # Run demonstrations
    demonstrate_dal_usage()
    demonstrate_fallback_behavior()
    demonstrate_advanced_features()
    
    logger.info("\n" + "="*60)
    logger.info("All demonstrations completed!")
    logger.info("="*60)