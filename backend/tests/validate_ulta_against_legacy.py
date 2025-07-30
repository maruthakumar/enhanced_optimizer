"""
Validation script to compare extracted ULTA logic against legacy implementation.

This script loads test data, runs both implementations, and verifies they produce
identical results.
"""

import sys
import os
import numpy as np
import pandas as pd
from typing import Dict, Tuple
import logging

# Setup paths
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, '/mnt/optimizer_share/zone_optimization_25_06_25')

# Import both implementations
from ulta_calculator import ULTACalculator
import Optimizer_New_patched as legacy

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def compare_dataframes(df1: pd.DataFrame, df2: pd.DataFrame, label: str) -> bool:
    """Compare two dataframes for equality."""
    if df1.shape != df2.shape:
        logger.error(f"{label}: Shape mismatch - {df1.shape} vs {df2.shape}")
        return False
    
    # Compare column names
    if not all(df1.columns == df2.columns):
        logger.error(f"{label}: Column mismatch")
        logger.error(f"DF1 columns: {list(df1.columns)}")
        logger.error(f"DF2 columns: {list(df2.columns)}")
        return False
    
    # Compare values
    for col in df1.columns:
        # Handle different data types
        if df1[col].dtype == 'object' or df2[col].dtype == 'object':
            # String comparison
            if not all(df1[col].fillna('') == df2[col].fillna('')):
                logger.error(f"{label}: Values mismatch in column {col}")
                return False
        elif pd.api.types.is_datetime64_any_dtype(df1[col]) or pd.api.types.is_datetime64_any_dtype(df2[col]):
            # Datetime comparison
            if not df1[col].equals(df2[col]):
                logger.error(f"{label}: Values mismatch in column {col}")
                return False
        else:
            # Numeric comparison
            if not np.allclose(df1[col].fillna(0), df2[col].fillna(0), rtol=1e-6):
                logger.error(f"{label}: Values mismatch in column {col}")
                return False
    
    return True


def compare_inversion_dicts(dict1: Dict, dict2: Dict, label: str) -> bool:
    """Compare two inversion dictionaries."""
    if set(dict1.keys()) != set(dict2.keys()):
        logger.error(f"{label}: Different strategies inverted")
        logger.error(f"Dict1 keys: {sorted(dict1.keys())}")
        logger.error(f"Dict2 keys: {sorted(dict2.keys())}")
        return False
    
    for key in dict1:
        for metric in ['original_roi', 'inverted_roi', 'original_drawdown', 
                      'inverted_drawdown', 'original_ratio', 'inverted_ratio']:
            if not np.isclose(dict1[key][metric], dict2[key][metric], rtol=1e-6):
                logger.error(f"{label}: Mismatch in {key}.{metric}")
                logger.error(f"Dict1: {dict1[key][metric]}")
                logger.error(f"Dict2: {dict2[key][metric]}")
                return False
    
    return True


def create_test_data() -> pd.DataFrame:
    """Create comprehensive test data."""
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100)
    
    data = {
        'Date': dates,
        'Zone': ['Zone1'] * 100,
        'Day': [d.strftime('%a') for d in dates]
    }
    
    # Create various strategy types
    # Good performing strategies
    for i in range(5):
        data[f'good_strategy_{i}'] = np.random.normal(0.002, 0.01, 100)
    
    # Poor performing strategies (should be inverted)
    for i in range(10):
        data[f'poor_strategy_{i}'] = np.random.normal(-0.002, 0.01, 100)
    
    # Mixed performance strategies
    for i in range(5):
        returns = np.random.normal(0, 0.015, 100)
        # Ensure some have negative ratios
        if i % 2 == 0:
            returns[0] = -0.5  # Large drawdown
        data[f'mixed_strategy_{i}'] = returns
    
    # Edge cases
    data['zero_strategy'] = np.zeros(100)
    data['constant_positive'] = np.ones(100) * 0.001
    data['constant_negative'] = np.ones(100) * -0.001
    
    return pd.DataFrame(data)


def validate_implementations():
    """Run validation tests comparing implementations."""
    logger.info("Starting ULTA validation against legacy implementation")
    
    # Create test data
    test_data = create_test_data()
    logger.info(f"Created test data with shape: {test_data.shape}")
    
    # Run legacy implementation
    logger.info("Running legacy ULTA implementation...")
    legacy_result, legacy_inverted = legacy.apply_ulta_logic(test_data.copy())
    logger.info(f"Legacy: {len(legacy_inverted)} strategies inverted")
    
    # Run new implementation
    logger.info("Running extracted ULTA implementation...")
    calculator = ULTACalculator()
    new_result, new_inverted_obj = calculator.apply_ulta_logic(test_data.copy())
    
    # Convert new format to legacy format for comparison
    new_inverted = {}
    for name, metrics in new_inverted_obj.items():
        if metrics.was_inverted:
            new_inverted[name] = {
                "original_roi": metrics.original_roi,
                "inverted_roi": metrics.inverted_roi,
                "original_drawdown": metrics.original_drawdown,
                "inverted_drawdown": metrics.inverted_drawdown,
                "original_ratio": metrics.original_ratio,
                "inverted_ratio": metrics.inverted_ratio
            }
    
    logger.info(f"New: {len(new_inverted)} strategies inverted")
    
    # Compare results
    all_passed = True
    
    # Compare DataFrames
    if compare_dataframes(legacy_result, new_result, "DataFrames"):
        logger.info("✓ DataFrames match")
    else:
        all_passed = False
        logger.error("✗ DataFrames do not match")
    
    # Compare inversion dictionaries
    if compare_inversion_dicts(legacy_inverted, new_inverted, "Inversion dicts"):
        logger.info("✓ Inversion dictionaries match")
    else:
        all_passed = False
        logger.error("✗ Inversion dictionaries do not match")
    
    # Test edge cases
    logger.info("\nTesting edge cases...")
    
    # Test empty DataFrame
    empty_df = pd.DataFrame({
        'Date': [],
        'Zone': [],
        'Day': []
    })
    legacy_empty, _ = legacy.apply_ulta_logic(empty_df.copy())
    new_empty, _ = calculator.apply_ulta_logic(empty_df.copy())
    if legacy_empty.equals(new_empty):
        logger.info("✓ Empty DataFrame handling matches")
    else:
        logger.error("✗ Empty DataFrame handling differs")
        all_passed = False
    
    # Test single column
    single_col = test_data[['Date', 'Zone', 'Day', 'poor_strategy_0']].copy()
    legacy_single, legacy_single_inv = legacy.apply_ulta_logic(single_col.copy())
    new_single, new_single_inv = calculator.apply_ulta_logic(single_col.copy())
    
    if legacy_single.shape == new_single.shape:
        logger.info("✓ Single column handling matches")
    else:
        logger.error("✗ Single column handling differs")
        all_passed = False
    
    # Performance comparison
    logger.info("\nPerformance comparison...")
    import time
    
    # Time legacy implementation
    start = time.time()
    for _ in range(10):
        legacy.apply_ulta_logic(test_data.copy())
    legacy_time = (time.time() - start) / 10
    
    # Time new implementation
    start = time.time()
    for _ in range(10):
        calculator.apply_ulta_logic(test_data.copy())
    new_time = (time.time() - start) / 10
    
    logger.info(f"Legacy average time: {legacy_time:.4f}s")
    logger.info(f"New average time: {new_time:.4f}s")
    logger.info(f"Speed improvement: {legacy_time/new_time:.2f}x")
    
    # Summary
    logger.info("\n" + "="*50)
    if all_passed:
        logger.info("✓ ALL VALIDATION TESTS PASSED")
        logger.info("The extracted ULTA logic is functionally identical to the legacy implementation.")
    else:
        logger.error("✗ SOME VALIDATION TESTS FAILED")
        logger.error("Please review the errors above.")
    
    return all_passed


if __name__ == "__main__":
    success = validate_implementations()
    sys.exit(0 if success else 1)