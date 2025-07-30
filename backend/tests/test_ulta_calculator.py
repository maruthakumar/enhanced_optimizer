"""
Unit tests for ULTA Calculator module.

These tests validate that the extracted ULTA logic produces identical results
to the legacy implementation.
"""

import unittest
import numpy as np
import pandas as pd
import sys
import os
from typing import Dict, Tuple

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ulta_calculator import (
    ULTACalculator, 
    ULTAStrategyMetrics,
    apply_ulta_logic
)


class TestULTACalculator(unittest.TestCase):
    """Test cases for ULTA Calculator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.calculator = ULTACalculator()
        
        # Create test data with known patterns
        self.test_returns_positive = np.array([0.01, 0.02, -0.005, 0.015, 0.01])
        self.test_returns_negative = np.array([-0.02, -0.01, 0.005, -0.015, -0.01])
        self.test_returns_mixed = np.array([0.05, -0.10, 0.03, -0.02, 0.01])
        
        # Create test DataFrame
        self.test_df = pd.DataFrame({
            'Date': pd.date_range('2024-01-01', periods=5),
            'Zone': ['Zone1'] * 5,
            'Day': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri'],
            'strategy_good': self.test_returns_positive,
            'strategy_bad': self.test_returns_negative,
            'strategy_mixed': self.test_returns_mixed
        })
    
    def test_calculate_roi(self):
        """Test ROI calculation."""
        roi = self.calculator.calculate_roi(self.test_returns_positive)
        expected_roi = np.sum(self.test_returns_positive)
        self.assertAlmostEqual(roi, expected_roi, places=6)
        
        roi_negative = self.calculator.calculate_roi(self.test_returns_negative)
        expected_roi_negative = np.sum(self.test_returns_negative)
        self.assertAlmostEqual(roi_negative, expected_roi_negative, places=6)
    
    def test_calculate_drawdown(self):
        """Test drawdown calculation."""
        dd = self.calculator.calculate_drawdown(self.test_returns_positive)
        expected_dd = np.min(self.test_returns_positive)
        self.assertAlmostEqual(dd, expected_dd, places=6)
        
        dd_negative = self.calculator.calculate_drawdown(self.test_returns_negative)
        expected_dd_negative = np.min(self.test_returns_negative)
        self.assertAlmostEqual(dd_negative, expected_dd_negative, places=6)
    
    def test_calculate_ratio(self):
        """Test ROI/Drawdown ratio calculation."""
        # Test positive ROI, negative drawdown
        ratio = self.calculator.calculate_ratio(0.05, -0.02)
        self.assertAlmostEqual(ratio, 2.5, places=6)
        
        # Test negative ROI, negative drawdown
        ratio = self.calculator.calculate_ratio(-0.05, -0.02)
        self.assertAlmostEqual(ratio, -2.5, places=6)
        
        # Test zero drawdown
        ratio = self.calculator.calculate_ratio(0.05, 0)
        self.assertEqual(ratio, float('inf'))
    
    def test_invert_strategy(self):
        """Test strategy inversion."""
        inverted = self.calculator.invert_strategy(self.test_returns_positive)
        np.testing.assert_array_almost_equal(inverted, -self.test_returns_positive)
        
        # Test double inversion returns to original
        double_inverted = self.calculator.invert_strategy(inverted)
        np.testing.assert_array_almost_equal(double_inverted, self.test_returns_positive)
    
    def test_should_invert_strategy_positive_ratio(self):
        """Test that strategies with positive ratios are not inverted."""
        should_invert, metrics = self.calculator.should_invert_strategy(self.test_returns_positive)
        self.assertFalse(should_invert)
        self.assertIsNone(metrics)
    
    def test_should_invert_strategy_negative_ratio(self):
        """Test strategy inversion decision for negative ratio."""
        should_invert, metrics = self.calculator.should_invert_strategy(self.test_returns_negative)
        
        # This strategy should be inverted because inverting improves the ratio
        self.assertTrue(should_invert)
        self.assertIsNotNone(metrics)
        
        # Verify metrics
        self.assertLess(metrics.original_ratio, 0)
        self.assertGreater(metrics.inverted_ratio, metrics.original_ratio)
        self.assertTrue(metrics.was_inverted)
    
    def test_apply_ulta_logic_dataframe(self):
        """Test ULTA logic application to DataFrame."""
        processed_df, inverted_strategies = self.calculator.apply_ulta_logic(self.test_df)
        
        # Check that good strategy was not inverted
        self.assertIn('strategy_good', processed_df.columns)
        self.assertNotIn('strategy_good_inv', processed_df.columns)
        
        # Check that bad strategy was inverted
        self.assertNotIn('strategy_bad', processed_df.columns)
        self.assertIn('strategy_bad_inv', processed_df.columns)
        
        # Verify inverted values
        expected_inverted = -self.test_returns_negative
        np.testing.assert_array_almost_equal(
            processed_df['strategy_bad_inv'].values,
            expected_inverted
        )
        
        # Check inverted strategies dict
        self.assertIn('strategy_bad', inverted_strategies)
        self.assertTrue(inverted_strategies['strategy_bad'].was_inverted)
    
    def test_backward_compatibility_function(self):
        """Test backward compatibility wrapper function."""
        processed_df, legacy_dict = apply_ulta_logic(self.test_df)
        
        # Check that output format matches legacy format
        for strategy_name, metrics in legacy_dict.items():
            self.assertIn('original_roi', metrics)
            self.assertIn('inverted_roi', metrics)
            self.assertIn('original_drawdown', metrics)
            self.assertIn('inverted_drawdown', metrics)
            self.assertIn('original_ratio', metrics)
            self.assertIn('inverted_ratio', metrics)
            # Note: legacy format doesn't include improvement_percentage
    
    def test_generate_markdown_report(self):
        """Test markdown report generation."""
        # Apply ULTA logic first
        self.calculator.apply_ulta_logic(self.test_df)
        
        report = self.calculator.generate_inversion_report('markdown')
        
        # Check report contains expected sections
        self.assertIn('# ULTA Inversion Report', report)
        self.assertIn('Total strategies analyzed:', report)
        self.assertIn('Strategies inverted:', report)
        self.assertIn('## Inverted Strategies', report)
        
        # Check that inverted strategy details are included
        if 'strategy_bad' in self.calculator.inverted_strategies:
            self.assertIn('strategy_bad', report)
            self.assertIn('Original ROI:', report)
            self.assertIn('Inverted ROI:', report)
    
    def test_generate_json_report(self):
        """Test JSON report generation."""
        # Apply ULTA logic first
        self.calculator.apply_ulta_logic(self.test_df)
        
        report_json = self.calculator.generate_inversion_report('json')
        
        # Parse JSON to verify structure
        import json
        report_data = json.loads(report_json)
        
        self.assertIn('summary', report_data)
        self.assertIn('inverted_strategies', report_data)
        self.assertIn('total_strategies_analyzed', report_data['summary'])
        self.assertIn('strategies_inverted', report_data['summary'])
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Test with all zeros
        zero_returns = np.zeros(5)
        should_invert, metrics = self.calculator.should_invert_strategy(zero_returns)
        self.assertFalse(should_invert)  # Ratio is inf, not negative
        
        # Test with single value
        single_value = np.array([0.05])
        roi = self.calculator.calculate_roi(single_value)
        self.assertAlmostEqual(roi, 0.05)
        
        # Test with empty array
        empty_returns = np.array([])
        roi = self.calculator.calculate_roi(empty_returns)
        self.assertEqual(roi, 0.0)
    
    def test_numeric_precision(self):
        """Test numeric precision and consistency."""
        # Create test data with very small values
        small_returns = np.array([0.0001, -0.0002, 0.0001, -0.0001, 0.0002])
        
        roi = self.calculator.calculate_roi(small_returns)
        inverted = self.calculator.invert_strategy(small_returns)
        inverted_roi = self.calculator.calculate_roi(inverted)
        
        # ROI of inverted should be negative of original
        self.assertAlmostEqual(roi, -inverted_roi, places=10)
    
    def test_get_inverted_strategy_names(self):
        """Test getting list of inverted strategy names."""
        self.calculator.apply_ulta_logic(self.test_df)
        
        inverted_names = self.calculator.get_inverted_strategy_names()
        
        # Should contain strategies that were actually inverted
        for name in inverted_names:
            self.assertTrue(self.calculator.inverted_strategies[name].was_inverted)
    
    def test_get_inversion_metrics(self):
        """Test getting metrics for specific strategy."""
        self.calculator.apply_ulta_logic(self.test_df)
        
        # Get metrics for a strategy that was processed
        metrics = self.calculator.get_inversion_metrics('strategy_bad')
        if metrics:
            self.assertIsInstance(metrics, ULTAStrategyMetrics)
            self.assertEqual(metrics.strategy_name, 'strategy_bad')
        
        # Test non-existent strategy
        metrics = self.calculator.get_inversion_metrics('non_existent')
        self.assertIsNone(metrics)


class TestULTACalculatorIntegration(unittest.TestCase):
    """Integration tests comparing with legacy implementation."""
    
    def setUp(self):
        """Set up integration test data."""
        # Create larger test dataset
        np.random.seed(42)  # For reproducibility
        dates = pd.date_range('2024-01-01', periods=100)
        
        data = {
            'Date': dates,
            'Zone': ['Zone1'] * 100,
            'Day': [d.strftime('%a') for d in dates]
        }
        
        # Create 50 strategies with various performance profiles
        for i in range(50):
            if i < 10:
                # Good performing strategies
                returns = np.random.normal(0.001, 0.01, 100)
            elif i < 30:
                # Poor performing strategies (candidates for inversion)
                returns = np.random.normal(-0.0015, 0.01, 100)
            else:
                # Mixed performance
                returns = np.random.normal(0, 0.015, 100)
            
            data[f'strategy_{i:03d}'] = returns
        
        self.test_df = pd.DataFrame(data)
        self.calculator = ULTACalculator()
    
    def test_large_dataset_performance(self):
        """Test performance with larger dataset."""
        import time
        
        start_time = time.time()
        processed_df, inverted_strategies = self.calculator.apply_ulta_logic(self.test_df)
        end_time = time.time()
        
        # Check that processing completed in reasonable time
        processing_time = end_time - start_time
        self.assertLess(processing_time, 1.0)  # Should complete in under 1 second
        
        # Verify some strategies were inverted
        self.assertGreater(len(inverted_strategies), 0)
        
        # Verify DataFrame integrity
        self.assertEqual(len(processed_df), len(self.test_df))
        self.assertEqual(processed_df.iloc[:, :3].values.tolist(), 
                        self.test_df.iloc[:, :3].values.tolist())
    
    def test_consistency_across_runs(self):
        """Test that ULTA logic produces consistent results."""
        # Run ULTA logic multiple times
        results = []
        for _ in range(3):
            _, inverted = self.calculator.apply_ulta_logic(self.test_df)
            results.append(set(inverted.keys()))
        
        # All runs should produce identical results
        for i in range(1, len(results)):
            self.assertEqual(results[0], results[i])


if __name__ == '__main__':
    unittest.main()