"""
Unit tests for Zone Optimizer module
"""

import unittest
import os
import sys
import tempfile
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zone_optimizer import (
    ZoneOptimizer, ZoneConfiguration, ZoneResult, 
    CombinedZoneResult
)
from zone_optimizer_dal import ZoneOptimizerDAL


class TestZoneConfiguration(unittest.TestCase):
    """Test ZoneConfiguration dataclass"""
    
    def test_default_configuration(self):
        """Test default zone configuration"""
        config = ZoneConfiguration()
        
        self.assertTrue(config.enabled)
        self.assertEqual(len(config.zones), 4)
        self.assertEqual(config.zones, ["zone1", "zone2", "zone3", "zone4"])
        self.assertEqual(config.min_portfolio_size, 10)
        self.assertEqual(config.max_portfolio_size, 25)
        self.assertEqual(config.combination_method, "weighted_average")
        
    def test_zone_weights_initialization(self):
        """Test automatic zone weight initialization"""
        config = ZoneConfiguration()
        
        # Should have equal weights
        self.assertEqual(len(config.zone_weights), 4)
        for zone in config.zones:
            self.assertAlmostEqual(config.zone_weights[zone], 0.25)
        
    def test_custom_configuration(self):
        """Test custom configuration"""
        config = ZoneConfiguration(
            zones=["A", "B", "C"],
            min_portfolio_size=5,
            combination_method="best_zone"
        )
        
        self.assertEqual(len(config.zones), 3)
        self.assertEqual(config.min_portfolio_size, 5)
        self.assertEqual(config.combination_method, "best_zone")
        
        # Check weights for custom zones
        for zone in config.zones:
            self.assertAlmostEqual(config.zone_weights[zone], 1/3)


class TestZoneOptimizer(unittest.TestCase):
    """Test ZoneOptimizer functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.optimizer = ZoneOptimizer()
        
        # Create test data with zones
        np.random.seed(42)
        n_rows = 100
        n_strategies = 20
        
        data = {
            'Date': pd.date_range('2024-01-01', periods=n_rows),
            'Time': '09:30:00',
            'Zone': ['zone1', 'zone2', 'zone3', 'zone4'] * 25
        }
        
        # Add strategy columns
        for i in range(n_strategies):
            data[f'Strategy_{i+1}'] = np.random.randn(n_rows).cumsum() + 100
        
        self.test_data = pd.DataFrame(data)
        
    def test_extract_zones_from_data(self):
        """Test zone extraction"""
        zones = self.optimizer.extract_zones_from_data(self.test_data)
        
        self.assertEqual(len(zones), 4)
        self.assertIn('zone1', zones)
        self.assertIn('zone2', zones)
        
        # Check each zone has data
        for zone_name, zone_df in zones.items():
            self.assertEqual(len(zone_df), 25)
            self.assertIn('Strategy_1', zone_df.columns)
            
    def test_extract_zones_missing_column(self):
        """Test extraction with missing Zone column"""
        bad_data = self.test_data.drop(columns=['Zone'])
        
        with self.assertRaises(ValueError) as context:
            self.optimizer.extract_zones_from_data(bad_data)
            
        self.assertIn("Zone", str(context.exception))
        
    def test_configuration_loading(self):
        """Test configuration loading from file"""
        # Create temporary config file
        config_content = """
[ZONE_SPECIFIC_OPTIMIZATION]
enable = True
min_size = 15
max_size = 30
population_size = 50
mutation_rate = 0.2

[OPTIMIZATION]
apply_ulta_logic = False
zone_weights = 0.3,0.3,0.2,0.2
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
            f.write(config_content)
            config_file = f.name
        
        try:
            optimizer = ZoneOptimizer(config_path=config_file)
            
            self.assertEqual(optimizer.config.min_portfolio_size, 15)
            self.assertEqual(optimizer.config.max_portfolio_size, 30)
            self.assertEqual(optimizer.config.population_size, 50)
            self.assertAlmostEqual(optimizer.config.mutation_rate, 0.2)
            self.assertFalse(optimizer.config.apply_ulta)
            
            # Check zone weights
            self.assertAlmostEqual(optimizer.config.zone_weights['zone1'], 0.3)
            self.assertAlmostEqual(optimizer.config.zone_weights['zone2'], 0.3)
            self.assertAlmostEqual(optimizer.config.zone_weights['zone3'], 0.2)
            self.assertAlmostEqual(optimizer.config.zone_weights['zone4'], 0.2)
            
        finally:
            os.unlink(config_file)
    
    def test_optimize_single_zone(self):
        """Test single zone optimization"""
        
        # Extract zones
        zones = self.optimizer.extract_zones_from_data(self.test_data)
        zone_data = zones['zone1']
        
        # Optimize
        result = self.optimizer.optimize_single_zone(
            'zone1', zone_data, portfolio_size=3, algorithm='genetic'
        )
        
        self.assertIsInstance(result, ZoneResult)
        self.assertEqual(result.zone_name, 'zone1')
        self.assertEqual(len(result.portfolio_indices), 3)
        self.assertIsInstance(result.fitness_score, float)
        self.assertGreater(result.fitness_score, 0)
        self.assertEqual(result.algorithm_used, 'genetic')
        self.assertIn('roi', result.metrics)
        
    def test_combine_weighted_average(self):
        """Test weighted average combination"""
        # Create mock results
        zone_results = {
            'zone1': ZoneResult(
                zone_name='zone1',
                portfolio_indices=[0, 1, 2],
                portfolio_columns=['Strategy_1', 'Strategy_2', 'Strategy_3'],
                fitness_score=0.8,
                metrics={},
                optimization_time=1.0,
                algorithm_used='genetic'
            ),
            'zone2': ZoneResult(
                zone_name='zone2',
                portfolio_indices=[2, 3, 4],
                portfolio_columns=['Strategy_3', 'Strategy_4', 'Strategy_5'],
                fitness_score=0.9,
                metrics={},
                optimization_time=1.0,
                algorithm_used='genetic'
            )
        }
        
        # Set custom weights
        self.optimizer.config.zone_weights = {'zone1': 0.4, 'zone2': 0.6}
        
        combined = self.optimizer.combine_zone_results(zone_results, method='weighted_average')
        
        self.assertIsInstance(combined, CombinedZoneResult)
        self.assertEqual(combined.combination_method, 'weighted_average')
        
        # Check weighted fitness
        expected_fitness = 0.8 * 0.4 + 0.9 * 0.6
        self.assertAlmostEqual(combined.combined_fitness, expected_fitness)
        
        # Check contributions
        self.assertAlmostEqual(combined.zone_contributions['zone1'], 0.4)
        self.assertAlmostEqual(combined.zone_contributions['zone2'], 0.6)
        
    def test_combine_best_zone(self):
        """Test best zone combination"""
        zone_results = {
            'zone1': ZoneResult(
                zone_name='zone1',
                portfolio_indices=[0, 1, 2],
                portfolio_columns=['Strategy_1', 'Strategy_2', 'Strategy_3'],
                fitness_score=0.7,
                metrics={},
                optimization_time=1.0,
                algorithm_used='genetic'
            ),
            'zone2': ZoneResult(
                zone_name='zone2',
                portfolio_indices=[3, 4, 5],
                portfolio_columns=['Strategy_4', 'Strategy_5', 'Strategy_6'],
                fitness_score=0.95,  # Best fitness
                metrics={},
                optimization_time=1.0,
                algorithm_used='genetic'
            )
        }
        
        combined = self.optimizer.combine_zone_results(zone_results, method='best_zone')
        
        self.assertEqual(combined.combination_method, 'best_zone')
        self.assertEqual(combined.combined_fitness, 0.95)
        self.assertEqual(combined.combined_portfolio, [3, 4, 5])
        self.assertEqual(combined.zone_contributions['zone2'], 1.0)
        self.assertEqual(combined.zone_contributions['zone1'], 0.0)
        self.assertEqual(combined.metadata['selected_zone'], 'zone2')
        
    def test_zone_correlation_analysis(self):
        """Test zone correlation analysis"""
        zone_results = {
            'zone1': ZoneResult(
                zone_name='zone1',
                portfolio_indices=[0, 1, 2],
                portfolio_columns=['Strategy_1', 'Strategy_2', 'Strategy_3'],
                fitness_score=0.8,
                metrics={},
                optimization_time=1.0,
                algorithm_used='genetic'
            ),
            'zone2': ZoneResult(
                zone_name='zone2',
                portfolio_indices=[1, 2, 3],
                portfolio_columns=['Strategy_2', 'Strategy_3', 'Strategy_4'],
                fitness_score=0.9,
                metrics={},
                optimization_time=1.0,
                algorithm_used='genetic'
            )
        }
        
        analysis = self.optimizer.analyze_zone_correlation(zone_results)
        
        self.assertIn('zone_similarity_matrix', analysis)
        self.assertIn('zone_names', analysis)
        self.assertIn('total_unique_strategies', analysis)
        self.assertIn('strategy_overlap', analysis)
        
        # Check similarity calculation (2 common strategies out of 4 total)
        # Jaccard index = 2/4 = 0.5
        self.assertAlmostEqual(analysis['strategy_overlap']['zone1-zone2'], 0.5)
        
    def test_report_generation(self):
        """Test zone report generation"""
        zone_results = {
            'zone1': ZoneResult(
                zone_name='zone1',
                portfolio_indices=[0, 1],
                portfolio_columns=['Strategy_1', 'Strategy_2'],
                fitness_score=0.75,
                metrics={'roi': 0.12, 'sharpe': 1.1},
                optimization_time=1.5,
                algorithm_used='genetic'
            )
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            report_path = f.name
        
        try:
            self.optimizer.generate_zone_report(zone_results, report_path)
            
            # Check report exists and contains expected content
            self.assertTrue(os.path.exists(report_path))
            
            with open(report_path, 'r') as f:
                content = f.read()
                
            self.assertIn("ZONE OPTIMIZATION REPORT", content)
            self.assertIn("ZONE1", content.upper())
            self.assertIn("0.750000", content)
            self.assertIn("genetic", content)
            
        finally:
            os.unlink(report_path)


class TestZoneOptimizerDAL(unittest.TestCase):
    """Test DAL integration"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create mock DAL
        self.mock_dal = Mock()
        self.mock_dal.is_connected = True
        self.mock_dal.supports_gpu = False
        
        self.optimizer = ZoneOptimizerDAL(dal=self.mock_dal)
        
    def test_load_zone_data_from_table(self):
        """Test loading zone data from database"""
        # Mock data
        test_data = pd.DataFrame({
            'Date': pd.date_range('2024-01-01', periods=8),
            'Zone': ['zone1', 'zone1', 'zone2', 'zone2', 'zone3', 'zone3', 'zone4', 'zone4'],
            'Strategy_1': np.random.randn(8),
            'Strategy_2': np.random.randn(8)
        })
        
        self.mock_dal.execute_gpu_query.return_value = test_data
        
        zones = self.optimizer.load_zone_data_from_table('test_table')
        
        # Should have called execute_gpu_query
        self.mock_dal.execute_gpu_query.assert_called()
        
        # Check zones
        self.assertEqual(len(zones), 4)
        for zone_name in ['zone1', 'zone2', 'zone3', 'zone4']:
            self.assertIn(zone_name, zones)
            self.assertEqual(len(zones[zone_name]), 2)
            
    def test_save_results_to_db(self):
        """Test saving results to database"""
        result = ZoneResult(
            zone_name='zone1',
            portfolio_indices=[0, 1, 2],
            portfolio_columns=['Strategy_1', 'Strategy_2', 'Strategy_3'],
            fitness_score=0.85,
            metrics={'roi': 0.15},
            optimization_time=2.5,
            algorithm_used='genetic'
        )
        
        self.mock_dal.create_dynamic_table.return_value = True
        
        success = self.optimizer._save_zone_result_to_db('zone1', result, 'source_table')
        
        self.assertTrue(success)
        self.mock_dal.create_dynamic_table.assert_called_once()
        
        # Check the DataFrame passed
        call_args = self.mock_dal.create_dynamic_table.call_args
        df = call_args[0][1]  # Second positional argument
        
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]['zone_name'], 'zone1')
        self.assertEqual(df.iloc[0]['fitness_score'], 0.85)


if __name__ == '__main__':
    unittest.main()