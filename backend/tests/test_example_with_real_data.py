#!/usr/bin/env python3
"""
Example test file showing how to use anonymized production data
Replace synthetic data generation with real anonymized data
"""

import pytest
import pandas as pd
import numpy as np
import os
from pathlib import Path


class TestWithAnonymizedData:
    """Example tests using anonymized production data"""
    
    @pytest.fixture
    def test_data_path(self):
        """Get path to anonymized test data"""
        # Check environment variable first
        env_path = os.environ.get('TEST_DATA_PATH')
        if env_path and os.path.exists(env_path):
            return env_path
        
        # Default to small test dataset
        default_path = Path(__file__).parent / 'data' / 'test_data_small.csv'
        if default_path.exists():
            return str(default_path)
        
        pytest.skip("Anonymized test data not found. Run setup-test-data.sh first.")
    
    @pytest.fixture
    def test_data(self, test_data_path):
        """Load anonymized test data"""
        df = pd.read_csv(test_data_path)
        
        # Identify strategy columns
        metadata_cols = ['Date', 'Day']
        strategy_cols = [col for col in df.columns 
                        if col not in metadata_cols and col.startswith('STRATEGY_')]
        
        return df, strategy_cols
    
    def test_data_structure(self, test_data):
        """Test that anonymized data has expected structure"""
        df, strategy_cols = test_data
        
        # Check basic properties
        assert len(df) > 0, "Test data should not be empty"
        assert len(strategy_cols) > 0, "Should have strategy columns"
        
        # Check column naming
        for col in strategy_cols:
            assert col.startswith('STRATEGY_'), f"Strategy column should be anonymized: {col}"
        
        # Check data types
        assert df['Date'].dtype == 'object', "Date should be string"
        for col in strategy_cols:
            assert pd.api.types.is_numeric_dtype(df[col]), f"{col} should be numeric"
    
    def test_portfolio_calculation(self, test_data):
        """Test portfolio calculations with real data"""
        df, strategy_cols = test_data
        
        # Select a small portfolio
        portfolio_size = min(10, len(strategy_cols))
        selected_strategies = strategy_cols[:portfolio_size]
        
        # Calculate portfolio returns
        portfolio_returns = df[selected_strategies].sum(axis=1)
        
        # Basic assertions
        assert len(portfolio_returns) == len(df)
        assert portfolio_returns.dtype in [np.float64, np.float32]
        
        # Calculate metrics
        total_return = portfolio_returns.sum()
        win_rate = (portfolio_returns > 0).mean()
        
        # Validate metrics are reasonable
        assert not np.isnan(total_return), "Total return should not be NaN"
        assert 0 <= win_rate <= 1, "Win rate should be between 0 and 1"
    
    def test_correlation_matrix(self, test_data):
        """Test correlation calculations with real data"""
        df, strategy_cols = test_data
        
        # Use subset for correlation (computationally expensive)
        sample_size = min(20, len(strategy_cols))
        sample_strategies = strategy_cols[:sample_size]
        
        # Calculate correlation matrix
        corr_matrix = df[sample_strategies].corr()
        
        # Validate correlation matrix
        assert corr_matrix.shape == (sample_size, sample_size)
        assert np.allclose(corr_matrix.values.diagonal(), 1.0), "Diagonal should be 1"
        assert corr_matrix.equals(corr_matrix.T), "Matrix should be symmetric"
        
        # Check correlation values are valid
        assert corr_matrix.min().min() >= -1.0, "Correlations should be >= -1"
        assert corr_matrix.max().max() <= 1.0, "Correlations should be <= 1"
    
    def test_edge_cases(self):
        """Test with edge case datasets"""
        edge_case_dir = Path(__file__).parent / 'data' / 'edge_cases'
        
        if not edge_case_dir.exists():
            pytest.skip("Edge case data not found")
        
        # Test max drawdown data
        max_dd_file = edge_case_dir / 'max_drawdown.csv'
        if max_dd_file.exists():
            df = pd.read_csv(max_dd_file)
            assert len(df) > 0, "Max drawdown data should not be empty"
            
            # Verify it contains drawdown column if added by extractor
            if 'portfolio_drawdown' in df.columns:
                assert df['portfolio_drawdown'].max() <= 0, "Drawdowns should be negative"
    
    @pytest.mark.parametrize("algorithm", [
        "genetic_algorithm",
        "particle_swarm_optimization",
        "simulated_annealing"
    ])
    def test_algorithm_with_real_data(self, test_data, algorithm):
        """Test algorithms with real anonymized data"""
        df, strategy_cols = test_data
        
        # This is a template - actual algorithm imports would go here
        # from algorithms.genetic_algorithm import GeneticAlgorithm
        
        # For now, just verify data is ready for algorithms
        assert len(strategy_cols) >= 10, f"Need at least 10 strategies for {algorithm}"
        
        # Verify data has no NaN values
        data_matrix = df[strategy_cols].values
        assert not np.isnan(data_matrix).any(), "Data should not contain NaN values"


@pytest.mark.integration
class TestIntegrationWithRealData:
    """Integration tests using larger anonymized datasets"""
    
    @pytest.fixture
    def medium_data_path(self):
        """Get path to medium anonymized dataset"""
        path = Path(__file__).parent / 'data' / 'test_data_medium.csv'
        if not path.exists():
            pytest.skip("Medium dataset not found. Run setup-test-data.sh first.")
        return str(path)
    
    def test_large_scale_processing(self, medium_data_path):
        """Test processing with medium-sized real data"""
        df = pd.read_csv(medium_data_path)
        
        # Verify we have substantial data
        assert len(df) > 1000, "Medium dataset should have >1000 rows"
        
        # Get strategy columns
        strategy_cols = [col for col in df.columns 
                        if col.startswith('STRATEGY_')]
        assert len(strategy_cols) > 1000, "Should have many strategies"
        
        # Test memory-efficient processing
        chunk_size = 1000
        total_sum = 0
        
        for start_idx in range(0, len(strategy_cols), chunk_size):
            end_idx = min(start_idx + chunk_size, len(strategy_cols))
            chunk_cols = strategy_cols[start_idx:end_idx]
            chunk_sum = df[chunk_cols].sum().sum()
            total_sum += chunk_sum
        
        assert not np.isnan(total_sum), "Chunked processing should produce valid results"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])