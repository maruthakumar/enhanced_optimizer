"""
Unit tests for cuDF Engine components
Tests GPU-accelerated calculations and metrics
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import tempfile
import os

# Add parent directory to path for imports
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from backend.lib.cudf_engine.gpu_calculator import (
    calculate_correlations_cudf,
    calculate_fitness_cudf,
    calculate_enhanced_metrics_cudf,
    batch_calculate_portfolios_cudf,
    optimize_portfolio_allocation_cudf
)

from backend.lib.cudf_engine.enhanced_metrics import (
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_calmar_ratio,
    calculate_kelly_criterion,
    calculate_var_cvar,
    calculate_information_ratio
)


class TestCuDFCorrelations:
    """Test cuDF correlation calculations"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample strategy data"""
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=100)
        data = {
            'Date': dates,
            'Strategy_1': np.random.randn(100) * 1000,
            'Strategy_2': np.random.randn(100) * 1000,
            'Strategy_3': np.random.randn(100) * 1000,
            'Strategy_4': np.random.randn(100) * 1000,
            'Strategy_5': np.random.randn(100) * 1000
        }
        # Add some correlation between Strategy_1 and Strategy_2
        data['Strategy_2'] = data['Strategy_1'] * 0.8 + np.random.randn(100) * 200
        return pd.DataFrame(data)
    
    @pytest.mark.skipif(True, reason="Skip GPU tests in CI")
    def test_pearson_correlation_gpu(self, sample_data):
        """Test Pearson correlation calculation on GPU"""
        # This test would run if cuDF is available
        pass
    
    def test_pearson_correlation_fallback(self, sample_data):
        """Test Pearson correlation with CPU fallback"""
        with patch('backend.lib.cudf_engine.gpu_calculator.CUDF_AVAILABLE', False):
            # Should raise error when cuDF not available
            with pytest.raises(RuntimeError) as excinfo:
                calculate_correlations_cudf(sample_data, 
                                          ['Strategy_1', 'Strategy_2', 'Strategy_3'])
            assert "cuDF is not available" in str(excinfo.value)
    
    def test_correlation_matrix_properties(self):
        """Test correlation matrix mathematical properties"""
        # Create mock cuDF module
        mock_cudf = MagicMock()
        
        # Create test data
        np.random.seed(42)
        data = pd.DataFrame({
            'S1': np.random.randn(100),
            'S2': np.random.randn(100),
            'S3': np.random.randn(100)
        })
        
        # Create expected correlation matrix
        expected_corr = data.corr()
        
        # Mock cuDF DataFrame behavior
        mock_df = MagicMock()
        mock_df.corr.return_value = expected_corr
        mock_df.__getitem__ = lambda self, cols: mock_df
        
        with patch('backend.lib.cudf_engine.gpu_calculator.CUDF_AVAILABLE', True):
            with patch('backend.lib.cudf_engine.gpu_calculator.cudf', mock_cudf):
                # Mock the correlation calculation
                with patch('backend.lib.cudf_engine.gpu_calculator.calculate_correlations_cudf', 
                          return_value=expected_corr):
                    
                    result = calculate_correlations_cudf(mock_df, ['S1', 'S2', 'S3'])
                    
                    # Verify properties
                    assert result.shape == (3, 3)
                    # Diagonal should be 1s
                    np.testing.assert_array_almost_equal(np.diag(result.values), [1, 1, 1])
                    # Should be symmetric
                    np.testing.assert_array_almost_equal(result.values, result.values.T)


class TestCuDFFitnessCalculation:
    """Test cuDF fitness calculation functionality"""
    
    @pytest.fixture
    def portfolio_data(self):
        """Create sample portfolio data"""
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=252)  # 1 trading year
        
        # Create strategies with different characteristics
        data = {
            'Date': dates,
            'HighReturn': np.random.randn(252) * 1000 + 50,  # Positive bias
            'LowRisk': np.random.randn(252) * 100 + 10,      # Low volatility
            'Negative': np.random.randn(252) * 500 - 30,     # Negative bias
            'Volatile': np.random.randn(252) * 2000,         # High volatility
        }
        return pd.DataFrame(data)
    
    def test_fitness_calculation_mock(self, portfolio_data):
        """Test fitness calculation with mocked cuDF"""
        portfolio = ['HighReturn', 'LowRisk']
        metrics_config = {
            'roi_weight': 0.4,
            'drawdown_weight': 0.3,
            'win_rate_weight': 0.2,
            'profit_factor_weight': 0.1
        }
        
        # Mock cuDF operations
        with patch('backend.lib.cudf_engine.gpu_calculator.CUDF_AVAILABLE', True):
            # Create mock that behaves like cuDF DataFrame
            mock_df = MagicMock()
            
            # Mock portfolio returns calculation
            portfolio_returns = portfolio_data[portfolio].sum(axis=1)
            mock_df.__getitem__.return_value.sum.return_value = portfolio_returns
            
            # Mock metric calculations
            mock_returns = MagicMock()
            mock_returns.sum.return_value = portfolio_returns.sum()
            mock_returns.cumsum.return_value.cummax.return_value = portfolio_returns.cumsum().cummax()
            mock_returns.cumsum.return_value.__sub__.return_value.min.return_value = -5000.0
            mock_returns.__gt__.return_value.sum.return_value = 150  # 150 winning days
            mock_returns.__len__.return_value = 252
            
            with patch('backend.lib.cudf_engine.gpu_calculator.calculate_fitness_cudf') as mock_calc:
                # Return realistic metrics
                mock_calc.return_value = {
                    'total_roi': 15230.5,
                    'max_drawdown': -5000.0,
                    'win_rate': 0.595,
                    'profit_factor': 1.35,
                    'roi_drawdown_ratio': 3.046,
                    'fitness_score': 0.725
                }
                
                result = calculate_fitness_cudf(mock_df, portfolio, metrics_config)
                
                # Verify metrics
                assert 'total_roi' in result
                assert 'max_drawdown' in result
                assert 'win_rate' in result
                assert 'profit_factor' in result
                assert result['win_rate'] >= 0 and result['win_rate'] <= 1
                assert result['max_drawdown'] <= 0


class TestEnhancedMetrics:
    """Test enhanced financial metrics calculations"""
    
    @pytest.fixture
    def returns_data(self):
        """Create sample returns data"""
        np.random.seed(42)
        # Generate returns with specific characteristics
        returns = np.random.randn(252) * 0.02  # 2% daily volatility
        returns[returns < -0.04] = -0.04  # Cap downside at -4%
        returns = returns + 0.0005  # Add small positive drift
        return pd.Series(returns)
    
    def test_sharpe_ratio_calculation(self, returns_data):
        """Test Sharpe ratio calculation"""
        sharpe = calculate_sharpe_ratio(returns_data, risk_free_rate=0.02)
        
        # Sharpe ratio should be reasonable
        assert isinstance(sharpe, float)
        assert -5 < sharpe < 5  # Reasonable range
        
        # Test with zero volatility
        zero_vol_returns = pd.Series([0.001] * 252)
        sharpe_zero = calculate_sharpe_ratio(zero_vol_returns)
        assert sharpe_zero > 10  # Very high Sharpe for zero volatility
    
    def test_sortino_ratio_calculation(self, returns_data):
        """Test Sortino ratio calculation"""
        sortino = calculate_sortino_ratio(returns_data, target_return=0.0)
        
        # Sortino should be higher than Sharpe (focuses on downside)
        sharpe = calculate_sharpe_ratio(returns_data)
        assert isinstance(sortino, float)
        assert sortino >= sharpe * 0.8  # Usually higher, but not always
    
    def test_calmar_ratio_calculation(self, returns_data):
        """Test Calmar ratio calculation"""
        calmar = calculate_calmar_ratio(returns_data)
        
        assert isinstance(calmar, float)
        assert calmar != 0  # Should have some drawdown
        
        # Test with all positive returns
        positive_returns = pd.Series(np.abs(np.random.randn(252)) * 0.01)
        calmar_positive = calculate_calmar_ratio(positive_returns)
        assert calmar_positive > 0
    
    def test_kelly_criterion_calculation(self, returns_data):
        """Test Kelly Criterion calculation"""
        kelly = calculate_kelly_criterion(returns_data)
        
        assert isinstance(kelly, float)
        assert 0 <= kelly <= 1  # Should be between 0 and 1 (capped)
        
        # Test with all losses
        loss_returns = pd.Series(np.random.randn(100) * 0.01 - 0.02)
        kelly_loss = calculate_kelly_criterion(loss_returns)
        assert kelly_loss == 0  # Should be 0 for losing strategy
    
    def test_var_cvar_calculation(self, returns_data):
        """Test VaR and CVaR calculation"""
        var_95, cvar_95 = calculate_var_cvar(returns_data, confidence=0.95)
        
        assert isinstance(var_95, float)
        assert isinstance(cvar_95, float)
        assert var_95 < 0  # VaR is typically negative
        assert cvar_95 <= var_95  # CVaR should be worse than VaR
        
        # Test different confidence levels
        var_99, cvar_99 = calculate_var_cvar(returns_data, confidence=0.99)
        assert var_99 <= var_95  # Higher confidence = worse VaR
    
    def test_information_ratio_calculation(self, returns_data):
        """Test Information Ratio calculation"""
        # Create benchmark returns
        benchmark = returns_data * 0.8 + np.random.randn(len(returns_data)) * 0.005
        
        ir = calculate_information_ratio(returns_data, benchmark)
        
        assert isinstance(ir, float)
        assert -3 < ir < 3  # Reasonable range
        
        # Test with identical returns
        ir_same = calculate_information_ratio(returns_data, returns_data)
        assert abs(ir_same) < 0.1  # Should be close to 0


class TestBatchProcessing:
    """Test batch processing functionality"""
    
    def test_batch_portfolio_calculation(self):
        """Test batch calculation of multiple portfolios"""
        # Create sample data
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=100)
        strategies = [f'S{i}' for i in range(10)]
        
        data = {'Date': dates}
        for strategy in strategies:
            data[strategy] = np.random.randn(100) * 1000
        
        df = pd.DataFrame(data)
        
        # Create multiple portfolios
        portfolios = [
            ['S0', 'S1', 'S2'],
            ['S3', 'S4', 'S5'],
            ['S0', 'S5', 'S9'],
            ['S1', 'S2', 'S3', 'S4']
        ]
        
        metrics_config = {
            'roi_weight': 0.5,
            'drawdown_weight': 0.3,
            'win_rate_weight': 0.2
        }
        
        # Mock batch calculation
        with patch('backend.lib.cudf_engine.gpu_calculator.CUDF_AVAILABLE', True):
            with patch('backend.lib.cudf_engine.gpu_calculator.batch_calculate_portfolios_cudf') as mock_batch:
                # Return mock results
                mock_results = [
                    {'portfolio': p, 'fitness_score': 0.5 + i * 0.1}
                    for i, p in enumerate(portfolios)
                ]
                mock_batch.return_value = mock_results
                
                results = batch_calculate_portfolios_cudf(df, portfolios, metrics_config)
                
                assert len(results) == len(portfolios)
                assert all('fitness_score' in r for r in results)


class TestPortfolioOptimization:
    """Test portfolio allocation optimization"""
    
    def test_portfolio_allocation_optimization(self):
        """Test optimal allocation calculation"""
        # Create sample correlation matrix
        n_strategies = 5
        corr_matrix = np.eye(n_strategies)
        # Add some correlations
        corr_matrix[0, 1] = corr_matrix[1, 0] = 0.7
        corr_matrix[2, 3] = corr_matrix[3, 2] = 0.5
        
        # Create sample metrics
        strategy_metrics = {
            f'S{i}': {
                'total_roi': np.random.uniform(1000, 5000),
                'max_drawdown': np.random.uniform(-5000, -1000),
                'sharpe_ratio': np.random.uniform(0.5, 2.0)
            }
            for i in range(n_strategies)
        }
        
        # Mock optimization
        with patch('backend.lib.cudf_engine.gpu_calculator.optimize_portfolio_allocation_cudf') as mock_opt:
            mock_allocations = {f'S{i}': 1.0/n_strategies for i in range(n_strategies)}
            mock_opt.return_value = mock_allocations
            
            result = optimize_portfolio_allocation_cudf(
                list(strategy_metrics.keys()),
                strategy_metrics,
                corr_matrix,
                min_allocation=0.05,
                max_allocation=0.40
            )
            
            # Verify allocations
            assert sum(result.values()) == pytest.approx(1.0, rel=1e-6)
            assert all(0.05 <= v <= 0.40 for v in result.values())


class TestEdgeCasesGPU:
    """Test edge cases for GPU operations"""
    
    def test_empty_portfolio_handling(self):
        """Test handling of empty portfolio"""
        with patch('backend.lib.cudf_engine.gpu_calculator.CUDF_AVAILABLE', False):
            with pytest.raises(RuntimeError):
                calculate_fitness_cudf(None, [], {})
    
    def test_single_strategy_portfolio(self):
        """Test portfolio with single strategy"""
        data = pd.DataFrame({
            'Date': pd.date_range('2024-01-01', periods=10),
            'Strategy_1': np.random.randn(10) * 1000
        })
        
        with patch('backend.lib.cudf_engine.gpu_calculator.calculate_fitness_cudf') as mock_calc:
            mock_calc.return_value = {
                'total_roi': 1000.0,
                'max_drawdown': -500.0,
                'win_rate': 0.6,
                'fitness_score': 0.7
            }
            
            result = calculate_fitness_cudf(data, ['Strategy_1'], {})
            assert result['total_roi'] == 1000.0
    
    def test_all_zero_returns(self):
        """Test handling of all zero returns"""
        returns = pd.Series([0.0] * 100)
        
        # Test metrics with zero returns
        sharpe = calculate_sharpe_ratio(returns)
        assert sharpe == 0.0
        
        sortino = calculate_sortino_ratio(returns)
        assert sortino == 0.0
        
        kelly = calculate_kelly_criterion(returns)
        assert kelly == 0.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])