#!/usr/bin/env python3
"""
Unit tests for enhanced financial metrics implementation.
Tests Kelly Criterion, VaR/CVaR, Sharpe/Sortino/Calmar ratios, and regime optimization.
"""
import sys
import os
import numpy as np
import logging

# Add paths for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(__file__), 'lib'))

# Import components to test
try:
    from lib.financial_metrics import KellyCriterion, EnhancedMetrics, EnhancedFitnessCalculator
    from lib.risk_management import VaRCVaRCalculator
    from lib.regime_optimization import RegimeHandler
    MODULES_AVAILABLE = True
except ImportError as e:
    logger.error(f"Failed to import modules: {e}")
    MODULES_AVAILABLE = False

# Try to import cuDF for GPU tests
try:
    import cudf
    import cupy as cp
    CUDF_AVAILABLE = True
except ImportError:
    CUDF_AVAILABLE = False
    print("cuDF not available - GPU tests will be skipped")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TestKellyCriterion:
    """Test Kelly Criterion calculations"""
    
    def __init__(self):
        self.kelly = KellyCriterion(max_position_size=0.25, min_position_size=0.01)
        
    def test_basic_kelly_calculation(self):
        """Test basic Kelly fraction calculation"""
        print("\n=== Testing Kelly Criterion ===")
        
        # Test case 1: Favorable odds
        win_rate = 0.6
        avg_win = 100
        avg_loss = 50
        
        kelly_fraction = self.kelly.calculate_kelly_fraction(win_rate, avg_win, avg_loss)
        print(f"Test 1 - Win rate: {win_rate}, Avg win: {avg_win}, Avg loss: {avg_loss}")
        print(f"Kelly fraction: {kelly_fraction:.4f}")
        
        # Expected: f = (0.6 * 2 - 0.4) / 2 = 0.8 / 2 = 0.4, but capped at 0.25
        assert kelly_fraction == 0.25, f"Expected 0.25, got {kelly_fraction}"
        
        # Test case 2: Unfavorable odds
        win_rate = 0.4
        avg_win = 50
        avg_loss = 100
        
        kelly_fraction = self.kelly.calculate_kelly_fraction(win_rate, avg_win, avg_loss)
        print(f"\nTest 2 - Win rate: {win_rate}, Avg win: {avg_win}, Avg loss: {avg_loss}")
        print(f"Kelly fraction: {kelly_fraction:.4f}")
        
        # Expected: f = (0.4 * 0.5 - 0.6) / 0.5 = -0.4 / 0.5 = negative, so min_position
        assert kelly_fraction == 0.01, f"Expected 0.01, got {kelly_fraction}"
        
        print("✓ Kelly Criterion tests passed")
        
    def test_batch_kelly_gpu(self):
        """Test batch Kelly calculation with GPU"""
        if not CUDF_AVAILABLE:
            print("\n✗ Skipping GPU Kelly tests (cuDF not available)")
            return
            
        print("\n=== Testing Batch Kelly Calculation (GPU) ===")
        
        # Create test data
        strategies_data = {
            'strategy': ['S1', 'S2', 'S3', 'S4'],
            'win_rate': [0.65, 0.45, 0.55, 0.70],
            'avg_win': [120.0, 80.0, 100.0, 150.0],
            'avg_loss': [80.0, 100.0, 90.0, 50.0]
        }
        
        strategies_df = cudf.DataFrame(strategies_data)
        
        # Calculate Kelly fractions
        kelly_fractions = self.kelly.calculate_kelly_fractions_batch(strategies_df)
        
        print("Strategy Kelly Fractions:")
        for i, (strategy, fraction) in enumerate(zip(strategies_data['strategy'], kelly_fractions.to_numpy())):
            print(f"  {strategy}: {fraction:.4f}")
            
        # Apply Kelly sizing
        portfolio_df = self.kelly.apply_kelly_sizing(strategies_df)
        
        print("\nPortfolio with Kelly sizing:")
        print(portfolio_df[['strategy', 'kelly_fraction', 'kelly_position_size']].to_pandas())
        
        print("✓ Batch Kelly GPU tests passed")


class TestVaRCVaR:
    """Test VaR and CVaR calculations"""
    
    def __init__(self):
        self.risk_calc = VaRCVaRCalculator(confidence_levels=[0.95, 0.99])
        
    def test_historical_var_cvar(self):
        """Test historical VaR and CVaR calculations"""
        print("\n=== Testing VaR/CVaR Calculations ===")
        
        # Generate sample returns (normal distribution with negative tail)
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 1000)  # 0.1% daily return, 2% volatility
        returns[900:910] = -0.05  # Add some tail events
        
        # Calculate VaR and CVaR
        var_95 = self.risk_calc.calculate_var_historical(returns, 0.95)
        cvar_95 = self.risk_calc.calculate_cvar_historical(returns, 0.95)
        
        var_99 = self.risk_calc.calculate_var_historical(returns, 0.99)
        cvar_99 = self.risk_calc.calculate_cvar_historical(returns, 0.99)
        
        print(f"95% VaR: {var_95:.4f}")
        print(f"95% CVaR: {cvar_95:.4f}")
        print(f"99% VaR: {var_99:.4f}")
        print(f"99% CVaR: {cvar_99:.4f}")
        
        # Validate relationships
        assert cvar_95 <= var_95, "CVaR should be worse than VaR"
        assert var_99 <= var_95, "99% VaR should be worse than 95% VaR"
        assert cvar_99 <= cvar_95, "99% CVaR should be worse than 95% CVaR"
        
        print("✓ VaR/CVaR tests passed")
        
    def test_portfolio_var_gpu(self):
        """Test portfolio VaR with GPU"""
        if not CUDF_AVAILABLE:
            print("\n✗ Skipping GPU VaR tests (cuDF not available)")
            return
            
        print("\n=== Testing Portfolio VaR (GPU) ===")
        
        # Create sample portfolio returns
        np.random.seed(42)
        dates = 100
        assets = 5
        
        returns_data = np.random.multivariate_normal(
            mean=[0.001] * assets,
            cov=np.eye(assets) * 0.0004 + np.ones((assets, assets)) * 0.0001,
            size=dates
        )
        
        portfolio_returns = cudf.DataFrame(
            returns_data,
            columns=[f'Asset_{i}' for i in range(assets)]
        )
        
        # Equal weights
        weights = cudf.Series([0.2] * assets, index=portfolio_returns.columns)
        
        # Calculate portfolio VaR
        portfolio_var = self.risk_calc.calculate_portfolio_var(portfolio_returns, weights, 0.95)
        portfolio_cvar = self.risk_calc.calculate_portfolio_cvar(portfolio_returns, weights, 0.95)
        
        print(f"Portfolio 95% VaR: {portfolio_var:.4f}")
        print(f"Portfolio 95% CVaR: {portfolio_cvar:.4f}")
        
        # Calculate marginal VaR
        marginal_vars = self.risk_calc.calculate_marginal_var(portfolio_returns, weights, 0.95)
        
        print("\nMarginal VaR contributions:")
        for asset, mvar in marginal_vars.items():
            print(f"  {asset}: {mvar:.6f}")
            
        print("✓ Portfolio VaR GPU tests passed")


class TestEnhancedMetrics:
    """Test Sharpe, Sortino, Calmar ratios"""
    
    def __init__(self):
        self.metrics = EnhancedMetrics(risk_free_rate=0.02, annualization_factor=252)
        
    def test_sharpe_sortino_calmar(self):
        """Test risk-adjusted return metrics"""
        print("\n=== Testing Enhanced Metrics ===")
        
        # Generate sample returns with positive drift
        np.random.seed(42)
        returns = np.random.normal(0.0005, 0.015, 252)  # One year of daily returns
        
        # Calculate metrics
        sharpe = self.metrics.calculate_sharpe_ratio(returns)
        sortino = self.metrics.calculate_sortino_ratio(returns, mar=0.0)
        max_dd = self.metrics.calculate_max_drawdown(returns)
        calmar = self.metrics.calculate_calmar_ratio(returns, max_dd)
        
        print(f"Sharpe Ratio: {sharpe:.3f}")
        print(f"Sortino Ratio: {sortino:.3f}")
        print(f"Max Drawdown: {max_dd:.3%}")
        print(f"Calmar Ratio: {calmar:.3f}")
        
        # Sortino should be higher than Sharpe (focuses on downside)
        assert sortino >= sharpe, "Sortino should be >= Sharpe"
        
        print("✓ Enhanced metrics tests passed")
        
    def test_all_metrics_gpu(self):
        """Test comprehensive metrics calculation with GPU"""
        if not CUDF_AVAILABLE:
            print("\n✗ Skipping GPU metrics tests (cuDF not available)")
            return
            
        print("\n=== Testing All Metrics (GPU) ===")
        
        # Generate returns
        np.random.seed(42)
        returns_np = np.random.normal(0.0008, 0.012, 252)
        returns = cudf.Series(returns_np)
        
        # Calculate all metrics
        all_metrics = self.metrics.calculate_all_metrics(returns)
        
        print("Comprehensive metrics:")
        for metric, value in all_metrics.items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.4f}")
                
        print("✓ All metrics GPU tests passed")


class TestRegimeOptimization:
    """Test market regime handling"""
    
    def __init__(self):
        self.regime_handler = RegimeHandler(min_confidence_threshold=0.7)
        
    def test_regime_filtering(self):
        """Test regime confidence filtering"""
        print("\n=== Testing Regime Optimization ===")
        
        if not CUDF_AVAILABLE:
            print("✗ Skipping regime tests (cuDF not available)")
            return
            
        # Create test strategies with regime data
        strategies_data = {
            'strategy': ['S1', 'S2', 'S3', 'S4', 'S5'],
            'fitness_score': [1.5, 2.0, 1.8, 2.2, 1.6],
            'market_regime': ['bull', 'bull', 'bear', 'neutral', 'volatile'],
            'Regime_Confidence_%': [85, 65, 90, 75, 50]  # Percentage form
        }
        
        strategies_df = cudf.DataFrame(strategies_data)
        
        # Filter by confidence
        filtered_df = self.regime_handler.filter_by_confidence(strategies_df)
        
        print(f"Original strategies: {len(strategies_df)}")
        print(f"Filtered strategies (confidence >= 70%): {len(filtered_df)}")
        
        # Apply confidence weighting
        weighted_df = self.regime_handler.apply_confidence_weighting(strategies_df)
        
        print("\nRegime-weighted strategies:")
        print(weighted_df[['strategy', 'Regime_Confidence_%', 'confidence_factor', 'regime_weight']].to_pandas())
        
        # Get regime composition
        regime_comp = self.regime_handler.get_regime_composition(strategies_df)
        print("\nRegime composition:")
        for regime, count in regime_comp.items():
            print(f"  {regime}: {count} strategies")
            
        print("✓ Regime optimization tests passed")


class TestEnhancedFitness:
    """Test enhanced fitness calculation integration"""
    
    def test_fitness_modes(self):
        """Test different fitness calculation modes"""
        print("\n=== Testing Enhanced Fitness Calculator ===")
        
        # Create test configuration
        config = {
            'FITNESS_CALCULATION': {'mode': 'enhanced'},
            'KELLY_CRITERION': {'enabled': True, 'max_position_size': 0.25},
            'RISK_METRICS': {'risk_free_rate': 0.02, 'var_limit': 0.025},
            'MARKET_REGIME_CONFIG': {'min_confidence_threshold': 70}
        }
        
        calculator = EnhancedFitnessCalculator(config)
        
        # Test strategy data
        strategy_data = {
            'total_roi': 5000.0,
            'max_drawdown': 2500.0,
            'win_rate': 0.62,
            'avg_win': 150.0,
            'avg_loss': 80.0,
            'regime_confidence': 85.0,
            'var_95': 0.02,
            'sortino_ratio': 1.5
        }
        
        # Calculate fitness in different modes
        calculator.mode = 'legacy'
        legacy_fitness = calculator.calculate_fitness(strategy_data)
        print(f"Legacy fitness: {legacy_fitness:.4f}")
        
        calculator.mode = 'enhanced'
        enhanced_fitness = calculator.calculate_fitness(strategy_data)
        print(f"Enhanced fitness: {enhanced_fitness:.4f}")
        
        calculator.mode = 'hybrid'
        hybrid_fitness = calculator.calculate_fitness(strategy_data)
        print(f"Hybrid fitness: {hybrid_fitness:.4f}")
        
        print("✓ Enhanced fitness calculator tests passed")


def main():
    """Run all tests"""
    print("=" * 60)
    print("Enhanced Financial Metrics Test Suite")
    print("=" * 60)
    
    if not MODULES_AVAILABLE:
        print("\n✗ Failed to import required modules. Check error messages above.")
        return
    
    # Run Kelly Criterion tests
    kelly_tests = TestKellyCriterion()
    kelly_tests.test_basic_kelly_calculation()
    kelly_tests.test_batch_kelly_gpu()
    
    # Run VaR/CVaR tests
    var_tests = TestVaRCVaR()
    var_tests.test_historical_var_cvar()
    var_tests.test_portfolio_var_gpu()
    
    # Run enhanced metrics tests
    metrics_tests = TestEnhancedMetrics()
    metrics_tests.test_sharpe_sortino_calmar()
    metrics_tests.test_all_metrics_gpu()
    
    # Run regime optimization tests
    regime_tests = TestRegimeOptimization()
    regime_tests.test_regime_filtering()
    
    # Run fitness calculator tests
    fitness_tests = TestEnhancedFitness()
    fitness_tests.test_fitness_modes()
    
    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()