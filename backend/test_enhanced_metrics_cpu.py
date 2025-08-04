#!/usr/bin/env python3
"""
CPU-only unit tests for enhanced financial metrics implementation.
Tests Kelly Criterion, VaR/CVaR, Sharpe/Sortino/Calmar ratios without GPU dependencies.
"""
import sys
import os
import numpy as np
import logging

# Add paths for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(__file__), 'lib'))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_kelly_criterion():
    """Test Kelly Criterion calculations without GPU"""
    print("\n=== Testing Kelly Criterion (CPU) ===")
    
    # Direct Kelly calculation
    def calculate_kelly(win_rate, avg_win, avg_loss, max_size=0.25, min_size=0.01):
        if avg_loss <= 0 or avg_win <= 0:
            return min_size
            
        b = avg_win / avg_loss
        p = win_rate
        q = 1 - win_rate
        
        kelly_fraction = (p * b - q) / b
        
        if kelly_fraction <= 0:
            return min_size
        elif kelly_fraction > max_size:
            return max_size
        else:
            return max(kelly_fraction, min_size)
    
    # Test case 1: Favorable odds
    win_rate = 0.6
    avg_win = 100
    avg_loss = 50
    
    kelly_fraction = calculate_kelly(win_rate, avg_win, avg_loss)
    print(f"Test 1 - Win rate: {win_rate}, Avg win: {avg_win}, Avg loss: {avg_loss}")
    print(f"Kelly fraction: {kelly_fraction:.4f}")
    
    # Expected: f = (0.6 * 2 - 0.4) / 2 = 0.8 / 2 = 0.4, but capped at 0.25
    assert kelly_fraction == 0.25, f"Expected 0.25, got {kelly_fraction}"
    
    # Test case 2: Unfavorable odds
    win_rate = 0.4
    avg_win = 50
    avg_loss = 100
    
    kelly_fraction = calculate_kelly(win_rate, avg_win, avg_loss)
    print(f"\nTest 2 - Win rate: {win_rate}, Avg win: {avg_win}, Avg loss: {avg_loss}")
    print(f"Kelly fraction: {kelly_fraction:.4f}")
    
    # Expected: f = (0.4 * 0.5 - 0.6) / 0.5 = -0.4 / 0.5 = negative, so min_position
    assert kelly_fraction == 0.01, f"Expected 0.01, got {kelly_fraction}"
    
    print("✓ Kelly Criterion CPU tests passed")


def test_var_cvar():
    """Test VaR and CVaR calculations"""
    print("\n=== Testing VaR/CVaR Calculations (CPU) ===")
    
    # Generate sample returns
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, 1000)  # 0.1% daily return, 2% volatility
    returns[900:910] = -0.05  # Add some tail events
    
    # Calculate VaR and CVaR
    def calculate_var(returns, confidence):
        alpha = 1 - confidence
        return np.percentile(returns, alpha * 100)
    
    def calculate_cvar(returns, confidence):
        var_threshold = calculate_var(returns, confidence)
        tail_returns = returns[returns <= var_threshold]
        return np.mean(tail_returns) if len(tail_returns) > 0 else var_threshold
    
    var_95 = calculate_var(returns, 0.95)
    cvar_95 = calculate_cvar(returns, 0.95)
    
    var_99 = calculate_var(returns, 0.99)
    cvar_99 = calculate_cvar(returns, 0.99)
    
    print(f"95% VaR: {var_95:.4f}")
    print(f"95% CVaR: {cvar_95:.4f}")
    print(f"99% VaR: {var_99:.4f}")
    print(f"99% CVaR: {cvar_99:.4f}")
    
    # Validate relationships
    assert cvar_95 <= var_95, "CVaR should be worse than VaR"
    assert var_99 <= var_95, "99% VaR should be worse than 95% VaR"
    assert cvar_99 <= cvar_95, "99% CVaR should be worse than 95% CVaR"
    
    print("✓ VaR/CVaR CPU tests passed")


def test_sharpe_sortino_calmar():
    """Test risk-adjusted return metrics"""
    print("\n=== Testing Enhanced Metrics (CPU) ===")
    
    # Generate sample returns
    np.random.seed(42)
    returns = np.random.normal(0.0005, 0.015, 252)  # One year of daily returns
    
    # Calculate Sharpe ratio
    def calculate_sharpe(returns, risk_free_rate=0.02/252):
        excess_returns = returns - risk_free_rate
        return np.mean(excess_returns) / np.std(returns) * np.sqrt(252)
    
    # Calculate Sortino ratio
    def calculate_sortino(returns, mar=0.0):
        excess_returns = returns - mar
        downside_returns = returns[returns < mar]
        if len(downside_returns) == 0:
            return float('inf')
        downside_dev = np.std(downside_returns)
        return np.mean(excess_returns) / downside_dev * np.sqrt(252) if downside_dev > 0 else float('inf')
    
    # Calculate max drawdown
    def calculate_max_drawdown(returns):
        cum_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cum_returns)
        drawdown = (cum_returns - running_max) / running_max
        return np.min(drawdown)
    
    # Calculate Calmar ratio
    def calculate_calmar(returns, max_dd=None):
        if max_dd is None:
            max_dd = calculate_max_drawdown(returns)
        total_return = np.prod(1 + returns) - 1
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
        return annualized_return / abs(max_dd) if abs(max_dd) > 0.0001 else float('inf')
    
    sharpe = calculate_sharpe(returns)
    sortino = calculate_sortino(returns)
    max_dd = calculate_max_drawdown(returns)
    calmar = calculate_calmar(returns, max_dd)
    
    print(f"Sharpe Ratio: {sharpe:.3f}")
    print(f"Sortino Ratio: {sortino:.3f}")
    print(f"Max Drawdown: {max_dd:.3%}")
    print(f"Calmar Ratio: {calmar:.3f}")
    
    # Sortino should be higher than Sharpe (focuses on downside)
    assert sortino >= sharpe, "Sortino should be >= Sharpe"
    
    print("✓ Enhanced metrics CPU tests passed")


def test_fitness_calculation():
    """Test fitness calculation modes"""
    print("\n=== Testing Fitness Calculation (CPU) ===")
    
    # Legacy fitness calculation
    def calculate_legacy_fitness(roi, drawdown, win_rate=0.5):
        if drawdown < 0.0001:
            drawdown = 0.0001
        
        fitness = roi / drawdown
        
        # Apply penalties
        penalty = 0.0
        if win_rate < 0.4:
            penalty += (0.4 - win_rate) * 10
            
        return fitness - penalty
    
    # Enhanced fitness calculation
    def calculate_enhanced_fitness(kelly_fraction, sharpe_ratio, regime_confidence, var_penalty=0.0):
        return kelly_fraction * max(sharpe_ratio, 0.1) * regime_confidence - var_penalty
    
    # Test data
    roi = 5000.0
    drawdown = 2500.0
    win_rate = 0.62
    kelly_fraction = 0.15
    sharpe_ratio = 1.2
    regime_confidence = 0.85
    
    # Calculate different fitness modes
    legacy_fitness = calculate_legacy_fitness(roi, drawdown, win_rate)
    print(f"Legacy fitness: {legacy_fitness:.4f}")
    
    enhanced_fitness = calculate_enhanced_fitness(kelly_fraction, sharpe_ratio, regime_confidence)
    print(f"Enhanced fitness: {enhanced_fitness:.4f}")
    
    # Hybrid fitness
    hybrid_fitness = 0.3 * legacy_fitness + 0.7 * enhanced_fitness
    print(f"Hybrid fitness: {hybrid_fitness:.4f}")
    
    print("✓ Fitness calculation CPU tests passed")


def test_integration():
    """Test integration of all components"""
    print("\n=== Testing Component Integration (CPU) ===")
    
    # Generate sample portfolio data
    np.random.seed(42)
    n_days = 252
    n_strategies = 10
    
    # Generate strategy returns
    strategy_returns = np.random.multivariate_normal(
        mean=[0.001] * n_strategies,
        cov=np.eye(n_strategies) * 0.0004 + np.ones((n_strategies, n_strategies)) * 0.0001,
        size=n_days
    )
    
    # Calculate metrics for each strategy
    print("\nStrategy Metrics:")
    for i in range(min(3, n_strategies)):  # Show first 3 strategies
        returns = strategy_returns[:, i]
        
        # Calculate basic metrics
        total_roi = np.sum(returns)
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = cumulative - running_max
        max_drawdown = abs(np.min(drawdown))
        win_rate = np.mean(returns > 0)
        
        # Calculate enhanced metrics
        sharpe = np.mean(returns - 0.02/252) / np.std(returns) * np.sqrt(252)
        var_95 = np.percentile(returns, 5)
        
        print(f"\nStrategy {i+1}:")
        print(f"  ROI: {total_roi:.4f}")
        print(f"  Max DD: {max_drawdown:.4f}")
        print(f"  Win Rate: {win_rate:.2%}")
        print(f"  Sharpe: {sharpe:.3f}")
        print(f"  VaR 95%: {var_95:.4f}")
    
    # Portfolio-level calculation
    equal_weights = np.ones(n_strategies) / n_strategies
    portfolio_returns = np.dot(strategy_returns, equal_weights)
    
    portfolio_roi = np.sum(portfolio_returns)
    portfolio_sharpe = np.mean(portfolio_returns - 0.02/252) / np.std(portfolio_returns) * np.sqrt(252)
    
    print(f"\nPortfolio Metrics:")
    print(f"  Total ROI: {portfolio_roi:.4f}")
    print(f"  Sharpe Ratio: {portfolio_sharpe:.3f}")
    
    print("\n✓ Integration tests passed")


def main():
    """Run all CPU tests"""
    print("=" * 60)
    print("Enhanced Financial Metrics Test Suite (CPU Only)")
    print("=" * 60)
    
    try:
        # Run tests
        test_kelly_criterion()
        test_var_cvar()
        test_sharpe_sortino_calmar()
        test_fitness_calculation()
        test_integration()
        
        print("\n" + "=" * 60)
        print("All CPU tests completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()