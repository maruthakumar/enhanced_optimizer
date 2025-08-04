#!/usr/bin/env python3
"""
Demonstration of Enhanced Financial Metrics Usage
Shows how to integrate Kelly Criterion, VaR/CVaR, Sharpe/Sortino ratios, 
and market regime optimization into the Heavy Optimizer workflow.
"""
import sys
import os
import numpy as np
import pandas as pd
import logging
from datetime import datetime

# Add paths for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(__file__), 'lib'))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def demonstrate_enhanced_workflow():
    """Demonstrate how to use enhanced metrics in a typical workflow"""
    print("=" * 80)
    print("Enhanced Financial Metrics Demonstration")
    print("=" * 80)
    
    # 1. Generate sample strategy data (simulating loaded CSV data)
    print("\n1. Loading Strategy Data...")
    np.random.seed(42)
    n_days = 82  # Production data has 82 trading days
    n_strategies = 20  # Subset of 25,544 strategies
    
    # Generate realistic strategy returns
    strategy_returns = np.random.multivariate_normal(
        mean=[0.0008] * n_strategies,  # Slight positive bias
        cov=np.eye(n_strategies) * 0.0004 + np.ones((n_strategies, n_strategies)) * 0.00005,
        size=n_days
    )
    
    # Create DataFrame
    dates = pd.date_range(start='2025-01-01', periods=n_days, freq='B')
    strategy_names = [f'SENSEX_Strategy_{i:04d}' for i in range(n_strategies)]
    
    df = pd.DataFrame(strategy_returns, index=dates, columns=strategy_names)
    print(f"Loaded {n_strategies} strategies with {n_days} days of data")
    
    # 2. Calculate basic metrics for each strategy
    print("\n2. Calculating Basic Metrics...")
    strategy_metrics = {}
    
    for strategy in strategy_names:
        returns = df[strategy].values
        
        # Basic metrics
        total_roi = np.sum(returns)
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = cumulative - running_max
        max_drawdown = abs(np.min(drawdown)) if len(drawdown) > 0 else 0.01
        
        # Win/loss statistics
        winning_returns = returns[returns > 0]
        losing_returns = returns[returns < 0]
        
        win_rate = len(winning_returns) / len(returns) if len(returns) > 0 else 0
        avg_win = np.mean(winning_returns) if len(winning_returns) > 0 else 0
        avg_loss = abs(np.mean(losing_returns)) if len(losing_returns) > 0 else 0.01
        
        # Profit factor
        total_wins = np.sum(winning_returns) if len(winning_returns) > 0 else 0
        total_losses = abs(np.sum(losing_returns)) if len(losing_returns) > 0 else 0.01
        profit_factor = total_wins / total_losses
        
        strategy_metrics[strategy] = {
            'total_roi': total_roi,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'returns': returns
        }
    
    # 3. Add market regime data (simulated)
    print("\n3. Adding Market Regime Data...")
    regimes = ['bull', 'bear', 'neutral', 'volatile']
    for strategy in strategy_names:
        # Simulate regime assignment based on performance
        if strategy_metrics[strategy]['total_roi'] > 0.05:
            regime = 'bull'
            confidence = 75 + np.random.randint(10, 20)
        elif strategy_metrics[strategy]['total_roi'] < -0.02:
            regime = 'bear'
            confidence = 70 + np.random.randint(5, 15)
        elif strategy_metrics[strategy]['profit_factor'] < 1.1:
            regime = 'volatile'
            confidence = 60 + np.random.randint(10, 25)
        else:
            regime = 'neutral'
            confidence = 65 + np.random.randint(10, 20)
            
        strategy_metrics[strategy]['market_regime'] = regime
        strategy_metrics[strategy]['regime_confidence'] = confidence
    
    # 4. Calculate Enhanced Metrics
    print("\n4. Calculating Enhanced Financial Metrics...")
    
    # Define calculation functions inline
    def calc_kelly(win_rate, avg_win, avg_loss, max_size=0.25, min_size=0.01):
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
    
    def calc_var(returns, confidence):
        alpha = 1 - confidence
        return np.percentile(returns, alpha * 100)
    
    def calc_cvar(returns, confidence):
        var_threshold = calc_var(returns, confidence)
        tail_returns = returns[returns <= var_threshold]
        return np.mean(tail_returns) if len(tail_returns) > 0 else var_threshold
    
    def calc_sharpe(returns, risk_free_rate=0.02/252):
        excess_returns = returns - risk_free_rate
        return np.mean(excess_returns) / np.std(returns) * np.sqrt(252)
    
    def calc_sortino(returns, mar=0.0):
        excess_returns = returns - mar
        downside_returns = returns[returns < mar]
        if len(downside_returns) == 0:
            return float('inf')
        downside_dev = np.std(downside_returns)
        return np.mean(excess_returns) / downside_dev * np.sqrt(252) if downside_dev > 0 else float('inf')
    
    def calc_max_dd(returns):
        cum_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cum_returns)
        drawdown = (cum_returns - running_max) / running_max
        return np.min(drawdown)
    
    def calc_calmar(returns, max_dd=None):
        if max_dd is None:
            max_dd = calc_max_dd(returns)
        total_return = np.prod(1 + returns) - 1
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
        return annualized_return / abs(max_dd) if abs(max_dd) > 0.0001 else float('inf')
    
    # Calculate enhanced metrics for each strategy
    for strategy, metrics in strategy_metrics.items():
        returns = metrics['returns']
        
        # Kelly Criterion
        kelly_fraction = calc_kelly(
            metrics['win_rate'],
            metrics['avg_win'] * 1000,  # Scale for realistic values
            metrics['avg_loss'] * 1000
        )
        metrics['kelly_fraction'] = kelly_fraction
        
        # VaR and CVaR
        metrics['var_95'] = calc_var(returns, 0.95)
        metrics['cvar_95'] = calc_cvar(returns, 0.95)
        metrics['var_99'] = calc_var(returns, 0.99)
        metrics['cvar_99'] = calc_cvar(returns, 0.99)
        
        # Risk-adjusted returns
        metrics['sharpe_ratio'] = calc_sharpe(returns)
        metrics['sortino_ratio'] = calc_sortino(returns)
        metrics['calmar_ratio'] = calc_calmar(returns)
    
    # 5. Calculate Fitness Scores
    print("\n5. Calculating Fitness Scores...")
    
    # Legacy fitness
    for strategy, metrics in strategy_metrics.items():
        # Legacy formula: ROI/Drawdown
        legacy_fitness = metrics['total_roi'] / metrics['max_drawdown']
        
        # Apply penalties
        if metrics['win_rate'] < 0.4:
            legacy_fitness -= (0.4 - metrics['win_rate']) * 10
            
        metrics['legacy_fitness'] = legacy_fitness
        
    # Enhanced fitness
    for strategy, metrics in strategy_metrics.items():
        # Enhanced formula: kelly * sharpe * regime_confidence - var_penalty
        kelly = metrics['kelly_fraction']
        sharpe = max(metrics['sharpe_ratio'], 0.1)
        regime_conf = metrics['regime_confidence'] / 100.0
        
        # VaR penalty
        var_limit = 0.025
        var_penalty = max(0, (abs(metrics['var_95']) - var_limit) * 100)
        
        enhanced_fitness = kelly * sharpe * regime_conf - var_penalty
        metrics['enhanced_fitness'] = enhanced_fitness
        
    # Hybrid fitness
    for strategy, metrics in strategy_metrics.items():
        metrics['hybrid_fitness'] = 0.3 * metrics['legacy_fitness'] + 0.7 * metrics['enhanced_fitness']
    
    # 6. Portfolio Selection
    print("\n6. Selecting Optimal Portfolio...")
    
    # Sort by different fitness modes
    strategies_list = list(strategy_metrics.keys())
    
    # Legacy mode selection
    legacy_sorted = sorted(strategies_list, 
                          key=lambda s: strategy_metrics[s]['legacy_fitness'], 
                          reverse=True)
    
    # Enhanced mode selection
    enhanced_sorted = sorted(strategies_list,
                           key=lambda s: strategy_metrics[s]['enhanced_fitness'],
                           reverse=True)
    
    # Hybrid mode selection
    hybrid_sorted = sorted(strategies_list,
                         key=lambda s: strategy_metrics[s]['hybrid_fitness'],
                         reverse=True)
    
    portfolio_size = 10
    
    print(f"\nTop {portfolio_size} strategies by fitness mode:")
    print("\nLegacy Mode:")
    for i, strategy in enumerate(legacy_sorted[:portfolio_size]):
        m = strategy_metrics[strategy]
        print(f"  {i+1}. {strategy}: Fitness={m['legacy_fitness']:.3f}, "
              f"ROI={m['total_roi']:.3f}, DD={m['max_drawdown']:.3f}")
    
    print("\nEnhanced Mode:")
    for i, strategy in enumerate(enhanced_sorted[:portfolio_size]):
        m = strategy_metrics[strategy]
        print(f"  {i+1}. {strategy}: Fitness={m['enhanced_fitness']:.3f}, "
              f"Kelly={m['kelly_fraction']:.3f}, Sharpe={m['sharpe_ratio']:.2f}, "
              f"Regime={m['market_regime']}({m['regime_confidence']}%)")
    
    # 7. Portfolio Analysis
    print("\n7. Portfolio Performance Analysis...")
    
    # Calculate portfolio metrics for each mode
    modes = {
        'Legacy': legacy_sorted[:portfolio_size],
        'Enhanced': enhanced_sorted[:portfolio_size],
        'Hybrid': hybrid_sorted[:portfolio_size]
    }
    
    for mode_name, portfolio in modes.items():
        # Equal weight portfolio returns
        portfolio_returns = np.mean([df[s].values for s in portfolio], axis=0)
        
        # Portfolio metrics
        total_return = np.sum(portfolio_returns)
        portfolio_sharpe = calc_sharpe(portfolio_returns)
        portfolio_dd = abs(calc_max_dd(portfolio_returns))
        portfolio_var = calc_var(portfolio_returns, 0.95)
        
        print(f"\n{mode_name} Portfolio:")
        print(f"  Total Return: {total_return:.3f}")
        print(f"  Sharpe Ratio: {portfolio_sharpe:.2f}")
        print(f"  Max Drawdown: {portfolio_dd:.3%}")
        print(f"  95% VaR: {portfolio_var:.4f}")
        
        # Regime composition
        regime_counts = {}
        for strategy in portfolio:
            regime = strategy_metrics[strategy]['market_regime']
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
        
        print(f"  Regime Mix: {regime_counts}")
    
    # 8. Position Sizing with Kelly Criterion
    print("\n8. Kelly Criterion Position Sizing...")
    
    enhanced_portfolio = enhanced_sorted[:portfolio_size]
    total_capital = 1.0
    
    # Calculate Kelly positions
    kelly_positions = {}
    kelly_total = 0
    
    for strategy in enhanced_portfolio:
        kelly_frac = strategy_metrics[strategy]['kelly_fraction']
        kelly_positions[strategy] = kelly_frac
        kelly_total += kelly_frac
    
    # Normalize if total exceeds capital
    if kelly_total > total_capital:
        print(f"Kelly total ({kelly_total:.2f}) exceeds capital, normalizing...")
        for strategy in kelly_positions:
            kelly_positions[strategy] = kelly_positions[strategy] / kelly_total * total_capital
    
    print("\nKelly Position Sizes:")
    for strategy, position in kelly_positions.items():
        print(f"  {strategy}: {position:.1%}")
    
    # 9. Risk Analysis
    print("\n9. Portfolio Risk Analysis...")
    
    # Calculate portfolio VaR with Kelly weights
    kelly_weights = np.array([kelly_positions[s] for s in enhanced_portfolio])
    portfolio_returns_kelly = np.sum([df[s].values * w for s, w in zip(enhanced_portfolio, kelly_weights)], axis=0)
    
    print("\nRisk Metrics (Kelly-weighted portfolio):")
    print(f"  95% VaR: {calc_var(portfolio_returns_kelly, 0.95):.4f}")
    print(f"  95% CVaR: {calc_cvar(portfolio_returns_kelly, 0.95):.4f}")
    print(f"  99% VaR: {calc_var(portfolio_returns_kelly, 0.99):.4f}")
    print(f"  99% CVaR: {calc_cvar(portfolio_returns_kelly, 0.99):.4f}")
    
    # 10. Summary
    print("\n" + "=" * 80)
    print("Enhanced Metrics Demonstration Complete")
    print("=" * 80)
    print("\nKey Insights:")
    print("1. Enhanced fitness considers risk-adjusted returns and market regimes")
    print("2. Kelly Criterion provides optimal position sizing based on win/loss statistics")
    print("3. VaR/CVaR metrics quantify downside risk at confidence levels")
    print("4. Sharpe/Sortino ratios measure risk-adjusted performance")
    print("5. Market regime confidence helps filter strategies by market conditions")
    
    print("\nTo enable enhanced metrics in production:")
    print("1. Set FITNESS_CALCULATION.mode = 'enhanced' in production_config.ini")
    print("2. Enable KELLY_CRITERION.enabled = true for position sizing")
    print("3. Enable MARKET_REGIME_CONFIG.confidence_weighting = true")
    print("4. Run workflow with enhanced configuration")


if __name__ == "__main__":
    demonstrate_enhanced_workflow()