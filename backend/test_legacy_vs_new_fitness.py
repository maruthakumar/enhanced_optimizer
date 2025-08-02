#!/usr/bin/env python3
"""Test fitness calculation parity between legacy and new systems"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

def calculate_new_system_fitness(data_file, portfolio_strategies):
    """Calculate fitness using new system logic"""
    # Load data
    data = pd.read_csv(data_file)
    
    # Get strategy columns (all columns that start with SENSEX)
    strategy_columns = [col for col in data.columns if col.startswith('SENSEX')]
    
    # Filter to portfolio strategies
    portfolio_columns = []
    for strategy in portfolio_strategies:
        for col in strategy_columns:
            if col == strategy:
                portfolio_columns.append(col)
                break
    
    # Calculate portfolio metrics
    portfolio_data = data[portfolio_columns]
    portfolio_returns = portfolio_data.sum(axis=1)
    
    # ROI - total returns
    roi = portfolio_returns.sum()
    
    # Drawdown calculation
    cumulative_returns = portfolio_returns.cumsum()
    running_max = cumulative_returns.cummax()
    drawdown = cumulative_returns - running_max
    max_drawdown = abs(drawdown.min())
    
    # Win rate
    profitable_days = (portfolio_returns > 0).sum()
    total_days = len(portfolio_returns)
    win_rate = profitable_days / total_days
    
    # Profit factor
    gains = portfolio_returns[portfolio_returns > 0].sum()
    losses = abs(portfolio_returns[portfolio_returns < 0].sum())
    profit_factor = gains / losses if losses > 0 else gains
    
    # Legacy fitness formula
    fitness = roi / max_drawdown if max_drawdown > 0 else 0
    
    return {
        'roi': roi,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'fitness': fitness
    }

def test_fitness_parity():
    """Test fitness calculation parity with known legacy results"""
    print("Testing Fitness Calculation Parity")
    print("=" * 60)
    
    # Known legacy result for size 37
    legacy_fitness = 30.45764862187442
    legacy_algorithm = 'SA'
    
    # Load legacy output to get selected strategies
    legacy_output = "/mnt/optimizer_share/zone_optimization_25_06_25/Output/run_20250726_163251/best_portfolio_size37_20250726_163251.txt"
    
    with open(legacy_output, 'r') as f:
        content = f.read()
    
    # Extract strategies from legacy output
    strategies = []
    lines = content.split('\n')
    for line in lines:
        # Look for numbered strategies like "1. Python_Multi_..."
        if line.strip() and line[0:1].isdigit() and '. ' in line:
            # Extract the strategy name after the number
            parts = line.split('. ', 1)
            if len(parts) == 2:
                strategy = parts[1].strip()
                # Convert to PNL_ format to match CSV columns
                # Extract the key parts: timeframe and stops
                if 'SENSEX' in strategy:
                    # Parse: Python_Multi_Consolidated_20250726_161921_SENSEX 1426-1456 SL73%TP32%
                    strategy_parts = strategy.split('_SENSEX ')
                    if len(strategy_parts) == 2:
                        time_and_stops = strategy_parts[1]
                        # Create column name format: SENSEX 1426-1456 SL73%TP32%
                        col_name = f"SENSEX {time_and_stops}"
                        strategies.append(col_name)
    
    print(f"Found {len(strategies)} strategies in legacy portfolio")
    
    # Calculate fitness with new system
    data_file = "/mnt/optimizer_share/input/Python_Multi_Consolidated_20250726_161921.csv"
    
    if Path(data_file).exists():
        new_metrics = calculate_new_system_fitness(data_file, strategies)
        
        print(f"\nLegacy System (Size 37, {legacy_algorithm}):")
        print(f"  Fitness: {legacy_fitness}")
        
        print(f"\nNew System Calculation:")
        print(f"  ROI: {new_metrics['roi']:,.2f}")
        print(f"  Max Drawdown: {new_metrics['max_drawdown']:,.2f}")
        print(f"  Win Rate: {new_metrics['win_rate']:.2%}")
        print(f"  Profit Factor: {new_metrics['profit_factor']:.3f}")
        print(f"  Fitness: {new_metrics['fitness']:.8f}")
        
        # Check parity
        difference = abs(legacy_fitness - new_metrics['fitness'])
        relative_diff = difference / legacy_fitness
        
        print(f"\nParity Check:")
        print(f"  Absolute Difference: {difference:.8f}")
        print(f"  Relative Difference: {relative_diff:.4%}")
        
        # Tolerance of 0.01%
        if relative_diff <= 0.0001:
            print("  ✓ FITNESS PARITY ACHIEVED!")
        else:
            print("  ✗ FITNESS MISMATCH - Outside tolerance")
            
            # Debug: Check if it's a scaling issue
            scaling_factor = legacy_fitness / new_metrics['fitness']
            print(f"\n  Debug - Scaling factor: {scaling_factor:.6f}")
            
    else:
        print(f"✗ Data file not found: {data_file}")

if __name__ == "__main__":
    test_fitness_parity()