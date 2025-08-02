#!/usr/bin/env python3
"""
Debug script to understand fitness calculation discrepancy
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from csv_only_heavydb_workflow import CSVOnlyHeavyDBOptimizer
import pandas as pd
import numpy as np

def debug_fitness():
    """Debug the fitness calculation to understand the discrepancy"""
    print("=" * 80)
    print("DEBUGGING FITNESS CALCULATION")
    print("=" * 80)
    
    # Load production data
    input_file = "../input/Python_Multi_Consolidated_20250726_161921.csv"
    df = pd.read_csv(input_file)
    
    print(f"\n1. Data shape: {df.shape}")
    print(f"   Columns: {list(df.columns[:5])} ... (showing first 5)")
    
    # Create optimizer instance
    optimizer = CSVOnlyHeavyDBOptimizer()
    
    # Get numeric columns
    numeric_columns = df.iloc[:, 1:].select_dtypes(include=[np.number]).columns.tolist()
    print(f"\n2. Numeric columns: {len(numeric_columns)}")
    
    # Test with a sample portfolio
    sample_portfolio_indices = list(range(37))  # First 37 strategies
    sample_columns = [numeric_columns[i] for i in sample_portfolio_indices]
    
    print(f"\n3. Testing fitness calculation with first 37 strategies...")
    
    # Create a custom fitness calculation that prints intermediate values
    portfolio_data = df[['Date'] + sample_columns].copy()
    portfolio_data['Date'] = pd.to_datetime(portfolio_data['Date'])
    portfolio_data.set_index('Date', inplace=True)
    
    # Calculate portfolio performance
    portfolio_returns = portfolio_data[sample_columns].sum(axis=1)
    
    print(f"\n   Portfolio returns shape: {portfolio_returns.shape}")
    print(f"   First 5 returns: {portfolio_returns.head().values}")
    print(f"   Total sum of returns: {portfolio_returns.sum():.2f}")
    
    # ROI calculation
    initial_value = 100000  # Standard initial capital
    final_value = initial_value + portfolio_returns.sum()
    roi = (final_value - initial_value) / initial_value * 100
    
    print(f"\n   Initial value: ${initial_value:,.2f}")
    print(f"   Final value: ${final_value:,.2f}")
    print(f"   ROI: {roi:.2f}%")
    
    # Drawdown calculation
    cumulative_returns = portfolio_returns.cumsum()
    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - running_max)
    max_drawdown = abs(drawdown.min()) if len(drawdown) > 0 else 0.01
    
    print(f"\n   Cumulative returns range: [{cumulative_returns.min():.2f}, {cumulative_returns.max():.2f}]")
    print(f"   Max drawdown: {max_drawdown:.2f}")
    
    # Win rate
    winning_days = (portfolio_returns > 0).sum()
    total_days = len(portfolio_returns)
    win_rate = winning_days / total_days if total_days > 0 else 0
    
    print(f"\n   Winning days: {winning_days}/{total_days}")
    print(f"   Win rate: {win_rate:.2%}")
    
    # Profit factor
    gains = portfolio_returns[portfolio_returns > 0].sum()
    losses = abs(portfolio_returns[portfolio_returns < 0].sum())
    profit_factor = gains / losses if losses > 0 else gains
    
    print(f"\n   Total gains: {gains:.2f}")
    print(f"   Total losses: {losses:.2f}")
    print(f"   Profit factor: {profit_factor:.2f}")
    
    # Legacy fitness calculation
    if max_drawdown > 0:
        fitness = roi / max_drawdown
    else:
        fitness = roi * 100
    
    print(f"\n4. FITNESS CALCULATION:")
    print(f"   Formula: ROI / Max Drawdown")
    print(f"   Calculation: {roi:.2f} / {max_drawdown:.2f}")
    print(f"   Result: {fitness:.6f}")
    
    # Compare with legacy values
    print(f"\n5. LEGACY COMPARISON:")
    print(f"   Legacy ROI: 13,653.37")
    print(f"   Legacy Max DD: 443.79")
    print(f"   Legacy Fitness: 30.458 (13653.37 / 443.79)")
    
    print(f"\n6. ANALYSIS:")
    print(f"   Our ROI as raw value: {portfolio_returns.sum():.2f}")
    print(f"   Our ROI as percentage: {roi:.2f}%")
    print(f"   \n   The discrepancy appears to be:")
    print(f"   - Legacy uses ROI as raw currency value (13,653.37)")
    print(f"   - We calculate ROI as percentage ({roi:.2f}%)")
    print(f"   - To match legacy: {portfolio_returns.sum():.2f} / {max_drawdown:.2f} = {portfolio_returns.sum() / max_drawdown:.6f}")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    debug_fitness()