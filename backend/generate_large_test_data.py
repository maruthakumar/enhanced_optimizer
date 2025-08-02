#!/usr/bin/env python3
"""
Generate large synthetic dataset for testing correlation optimization
Simulates the 25,544 strategies x 82 trading days production dataset
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import argparse

def generate_synthetic_trading_data(n_strategies: int = 25544, 
                                  n_days: int = 82,
                                  output_file: str = None):
    """
    Generate synthetic trading data similar to production dataset
    
    Args:
        n_strategies: Number of strategies (default 25,544)
        n_days: Number of trading days (default 82)
        output_file: Output CSV file path
    """
    print(f"🔄 Generating synthetic dataset: {n_strategies} strategies x {n_days} days")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate date range (similar to production: 2024-01-04 to 2024-07-26)
    start_date = datetime(2024, 1, 4)
    dates = []
    current_date = start_date
    
    # Generate trading days (skip weekends)
    while len(dates) < n_days:
        if current_date.weekday() < 5:  # Monday = 0, Friday = 4
            dates.append(current_date)
        current_date += timedelta(days=1)
    
    # Generate strategy names
    strategy_names = [f'SENSEX_{i:05d}' for i in range(n_strategies)]
    
    # Generate correlated returns data
    print("📊 Generating correlated returns...")
    
    # Create correlation structure
    # - Some strategies are highly correlated (sector groups)
    # - Some are negatively correlated (hedges)
    # - Most have low correlation
    
    # Create base factors
    n_factors = 50  # Market factors
    factor_returns = np.random.randn(n_days, n_factors) * 0.01  # 1% daily vol
    
    # Generate strategy loadings on factors
    factor_loadings = np.random.randn(n_strategies, n_factors) * 0.3
    
    # Add some structure - create sector groups
    n_sectors = 10
    strategies_per_sector = n_strategies // n_sectors
    
    for sector in range(n_sectors):
        start_idx = sector * strategies_per_sector
        end_idx = min((sector + 1) * strategies_per_sector, n_strategies)
        
        # Strategies in same sector load on similar factors
        sector_loading = np.random.randn(n_factors) * 0.5
        factor_loadings[start_idx:end_idx] += sector_loading
    
    # Generate returns
    strategy_returns = factor_returns @ factor_loadings.T
    
    # Add idiosyncratic risk
    idio_returns = np.random.randn(n_days, n_strategies) * 0.005  # 0.5% idio vol
    strategy_returns += idio_returns
    
    # Convert to P&L (cumulative returns with initial capital)
    initial_capital = 1000000  # $1M per strategy
    strategy_pnl = np.zeros_like(strategy_returns)
    
    for i in range(n_strategies):
        cumulative = (1 + strategy_returns[:, i]).cumprod()
        strategy_pnl[:, i] = initial_capital * (cumulative - 1)
    
    # Add some realistic features
    # - Transaction costs
    strategy_pnl *= 0.98  # 2% transaction cost impact
    
    # - Some strategies fail (negative drift)
    failing_strategies = np.random.choice(n_strategies, size=n_strategies//20, replace=False)
    strategy_pnl[:, failing_strategies] -= np.arange(n_days).reshape(-1, 1) * 1000
    
    # Create DataFrame
    print("📝 Creating DataFrame...")
    df = pd.DataFrame(strategy_pnl, columns=strategy_names)
    df.insert(0, 'Date', dates)
    
    # Add some additional columns similar to production data
    df['Day'] = df['Date'].dt.day_name()
    
    # Calculate some statistics
    total_pnl = strategy_pnl.sum()
    mean_daily_pnl = strategy_pnl.mean()
    max_drawdown = np.min([
        (strategy_pnl[:, i].cumsum() - 
         pd.Series(strategy_pnl[:, i].cumsum()).expanding().max()).min()
        for i in range(min(100, n_strategies))
    ])
    
    print(f"\n📊 Dataset Statistics:")
    print(f"  - Total strategies: {n_strategies:,}")
    print(f"  - Trading days: {n_days}")
    print(f"  - Date range: {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")
    print(f"  - Total P&L: ${total_pnl:,.2f}")
    print(f"  - Mean daily P&L: ${mean_daily_pnl:,.2f}")
    print(f"  - Sample max drawdown: ${max_drawdown:,.2f}")
    print(f"  - Data shape: {df.shape}")
    print(f"  - Memory usage: {df.memory_usage(deep=True).sum() / (1024**2):.1f} MB")
    
    # Save to file
    if output_file is None:
        output_file = f"/mnt/optimizer_share/input/synthetic_data_{n_strategies}x{n_days}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    print(f"\n💾 Saving to {output_file}...")
    df.to_csv(output_file, index=False)
    
    file_size_mb = os.path.getsize(output_file) / (1024**2)
    print(f"✅ File saved: {file_size_mb:.1f} MB")
    
    # Quick correlation check
    print("\n🔍 Sample correlation matrix (first 10 strategies):")
    sample_corr = df[strategy_names[:10]].corr()
    print(f"  - Mean correlation: {sample_corr.values[np.triu_indices(10, k=1)].mean():.3f}")
    print(f"  - Max correlation: {sample_corr.values[np.triu_indices(10, k=1)].max():.3f}")
    print(f"  - Min correlation: {sample_corr.values[np.triu_indices(10, k=1)].min():.3f}")
    
    return output_file

def main():
    parser = argparse.ArgumentParser(description='Generate synthetic trading data for testing')
    parser.add_argument('--strategies', type=int, default=25544,
                       help='Number of strategies (default: 25544)')
    parser.add_argument('--days', type=int, default=82,
                       help='Number of trading days (default: 82)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file path')
    parser.add_argument('--small', action='store_true',
                       help='Generate small test dataset (1000x82)')
    parser.add_argument('--medium', action='store_true',
                       help='Generate medium test dataset (5000x82)')
    
    args = parser.parse_args()
    
    if args.small:
        n_strategies = 1000
        print("📊 Generating SMALL test dataset")
    elif args.medium:
        n_strategies = 5000
        print("📊 Generating MEDIUM test dataset")
    else:
        n_strategies = args.strategies
        print("📊 Generating LARGE production-size dataset")
    
    output_file = generate_synthetic_trading_data(
        n_strategies=n_strategies,
        n_days=args.days,
        output_file=args.output
    )
    
    print(f"\n✅ Dataset generated successfully!")
    print(f"📍 Location: {output_file}")
    print("\n🚀 You can now test with:")
    print(f"   python3 csv_only_heavydb_workflow.py --input {output_file} --portfolio-size 35")

if __name__ == "__main__":
    main()