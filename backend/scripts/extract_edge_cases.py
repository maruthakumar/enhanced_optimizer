#!/usr/bin/env python3
"""
Extract edge case datasets from anonymized test data
Identifies and saves specific scenarios for targeted testing
"""

import pandas as pd
import numpy as np
import os
import sys
import json
from pathlib import Path


class EdgeCaseExtractor:
    """Extract edge cases from anonymized test data"""
    
    def __init__(self, data_path):
        """Initialize with test data"""
        self.data_path = data_path
        self.df = pd.read_csv(data_path)
        
        # Identify metadata vs strategy columns
        self.metadata_cols = ['Date', 'Day', 'start_time', 'end_time', 'market_regime', 
                             'Regime_Confidence_%', 'Market_regime_transition_threshold',
                             'capital', 'zone']
        self.strategy_cols = [col for col in self.df.columns 
                             if col not in self.metadata_cols and col.startswith('STRATEGY_')]
        
        print(f"Loaded data: {len(self.df)} rows, {len(self.strategy_cols)} strategies")
    
    def extract_zero_returns(self, output_path):
        """Extract days with zero returns across strategies"""
        print("Extracting zero return cases...")
        
        # Find rows where at least 10% of strategies have zero returns
        zero_counts = (self.df[self.strategy_cols] == 0).sum(axis=1)
        threshold = len(self.strategy_cols) * 0.1
        zero_return_mask = zero_counts >= threshold
        
        zero_return_df = self.df[zero_return_mask].copy()
        
        if len(zero_return_df) > 0:
            zero_return_df.to_csv(output_path, index=False)
            print(f"Saved {len(zero_return_df)} zero return days to {output_path}")
            
            # Save summary
            summary = {
                'total_days': len(zero_return_df),
                'dates': zero_return_df['Date'].tolist() if 'Date' in zero_return_df.columns else [],
                'avg_zero_strategies_per_day': float(zero_counts[zero_return_mask].mean()),
                'max_zero_strategies': int(zero_counts[zero_return_mask].max())
            }
            return summary
        else:
            print("No significant zero return days found")
            return None
    
    def extract_max_drawdown(self, output_path, percentile=5):
        """Extract periods with maximum drawdowns"""
        print(f"Extracting maximum drawdown cases (worst {percentile}%)...")
        
        # Calculate daily portfolio returns (sum of all strategies)
        daily_returns = self.df[self.strategy_cols].sum(axis=1)
        
        # Calculate cumulative returns and drawdown
        cumulative_returns = daily_returns.cumsum()
        running_max = cumulative_returns.expanding().max()
        drawdown = cumulative_returns - running_max
        
        # Find worst drawdown days
        threshold = np.percentile(drawdown, percentile)
        drawdown_mask = drawdown <= threshold
        
        drawdown_df = self.df[drawdown_mask].copy()
        drawdown_df['portfolio_drawdown'] = drawdown[drawdown_mask].values
        
        if len(drawdown_df) > 0:
            drawdown_df.to_csv(output_path, index=False)
            print(f"Saved {len(drawdown_df)} max drawdown days to {output_path}")
            
            # Save summary
            summary = {
                'total_days': len(drawdown_df),
                'worst_drawdown': float(drawdown[drawdown_mask].min()),
                'avg_drawdown': float(drawdown[drawdown_mask].mean()),
                'threshold_used': float(threshold)
            }
            return summary
        else:
            print("No significant drawdown days found")
            return None
    
    def extract_high_volatility(self, output_path, window=5):
        """Extract periods of high volatility"""
        print(f"Extracting high volatility periods (rolling {window} days)...")
        
        # Calculate rolling standard deviation of portfolio returns
        daily_returns = self.df[self.strategy_cols].sum(axis=1)
        rolling_std = daily_returns.rolling(window=window, min_periods=1).std()
        
        # Find top 10% volatile periods
        threshold = rolling_std.quantile(0.9)
        volatility_mask = rolling_std >= threshold
        
        volatility_df = self.df[volatility_mask].copy()
        volatility_df['portfolio_volatility'] = rolling_std[volatility_mask].values
        
        if len(volatility_df) > 0:
            volatility_df.to_csv(output_path, index=False)
            print(f"Saved {len(volatility_df)} high volatility days to {output_path}")
            
            # Save summary
            summary = {
                'total_days': len(volatility_df),
                'max_volatility': float(rolling_std[volatility_mask].max()),
                'avg_volatility': float(rolling_std[volatility_mask].mean()),
                'threshold_used': float(threshold)
            }
            return summary
        else:
            print("No high volatility periods found")
            return None
    
    def extract_correlation_extremes(self, output_path, sample_size=10):
        """Extract strategies with extreme correlations"""
        print("Calculating strategy correlations...")
        
        # For large datasets, sample strategies to make correlation calculation feasible
        if len(self.strategy_cols) > 100:
            # Sample strategies evenly across the range
            sample_indices = np.linspace(0, len(self.strategy_cols)-1, sample_size, dtype=int)
            sampled_strategies = [self.strategy_cols[i] for i in sample_indices]
        else:
            sampled_strategies = self.strategy_cols
        
        # Calculate correlation matrix
        corr_matrix = self.df[sampled_strategies].corr()
        
        # Find highly correlated pairs (excluding diagonal)
        high_corr_pairs = []
        for i in range(len(sampled_strategies)):
            for j in range(i+1, len(sampled_strategies)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) > 0.8:  # High correlation threshold
                    high_corr_pairs.append({
                        'strategy_1': sampled_strategies[i],
                        'strategy_2': sampled_strategies[j],
                        'correlation': corr_value
                    })
        
        # Create dataset with highly correlated strategies
        if high_corr_pairs:
            # Get unique strategies from high correlation pairs
            unique_strategies = list(set(
                [pair['strategy_1'] for pair in high_corr_pairs] + 
                [pair['strategy_2'] for pair in high_corr_pairs]
            ))
            
            # Save data for these strategies
            corr_df = self.df[['Date', 'Day'] + unique_strategies].copy()
            corr_df.to_csv(output_path, index=False)
            print(f"Saved {len(unique_strategies)} highly correlated strategies to {output_path}")
            
            # Save correlation summary
            summary = {
                'high_correlation_pairs': len(high_corr_pairs),
                'unique_strategies': len(unique_strategies),
                'sample_size_used': len(sampled_strategies),
                'correlation_pairs': high_corr_pairs[:10]  # Save top 10 pairs
            }
            return summary
        else:
            print("No highly correlated strategies found")
            return None
    
    def extract_winning_streaks(self, output_path, min_streak=5):
        """Extract periods with consecutive winning days"""
        print(f"Extracting winning streaks (minimum {min_streak} days)...")
        
        # Calculate daily portfolio returns
        daily_returns = self.df[self.strategy_cols].sum(axis=1)
        
        # Find winning days
        winning_days = daily_returns > 0
        
        # Find streaks
        streaks = []
        current_streak_start = None
        current_streak_length = 0
        
        for i, (is_winner, date) in enumerate(zip(winning_days, self.df.get('Date', range(len(self.df))))):
            if is_winner:
                if current_streak_start is None:
                    current_streak_start = i
                current_streak_length += 1
            else:
                if current_streak_length >= min_streak:
                    streaks.append({
                        'start_idx': current_streak_start,
                        'end_idx': i - 1,
                        'length': current_streak_length,
                        'total_return': float(daily_returns.iloc[current_streak_start:i].sum())
                    })
                current_streak_start = None
                current_streak_length = 0
        
        # Check final streak
        if current_streak_length >= min_streak:
            streaks.append({
                'start_idx': current_streak_start,
                'end_idx': len(self.df) - 1,
                'length': current_streak_length,
                'total_return': float(daily_returns.iloc[current_streak_start:].sum())
            })
        
        # Extract data for all streak periods
        if streaks:
            streak_indices = []
            for streak in streaks:
                streak_indices.extend(range(streak['start_idx'], streak['end_idx'] + 1))
            
            streak_df = self.df.iloc[streak_indices].copy()
            streak_df.to_csv(output_path, index=False)
            print(f"Saved {len(streak_df)} winning streak days to {output_path}")
            
            # Save summary
            summary = {
                'total_streaks': len(streaks),
                'total_days': len(streak_indices),
                'longest_streak': max(s['length'] for s in streaks),
                'best_streak_return': max(s['total_return'] for s in streaks),
                'streaks': streaks[:5]  # Save top 5 streaks
            }
            return summary
        else:
            print(f"No winning streaks of {min_streak}+ days found")
            return None
    
    def extract_all_edge_cases(self, output_dir):
        """Extract all edge case datasets"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        summaries = {}
        
        # Extract each type of edge case
        edge_cases = [
            ('zero_returns', self.extract_zero_returns),
            ('max_drawdown', self.extract_max_drawdown),
            ('high_volatility', self.extract_high_volatility),
            ('high_correlation', self.extract_correlation_extremes),
            ('winning_streaks', self.extract_winning_streaks)
        ]
        
        for name, extractor_func in edge_cases:
            output_path = os.path.join(output_dir, f'{name}.csv')
            try:
                summary = extractor_func(output_path)
                if summary:
                    summaries[name] = summary
            except Exception as e:
                print(f"Error extracting {name}: {e}")
        
        # Save combined summary
        summary_path = os.path.join(output_dir, 'edge_cases_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summaries, f, indent=2)
        
        print(f"\nEdge case extraction complete!")
        print(f"Summary saved to: {summary_path}")
        
        return summaries


def main():
    if len(sys.argv) < 3:
        print("Usage: python extract_edge_cases.py input_data.csv output_directory")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_dir = sys.argv[2]
    
    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        sys.exit(1)
    
    extractor = EdgeCaseExtractor(input_file)
    summaries = extractor.extract_all_edge_cases(output_dir)
    
    print("\nExtraction Summary:")
    for case_type, summary in summaries.items():
        print(f"- {case_type}: {summary.get('total_days', 'N/A')} days")


if __name__ == "__main__":
    main()