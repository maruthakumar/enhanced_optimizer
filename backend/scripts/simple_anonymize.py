#!/usr/bin/env python3
"""
Simple anonymization script for Heavy Optimizer test data
Focuses on column renaming without correlation validation for large datasets
"""

import pandas as pd
import sys
from datetime import datetime, timedelta
import json

def anonymize_csv(input_file, output_file, date_shift=180):
    """Simple anonymization - rename columns and shift dates"""
    
    print(f"Reading {input_file}...")
    df = pd.read_csv(input_file)
    
    # Get metadata columns
    metadata_cols = ['Date', 'Day', 'start_time', 'end_time', 'market_regime', 
                    'Regime_Confidence_%', 'Market_regime_transition_threshold',
                    'capital', 'zone']
    
    # Create strategy mapping
    strategy_mapping = {}
    counter = 1
    
    new_columns = {}
    for col in df.columns:
        if col in metadata_cols:
            new_columns[col] = col
        else:
            new_name = f"STRATEGY_{counter:04d}"
            new_columns[col] = new_name
            strategy_mapping[col] = new_name
            counter += 1
    
    # Rename columns
    df_anon = df.rename(columns=new_columns)
    
    # Shift dates if Date column exists
    if 'Date' in df_anon.columns:
        try:
            df_anon['Date'] = pd.to_datetime(df_anon['Date'])
            df_anon['Date'] = df_anon['Date'] - timedelta(days=date_shift)
            df_anon['Date'] = df_anon['Date'].dt.strftime('%Y-%m-%d')
        except:
            print("Warning: Could not shift dates")
    
    # Save anonymized data
    print(f"Saving to {output_file}...")
    df_anon.to_csv(output_file, index=False)
    
    # Save mapping
    mapping_file = output_file.replace('.csv', '_mapping.json')
    with open(mapping_file, 'w') as f:
        json.dump({
            'strategy_mapping': strategy_mapping,
            'date_shift': date_shift,
            'total_strategies': len(strategy_mapping)
        }, f, indent=2)
    
    print(f"Done! Anonymized {len(strategy_mapping)} strategies")
    print(f"Mapping saved to {mapping_file}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python simple_anonymize.py input.csv output.csv [date_shift]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    date_shift = int(sys.argv[3]) if len(sys.argv) > 3 else 180
    
    anonymize_csv(input_file, output_file, date_shift)