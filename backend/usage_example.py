#!/usr/bin/env python3
"""
Production Usage Example
How to use the Enhanced HeavyDB Optimization System in production
"""

import pandas as pd
import numpy as np
from enhanced_optimizer import FixedCompleteEnhancedOptimizer

def main():
    """Production usage example"""
    
    # Load your data
    df = pd.read_excel('your_data_file.xlsx')
    
    # Prepare data matrix
    reserved_columns = ['Date', 'Day', 'Zone', 'date', 'day', 'zone']
    strategy_columns = [col for col in df.columns if col not in reserved_columns]
    
    daily_matrix = np.zeros((len(df), len(strategy_columns)))
    for i, col in enumerate(strategy_columns):
        daily_matrix[:, i] = pd.to_numeric(df[col], errors='coerce').fillna(0).values
    
    # Initialize optimizer
    optimizer = FixedCompleteEnhancedOptimizer(connection_pool_size=3)
    
    # Single algorithm optimization
    result = optimizer.genetic_algorithm(daily_matrix, 20, 'ratio', generations=50)
    print(f"GA Result: fitness={result.fitness:.6f}, time={result.execution_time:.2f}s")
    
    # Parallel optimization (all algorithms)
    results = optimizer.optimize_parallel(daily_matrix, 20, 'ratio')
    print(f"Best Result: {results['best_algorithm']} with fitness={results['best_fitness']:.6f}")
    
    return results

if __name__ == "__main__":
    main()
