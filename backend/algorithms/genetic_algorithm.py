#!/usr/bin/env python3
"""
Test Genetic Algorithm with HeavyDB
Implements GA optimization using HeavyDB for fitness evaluation
Tests with real SENSEX data (subset)
"""

import numpy as np
import pymapd
import random
import time
import logging
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class GeneticAlgorithmHeavyDB:
    """GA implementation using HeavyDB for fitness evaluation"""
    
    def __init__(self, connection, table_name: str, total_strategies: int):
        self.con = connection
        self.table_name = table_name
        self.total_strategies = total_strategies
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Pre-calculate correlation matrix
        self.logger.info("Pre-calculating correlation matrix...")
        self.corr_matrix = self._calculate_correlation_matrix()
        
    def _calculate_correlation_matrix(self) -> np.ndarray:
        """Pre-calculate correlation matrix from HeavyDB data"""
        # Get all strategy data
        strategy_cols = [f"strategy_{i:04d}" for i in range(self.total_strategies)]
        
        query = f"""
        SELECT {', '.join(strategy_cols)}
        FROM {self.table_name}
        ORDER BY row_id
        """
        
        result = self.con.execute(query)
        data_matrix = np.array(list(result))
        
        # Calculate correlation matrix
        corr_matrix = np.corrcoef(data_matrix.T)
        
        self.logger.info(f"Correlation matrix calculated: {corr_matrix.shape}")
        return corr_matrix
    
    def evaluate_fitness(self, individual: List[int], metric: str = "ratio") -> float:
        """Evaluate fitness using HeavyDB queries"""
        # Build portfolio query
        portfolio_parts = []
        n = len(individual)
        
        for idx in individual:
            col_name = f"strategy_{idx:04d}"
            portfolio_parts.append(f"{col_name} / {n}.0")
        
        portfolio_calc = " + ".join(portfolio_parts)
        
        # Calculate ROI and Max Drawdown in one query
        query = f"""
        WITH portfolio_returns AS (
            SELECT 
                row_id,
                {portfolio_calc} as return_value
            FROM {self.table_name}
            ORDER BY row_id
        ),
        cumulative AS (
            SELECT 
                row_id,
                return_value,
                SUM(return_value) OVER (ORDER BY row_id) as cum_return
            FROM portfolio_returns
        ),
        peaks AS (
            SELECT 
                cum_return,
                MAX(cum_return) OVER (ORDER BY row_id ROWS UNBOUNDED PRECEDING) as peak
            FROM cumulative
        )
        SELECT 
            (SELECT SUM(return_value) FROM portfolio_returns) as total_roi,
            MAX(peak - cum_return) as max_dd
        FROM peaks
        """
        
        result = self.con.execute(query)
        roi, max_dd = list(result)[0]
        
        # Calculate base fitness (ratio)
        if max_dd > 1e-6:
            base_fitness = roi / max_dd
        elif roi > 0:
            base_fitness = roi * 100
        elif roi < 0:
            base_fitness = roi * 10
        else:
            base_fitness = 0
        
        # Add correlation penalty
        avg_corr = self._compute_avg_correlation(individual)
        penalty = 10 * avg_corr  # Fixed penalty weight from original
        
        return base_fitness - penalty
    
    def _compute_avg_correlation(self, individual: List[int]) -> float:
        """Compute average pairwise correlation"""
        if len(individual) < 2:
            return 0
        
        correlations = []
        n = len(individual)
        for i in range(n):
            for j in range(i + 1, n):
                correlations.append(self.corr_matrix[individual[i], individual[j]])
        
        return np.mean(correlations) if correlations else 0
    
    def crossover(self, parent1: List[int], parent2: List[int]) -> List[int]:
        """Uniform crossover"""
        child = []
        used = set()
        
        # Take from parents randomly
        for i in range(len(parent1)):
            if random.random() < 0.5:
                if parent1[i] not in used:
                    child.append(parent1[i])
                    used.add(parent1[i])
                elif parent2[i] not in used:
                    child.append(parent2[i])
                    used.add(parent2[i])
            else:
                if parent2[i] not in used:
                    child.append(parent2[i])
                    used.add(parent2[i])
                elif parent1[i] not in used:
                    child.append(parent1[i])
                    used.add(parent1[i])
        
        # Fill remaining slots
        while len(child) < len(parent1):
            strategy = random.randint(0, self.total_strategies - 1)
            if strategy not in used:
                child.append(strategy)
                used.add(strategy)
        
        return child
    
    def mutate(self, individual: List[int], mutation_rate: float = 0.1) -> List[int]:
        """Mutation operator"""
        mutated = individual.copy()
        
        for i in range(len(mutated)):
            if random.random() < mutation_rate:
                # Replace with a random strategy not in portfolio
                available = [s for s in range(self.total_strategies) if s not in mutated]
                if available:
                    mutated[i] = random.choice(available)
        
        return mutated
    
    def run_ga(self, portfolio_size: int, generations: int = 50, 
               population_size: int = 30, mutation_rate: float = 0.1) -> Tuple[List[int], float]:
        """Run genetic algorithm optimization"""
        self.logger.info(f"Starting GA: size={portfolio_size}, gens={generations}, pop={population_size}")
        
        # Initialize population
        population = []
        for _ in range(population_size):
            individual = random.sample(range(self.total_strategies), portfolio_size)
            population.append(individual)
        
        best_individual = None
        best_fitness = -np.inf
        fitness_history = []
        
        for gen in range(generations):
            # Evaluate fitness for all individuals
            fitness_scores = []
            start_time = time.time()
            
            for individual in population:
                fitness = self.evaluate_fitness(individual)
                fitness_scores.append(fitness)
                
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_individual = individual.copy()
            
            eval_time = time.time() - start_time
            
            # Record stats
            avg_fitness = np.mean(fitness_scores)
            fitness_history.append({
                'generation': gen,
                'best': best_fitness,
                'average': avg_fitness,
                'eval_time': eval_time
            })
            
            if gen % 10 == 0:
                self.logger.info(
                    f"Gen {gen}: Best={best_fitness:.4f}, "
                    f"Avg={avg_fitness:.4f}, Time={eval_time:.2f}s"
                )
            
            # Selection and reproduction
            if gen < generations - 1:  # Don't evolve on last generation
                # Tournament selection
                new_population = []
                
                # Keep best individual (elitism)
                new_population.append(best_individual)
                
                while len(new_population) < population_size:
                    # Tournament selection
                    tournament_size = 3
                    tournament_indices = random.sample(range(population_size), tournament_size)
                    tournament_fitness = [fitness_scores[i] for i in tournament_indices]
                    winner_idx = tournament_indices[np.argmax(tournament_fitness)]
                    parent1 = population[winner_idx]
                    
                    # Select second parent
                    tournament_indices = random.sample(range(population_size), tournament_size)
                    tournament_fitness = [fitness_scores[i] for i in tournament_indices]
                    winner_idx = tournament_indices[np.argmax(tournament_fitness)]
                    parent2 = population[winner_idx]
                    
                    # Crossover and mutation
                    child = self.crossover(parent1, parent2)
                    child = self.mutate(child, mutation_rate)
                    
                    new_population.append(child)
                
                population = new_population
        
        return best_individual, best_fitness, fitness_history


def test_ga_with_sensex():
    """Test GA with SENSEX subset data"""
    print("="*60)
    print("Testing Genetic Algorithm with HeavyDB")
    print("="*60)
    
    # Connect to HeavyDB
    try:
        con = pymapd.connect(
            host='localhost',
            port=6274,
            user='portfolio_user',
            password='portfolio123',
            dbname='portfolio_optimizer'
        )
        print("✓ Connected to HeavyDB")
    except Exception as e:
        print(f"✗ Failed to connect: {e}")
        return False
    
    # Use subset table with 100 strategies
    table_name = "sensex_subset_20250719_211121"
    total_strategies = 100
    
    ga = GeneticAlgorithmHeavyDB(con, table_name, total_strategies)
    
    # Test different portfolio sizes
    portfolio_sizes = [10, 30, 50]
    results = {}
    
    for size in portfolio_sizes:
        print(f"\n--- Optimizing portfolio size {size} ---")
        
        start_time = time.time()
        best_portfolio, best_fitness, history = ga.run_ga(
            portfolio_size=size,
            generations=30,  # Reduced for testing
            population_size=20,
            mutation_rate=0.1
        )
        total_time = time.time() - start_time
        
        print(f"\nOptimization completed in {total_time:.2f}s")
        print(f"Best fitness: {best_fitness:.4f}")
        print(f"Best portfolio (first 10): {best_portfolio[:10]}")
        
        # Calculate detailed metrics for best portfolio
        portfolio_parts = []
        for idx in best_portfolio:
            col_name = f"strategy_{idx:04d}"
            portfolio_parts.append(f"{col_name} / {len(best_portfolio)}.0")
        
        portfolio_calc = " + ".join(portfolio_parts)
        
        # Get performance metrics
        query = f"""
        WITH portfolio_returns AS (
            SELECT {portfolio_calc} as return_value
            FROM {table_name}
        )
        SELECT 
            SUM(return_value) as total_roi,
            COUNT(*) as days,
            SUM(CASE WHEN return_value > 0 THEN 1 ELSE 0 END) as win_days,
            AVG(return_value) as avg_return
        FROM portfolio_returns
        """
        
        result = con.execute(query)
        roi, days, win_days, avg_return = list(result)[0]
        
        print(f"\nPerformance metrics:")
        print(f"  Total ROI: {roi:.2f}")
        print(f"  Win rate: {win_days/days*100:.1f}%")
        print(f"  Avg daily return: {avg_return:.4f}")
        print(f"  Avg correlation: {ga._compute_avg_correlation(best_portfolio):.4f}")
        
        results[size] = {
            'best_fitness': best_fitness,
            'total_roi': roi,
            'win_rate': win_days/days,
            'optimization_time': total_time,
            'best_portfolio': best_portfolio,
            'fitness_history': history
        }
    
    # Save results
    summary = {
        'timestamp': datetime.now().isoformat(),
        'table': table_name,
        'total_strategies': total_strategies,
        'results': results
    }
    
    with open('ga_heavydb_results.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    
    for size, result in results.items():
        print(f"\nPortfolio size {size}:")
        print(f"  Best fitness: {result['best_fitness']:.4f}")
        print(f"  Total ROI: {result['total_roi']:.2f}")
        print(f"  Win rate: {result['win_rate']*100:.1f}%")
        print(f"  Time: {result['optimization_time']:.2f}s")
    
    con.close()
    print("\n✅ GA test completed successfully!")
    print("Results saved to: ga_heavydb_results.json")
    
    return True


if __name__ == "__main__":
    import sys
    success = test_ga_with_sensex()
    sys.exit(0 if success else 1)