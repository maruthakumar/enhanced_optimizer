#!/usr/bin/env python3
"""
Example integration of the Correlation Matrix Calculator with optimization algorithms.
Shows how to preserve the legacy correlation penalty logic in the new architecture.
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from correlation import CorrelationMatrixCalculator


class OptimizationAlgorithmWithCorrelation:
    """
    Example base class for optimization algorithms that use correlation-based diversification.
    """
    
    def __init__(self, config_path: str = None):
        # Initialize correlation calculator
        self.correlation_calculator = CorrelationMatrixCalculator(config_path)
        
        # Cache for full correlation matrix (shared across all algorithms in a run)
        self._full_correlation_matrix = None
        self._correlation_matrix_hash = None
    
    def _ensure_correlation_matrix(self, daily_matrix: np.ndarray):
        """Ensure we have the correlation matrix calculated and cached"""
        # Generate hash of the data to check if it's the same
        data_hash = self.correlation_calculator._generate_cache_key(daily_matrix)
        
        if self._correlation_matrix_hash != data_hash:
            # Calculate new correlation matrix
            self._full_correlation_matrix = self.correlation_calculator.calculate_full_correlation_matrix(daily_matrix)
            self._correlation_matrix_hash = data_hash
            
            # Analyze the matrix for insights
            analysis = self.correlation_calculator.analyze_correlation_matrix(self._full_correlation_matrix)
            print(f"📊 Correlation Matrix Analysis:")
            print(f"   Average correlation: {analysis['avg_correlation']:.4f}")
            print(f"   Max correlation: {analysis['max_correlation']:.4f}")
            print(f"   High correlation pairs (>{self.correlation_calculator.config.correlation_threshold}): "
                  f"{len(analysis['high_correlation_pairs'])}")
    
    def evaluate_portfolio(self, daily_matrix: np.ndarray, portfolio: np.ndarray) -> float:
        """
        Evaluate portfolio fitness with correlation penalty.
        This preserves the legacy formula exactly.
        """
        # Calculate base fitness (e.g., Sharpe ratio or ROI/Drawdown)
        base_fitness = self._calculate_base_fitness(daily_matrix, portfolio)
        
        # Apply correlation penalty using the legacy formula
        adjusted_fitness, penalty = self.correlation_calculator.evaluate_fitness_with_correlation(
            base_fitness, daily_matrix, portfolio
        )
        
        return adjusted_fitness
    
    def _calculate_base_fitness(self, daily_matrix: np.ndarray, portfolio: np.ndarray) -> float:
        """Calculate base fitness metric (to be implemented by specific algorithms)"""
        # Example: Simple average return
        portfolio_returns = daily_matrix[:, portfolio]
        return np.mean(np.sum(portfolio_returns, axis=1))
    
    def get_low_correlation_candidates(self, daily_matrix: np.ndarray, 
                                     current_portfolio: np.ndarray,
                                     candidate_strategies: np.ndarray,
                                     max_correlation: float = 0.5) -> np.ndarray:
        """
        Get candidate strategies that have low correlation with current portfolio.
        Useful for diversification during optimization.
        """
        if len(current_portfolio) == 0:
            return candidate_strategies
        
        # Ensure we have the full correlation matrix
        self._ensure_correlation_matrix(daily_matrix)
        
        # Calculate average correlation of each candidate with current portfolio
        low_corr_candidates = []
        
        for candidate in candidate_strategies:
            if candidate in current_portfolio:
                continue
                
            # Get correlations between candidate and portfolio strategies
            correlations = self._full_correlation_matrix[candidate, current_portfolio]
            avg_corr = np.mean(np.abs(correlations))
            
            if avg_corr < max_correlation:
                low_corr_candidates.append(candidate)
        
        return np.array(low_corr_candidates)


class GeneticAlgorithmWithCorrelation(OptimizationAlgorithmWithCorrelation):
    """
    Example: Genetic Algorithm with correlation-based diversification
    """
    
    def optimize(self, daily_matrix: np.ndarray, portfolio_size: int, 
                 generations: int = 50, population_size: int = 100) -> np.ndarray:
        """
        Run genetic algorithm optimization with correlation penalty.
        """
        n_strategies = daily_matrix.shape[1]
        
        # Ensure correlation matrix is calculated
        self._ensure_correlation_matrix(daily_matrix)
        
        # Initialize population
        population = []
        for _ in range(population_size):
            portfolio = np.random.choice(n_strategies, portfolio_size, replace=False)
            population.append(portfolio)
        
        # Evolution loop
        for generation in range(generations):
            # Evaluate fitness for all portfolios
            fitness_scores = []
            for portfolio in population:
                fitness = self.evaluate_portfolio(daily_matrix, portfolio)
                fitness_scores.append(fitness)
            
            # Select top performers
            sorted_indices = np.argsort(fitness_scores)[::-1]
            top_performers = [population[i] for i in sorted_indices[:population_size//2]]
            
            # Create new generation through crossover and mutation
            new_population = top_performers.copy()
            
            while len(new_population) < population_size:
                # Select parents
                parent1 = top_performers[np.random.randint(len(top_performers))]
                parent2 = top_performers[np.random.randint(len(top_performers))]
                
                # Crossover with diversification
                child = self._crossover_with_diversification(
                    daily_matrix, parent1, parent2, portfolio_size
                )
                
                # Mutation
                if np.random.random() < 0.1:  # 10% mutation rate
                    child = self._mutate_with_diversification(
                        daily_matrix, child, n_strategies
                    )
                
                new_population.append(child)
            
            population = new_population
            
            # Log progress
            best_fitness = max(fitness_scores)
            avg_fitness = np.mean(fitness_scores)
            print(f"Generation {generation+1}: Best={best_fitness:.4f}, Avg={avg_fitness:.4f}")
        
        # Return best portfolio
        final_fitness_scores = [self.evaluate_portfolio(daily_matrix, p) for p in population]
        best_idx = np.argmax(final_fitness_scores)
        return population[best_idx]
    
    def _crossover_with_diversification(self, daily_matrix: np.ndarray,
                                      parent1: np.ndarray, parent2: np.ndarray,
                                      portfolio_size: int) -> np.ndarray:
        """Crossover that considers correlation for diversification"""
        # Combine unique strategies from both parents
        combined = np.unique(np.concatenate([parent1, parent2]))
        
        if len(combined) <= portfolio_size:
            # Need to add more strategies
            all_strategies = np.arange(daily_matrix.shape[1])
            remaining = np.setdiff1d(all_strategies, combined)
            
            # Get low-correlation candidates
            low_corr = self.get_low_correlation_candidates(
                daily_matrix, combined, remaining, max_correlation=0.6
            )
            
            if len(low_corr) > 0:
                n_needed = portfolio_size - len(combined)
                additional = np.random.choice(low_corr, min(n_needed, len(low_corr)), replace=False)
                combined = np.concatenate([combined, additional])
        
        # Select final portfolio
        if len(combined) > portfolio_size:
            # Evaluate each strategy's contribution
            scores = []
            for strategy in combined:
                temp_portfolio = np.array([strategy])
                score = self._calculate_base_fitness(daily_matrix, temp_portfolio)
                scores.append(score)
            
            # Select top performers
            sorted_indices = np.argsort(scores)[::-1]
            child = combined[sorted_indices[:portfolio_size]]
        else:
            child = combined
        
        return child
    
    def _mutate_with_diversification(self, daily_matrix: np.ndarray,
                                   portfolio: np.ndarray,
                                   n_strategies: int) -> np.ndarray:
        """Mutation that considers correlation for diversification"""
        mutated = portfolio.copy()
        
        # Replace one strategy with a low-correlation alternative
        idx_to_replace = np.random.randint(len(mutated))
        
        # Find low-correlation candidates
        all_strategies = np.arange(n_strategies)
        candidates = np.setdiff1d(all_strategies, mutated)
        
        low_corr = self.get_low_correlation_candidates(
            daily_matrix, mutated, candidates, max_correlation=0.5
        )
        
        if len(low_corr) > 0:
            new_strategy = np.random.choice(low_corr)
            mutated[idx_to_replace] = new_strategy
        
        return mutated


def demonstrate_integration():
    """Demonstrate the integration with sample data"""
    print("🚀 Correlation Module Integration Example")
    print("="*60)
    
    # Generate sample data
    np.random.seed(42)
    n_days = 82
    n_strategies = 100
    
    # Create correlated data
    base_returns = np.random.randn(n_days, 10) * 0.02
    weights = np.random.randn(10, n_strategies)
    daily_matrix = base_returns @ weights + np.random.randn(n_days, n_strategies) * 0.01
    
    # Initialize algorithm with correlation
    config_path = "/mnt/optimizer_share/config/production_config.ini"
    ga = GeneticAlgorithmWithCorrelation(config_path)
    
    # Run optimization
    portfolio_size = 20
    print(f"\nOptimizing portfolio of {portfolio_size} strategies from {n_strategies} total")
    print("Using Genetic Algorithm with correlation-based diversification\n")
    
    best_portfolio = ga.optimize(
        daily_matrix, 
        portfolio_size,
        generations=20,
        population_size=50
    )
    
    print(f"\n✅ Best portfolio found: {sorted(best_portfolio)}")
    
    # Calculate final metrics
    final_fitness = ga.evaluate_portfolio(daily_matrix, best_portfolio)
    avg_correlation = ga.correlation_calculator.compute_avg_pairwise_correlation(
        daily_matrix, best_portfolio
    )
    
    print(f"\n📊 Final Portfolio Metrics:")
    print(f"   Fitness (with correlation penalty): {final_fitness:.4f}")
    print(f"   Average pairwise correlation: {avg_correlation:.4f}")
    print(f"   Correlation penalty weight: {ga.correlation_calculator.config.penalty_weight}")


if __name__ == "__main__":
    demonstrate_integration()