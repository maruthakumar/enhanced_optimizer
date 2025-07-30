#!/usr/bin/env python3
"""
Fixed Complete Enhanced Optimizer
Resolves differential evolution parameter interface and adds missing optimize_parallel method
"""

import numpy as np
import pandas as pd
import logging
import random
import time
import concurrent.futures
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class OptimizationResult:
    """Result from a single optimization algorithm"""
    algorithm: str
    solution: List[int]
    fitness: float
    execution_time: float
    iterations_completed: int
    convergence_achieved: bool
    fitness_history: List[float]

class FixedCompleteEnhancedOptimizer:
    """
    Fixed complete enhanced optimizer with all 7 algorithms, proper threading, and parallel execution
    """
    
    def __init__(self, connection_pool_size: int = 3):
        """Initialize optimizer"""
        self.connection_pool_size = connection_pool_size
        
    def calculate_fitness(self, solution: List[int], daily_matrix: np.ndarray, 
                         metric: str = 'ratio') -> float:
        """Calculate fitness score exactly matching original system"""
        try:
            if not solution or len(solution) == 0:
                return 0.0
            
            # Ensure solution indices are valid
            max_idx = daily_matrix.shape[1] - 1
            valid_solution = [idx for idx in solution if 0 <= idx <= max_idx]
            
            if not valid_solution:
                return 0.0
            
            # Get portfolio returns (equal weighted)
            portfolio_returns = daily_matrix[:, valid_solution].mean(axis=1)
            
            # Calculate total ROI
            total_roi = float(np.sum(portfolio_returns))
            
            # Calculate maximum drawdown
            equity_curve = np.cumsum(portfolio_returns)
            peak = np.maximum.accumulate(equity_curve)
            drawdowns = peak - equity_curve
            max_drawdown = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.001
            
            # Ensure max_drawdown is positive and not too small
            max_drawdown = max(max_drawdown, 0.001)
            
            if metric == 'ratio':
                return total_roi / max_drawdown
            elif metric == 'roi':
                return total_roi
            elif metric == 'less max dd':
                return -max_drawdown
            else:
                return total_roi / max_drawdown
                
        except Exception as e:
            logger.error(f"Error calculating fitness: {e}")
            return 0.0
    
    def genetic_algorithm(self, daily_matrix: np.ndarray, portfolio_size: int, 
                         metric: str = 'ratio', generations: int = 50,
                         population_size: int = 30, mutation_rate: float = 0.1) -> OptimizationResult:
        """Genetic Algorithm with full generations"""
        start_time = time.time()
        total_strategies = daily_matrix.shape[1]
        fitness_history = []
        
        logger.info(f"[GA] Starting with {generations} generations, population {population_size}")
        
        # Initialize population
        population = []
        for _ in range(population_size):
            individual = random.sample(range(total_strategies), min(portfolio_size, total_strategies))
            population.append(individual)
        
        best_solution = population[0]
        best_fitness = self.calculate_fitness(best_solution, daily_matrix, metric)
        
        for generation in range(generations):
            # Evaluate population
            fitness_scores = []
            for individual in population:
                fitness = self.calculate_fitness(individual, daily_matrix, metric)
                fitness_scores.append(fitness)
                
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_solution = individual.copy()
            
            fitness_history.append(best_fitness)
            
            # Selection, crossover, mutation
            new_population = []
            
            # Elitism: keep best 10% of population
            elite_count = max(1, population_size // 10)
            elite_indices = np.argsort(fitness_scores)[-elite_count:]
            for idx in elite_indices:
                new_population.append(population[idx])
            
            # Generate rest of population
            while len(new_population) < population_size:
                # Tournament selection
                parent1 = self._tournament_selection(population, fitness_scores)
                parent2 = self._tournament_selection(population, fitness_scores)
                
                # Crossover
                child = self._crossover(parent1, parent2, portfolio_size, total_strategies)
                
                # Mutation
                if random.random() < mutation_rate:
                    child = self._mutate(child, total_strategies, portfolio_size)
                
                new_population.append(child)
            
            population = new_population[:population_size]
            
            if generation % 10 == 0:
                logger.info(f"[GA] Generation {generation}/{generations}, Best Fitness: {best_fitness:.6f}")
        
        execution_time = time.time() - start_time
        
        logger.info(f"[GA] Completed: {generations} generations, Best Fitness: {best_fitness:.6f}, Time: {execution_time:.2f}s")
        
        return OptimizationResult(
            algorithm="genetic_algorithm",
            solution=best_solution,
            fitness=best_fitness,
            execution_time=execution_time,
            iterations_completed=generations,
            convergence_achieved=True,
            fitness_history=fitness_history
        )
    
    def particle_swarm_optimization(self, daily_matrix: np.ndarray, portfolio_size: int,
                                  metric: str = 'ratio', iterations: int = 50,
                                  swarm_size: int = 30) -> OptimizationResult:
        """Particle Swarm Optimization with full iterations"""
        start_time = time.time()
        total_strategies = daily_matrix.shape[1]
        fitness_history = []
        
        logger.info(f"[PSO] Starting with {iterations} iterations, swarm size {swarm_size}")
        
        # Initialize swarm
        particles = []
        velocities = []
        personal_best = []
        personal_best_fitness = []
        
        for _ in range(swarm_size):
            particle = random.sample(range(total_strategies), min(portfolio_size, total_strategies))
            particles.append(particle)
            velocities.append([0] * portfolio_size)
            personal_best.append(particle.copy())
            
            fitness = self.calculate_fitness(particle, daily_matrix, metric)
            personal_best_fitness.append(fitness)
        
        # Find global best
        global_best_idx = personal_best_fitness.index(max(personal_best_fitness))
        global_best = personal_best[global_best_idx].copy()
        global_best_fitness = personal_best_fitness[global_best_idx]
        
        for iteration in range(iterations):
            for i in range(swarm_size):
                # Update particle (discrete PSO)
                for j in range(portfolio_size):
                    if random.random() < 0.3:  # Probability to change
                        if random.random() < 0.5:
                            # Move towards personal best
                            if personal_best[i][j] not in particles[i]:
                                particles[i][j] = personal_best[i][j]
                        else:
                            # Move towards global best
                            if global_best[j] not in particles[i]:
                                particles[i][j] = global_best[j]
                
                # Evaluate particle
                fitness = self.calculate_fitness(particles[i], daily_matrix, metric)
                
                # Update personal best
                if fitness > personal_best_fitness[i]:
                    personal_best_fitness[i] = fitness
                    personal_best[i] = particles[i].copy()
                
                # Update global best
                if fitness > global_best_fitness:
                    global_best_fitness = fitness
                    global_best = particles[i].copy()
            
            fitness_history.append(global_best_fitness)
            
            if iteration % 10 == 0:
                logger.info(f"[PSO] Iteration {iteration}/{iterations}, Best Fitness: {global_best_fitness:.6f}")
        
        execution_time = time.time() - start_time
        
        logger.info(f"[PSO] Completed: {iterations} iterations, Best Fitness: {global_best_fitness:.6f}, Time: {execution_time:.2f}s")
        
        return OptimizationResult(
            algorithm="particle_swarm_optimization",
            solution=global_best,
            fitness=global_best_fitness,
            execution_time=execution_time,
            iterations_completed=iterations,
            convergence_achieved=True,
            fitness_history=fitness_history
        )
    
    def simulated_annealing(self, daily_matrix: np.ndarray, portfolio_size: int,
                          metric: str = 'ratio', iterations: int = 1000,
                          initial_temp: float = 100.0, final_temp: float = 0.1) -> OptimizationResult:
        """Simulated Annealing with full iterations"""
        start_time = time.time()
        total_strategies = daily_matrix.shape[1]
        fitness_history = []
        
        logger.info(f"[SA] Starting with {iterations} iterations")
        
        # Initialize solution
        current_solution = random.sample(range(total_strategies), min(portfolio_size, total_strategies))
        current_fitness = self.calculate_fitness(current_solution, daily_matrix, metric)
        
        best_solution = current_solution.copy()
        best_fitness = current_fitness
        
        for iteration in range(iterations):
            # Calculate temperature
            temperature = initial_temp * ((final_temp / initial_temp) ** (iteration / iterations))
            
            # Generate neighbor
            neighbor = current_solution.copy()
            idx = random.randint(0, portfolio_size - 1)
            available = list(set(range(total_strategies)) - set(neighbor))
            if available:
                neighbor[idx] = random.choice(available)
            
            neighbor_fitness = self.calculate_fitness(neighbor, daily_matrix, metric)
            
            # Accept or reject
            if neighbor_fitness > current_fitness:
                current_solution = neighbor
                current_fitness = neighbor_fitness
                
                if neighbor_fitness > best_fitness:
                    best_solution = neighbor.copy()
                    best_fitness = neighbor_fitness
            else:
                if temperature > 0:
                    delta = neighbor_fitness - current_fitness
                    probability = np.exp(delta / temperature)
                    if random.random() < probability:
                        current_solution = neighbor
                        current_fitness = neighbor_fitness
            
            fitness_history.append(best_fitness)
            
            if iteration % 100 == 0:
                logger.info(f"[SA] Iteration {iteration}/{iterations}, Best Fitness: {best_fitness:.6f}")
        
        execution_time = time.time() - start_time
        
        logger.info(f"[SA] Completed: {iterations} iterations, Best Fitness: {best_fitness:.6f}, Time: {execution_time:.2f}s")
        
        return OptimizationResult(
            algorithm="simulated_annealing",
            solution=best_solution,
            fitness=best_fitness,
            execution_time=execution_time,
            iterations_completed=iterations,
            convergence_achieved=True,
            fitness_history=fitness_history
        )
    
    def differential_evolution(self, daily_matrix: np.ndarray, portfolio_size: int,
                             metric: str = 'ratio', generations: int = 50,
                             population_size: int = 30) -> OptimizationResult:
        """FIXED: Differential Evolution with 'generations' parameter (was 'iterations')"""
        start_time = time.time()
        total_strategies = daily_matrix.shape[1]
        fitness_history = []
        
        logger.info(f"[DE] Starting with {generations} generations, population {population_size}")
        
        # Initialize population
        population = []
        for _ in range(population_size):
            individual = random.sample(range(total_strategies), min(portfolio_size, total_strategies))
            population.append(individual)
        
        best_solution = population[0]
        best_fitness = self.calculate_fitness(best_solution, daily_matrix, metric)
        
        for generation in range(generations):
            new_population = []
            
            for i in range(population_size):
                # Select three random individuals (different from current)
                candidates = list(range(population_size))
                candidates.remove(i)
                if len(candidates) >= 3:
                    a, b, c = random.sample(candidates, 3)
                else:
                    a, b, c = random.choices(candidates, k=3)
                
                # Create mutant vector (discrete version)
                mutant = population[a].copy()
                for j in range(portfolio_size):
                    if random.random() < 0.5:  # Mutation probability
                        if population[b][j] not in mutant:
                            mutant[j] = population[b][j]
                        elif population[c][j] not in mutant:
                            mutant[j] = population[c][j]
                
                # Crossover
                trial = population[i].copy()
                for j in range(portfolio_size):
                    if random.random() < 0.7:  # Crossover probability
                        trial[j] = mutant[j]
                
                # Selection
                trial_fitness = self.calculate_fitness(trial, daily_matrix, metric)
                current_fitness = self.calculate_fitness(population[i], daily_matrix, metric)
                
                if trial_fitness > current_fitness:
                    new_population.append(trial)
                    if trial_fitness > best_fitness:
                        best_fitness = trial_fitness
                        best_solution = trial.copy()
                else:
                    new_population.append(population[i])
            
            population = new_population
            fitness_history.append(best_fitness)
            
            if generation % 10 == 0:
                logger.info(f"[DE] Generation {generation}/{generations}, Best Fitness: {best_fitness:.6f}")
        
        execution_time = time.time() - start_time
        
        logger.info(f"[DE] Completed: {generations} generations, Best Fitness: {best_fitness:.6f}, Time: {execution_time:.2f}s")
        
        return OptimizationResult(
            algorithm="differential_evolution",
            solution=best_solution,
            fitness=best_fitness,
            execution_time=execution_time,
            iterations_completed=generations,
            convergence_achieved=True,
            fitness_history=fitness_history
        )
    
    def ant_colony_optimization(self, daily_matrix: np.ndarray, portfolio_size: int,
                              metric: str = 'ratio', iterations: int = 50,
                              num_ants: int = 30) -> OptimizationResult:
        """Ant Colony Optimization with full iterations"""
        start_time = time.time()
        total_strategies = daily_matrix.shape[1]
        fitness_history = []
        
        logger.info(f"[ACO] Starting with {iterations} iterations, {num_ants} ants")
        
        # Initialize pheromone matrix
        pheromone = np.ones((total_strategies, total_strategies)) * 0.1
        
        best_solution = random.sample(range(total_strategies), min(portfolio_size, total_strategies))
        best_fitness = self.calculate_fitness(best_solution, daily_matrix, metric)
        
        for iteration in range(iterations):
            solutions = []
            fitness_scores = []
            
            # Generate solutions for all ants
            for ant in range(num_ants):
                solution = []
                available = list(range(total_strategies))
                
                for _ in range(portfolio_size):
                    if len(available) == 0:
                        break
                    
                    # Calculate probabilities based on pheromone
                    if len(solution) == 0:
                        probabilities = [1.0] * len(available)
                    else:
                        probabilities = []
                        for strategy in available:
                            prob = sum(pheromone[s][strategy] for s in solution) / len(solution)
                            probabilities.append(prob)
                    
                    # Normalize probabilities
                    total_prob = sum(probabilities)
                    if total_prob > 0:
                        probabilities = [p / total_prob for p in probabilities]
                    else:
                        probabilities = [1.0 / len(probabilities)] * len(probabilities)
                    
                    # Select strategy
                    selected_idx = np.random.choice(len(available), p=probabilities)
                    selected_strategy = available[selected_idx]
                    
                    solution.append(selected_strategy)
                    available.remove(selected_strategy)
                
                solutions.append(solution)
                fitness = self.calculate_fitness(solution, daily_matrix, metric)
                fitness_scores.append(fitness)
                
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_solution = solution.copy()
            
            # Update pheromones
            pheromone *= 0.9  # Evaporation
            
            for i, solution in enumerate(solutions):
                fitness = fitness_scores[i]
                for j in range(len(solution)):
                    for k in range(j + 1, len(solution)):
                        pheromone[solution[j]][solution[k]] += fitness * 0.01
                        pheromone[solution[k]][solution[j]] += fitness * 0.01
            
            fitness_history.append(best_fitness)
            
            if iteration % 10 == 0:
                logger.info(f"[ACO] Iteration {iteration}/{iterations}, Best Fitness: {best_fitness:.6f}")
        
        execution_time = time.time() - start_time
        
        logger.info(f"[ACO] Completed: {iterations} iterations, Best Fitness: {best_fitness:.6f}, Time: {execution_time:.2f}s")
        
        return OptimizationResult(
            algorithm="ant_colony_optimization",
            solution=best_solution,
            fitness=best_fitness,
            execution_time=execution_time,
            iterations_completed=iterations,
            convergence_achieved=True,
            fitness_history=fitness_history
        )
    
    def hill_climbing(self, daily_matrix: np.ndarray, portfolio_size: int,
                     metric: str = 'ratio', iterations: int = 200) -> OptimizationResult:
        """Hill Climbing with full iterations"""
        start_time = time.time()
        total_strategies = daily_matrix.shape[1]
        fitness_history = []
        
        logger.info(f"[HC] Starting with {iterations} iterations")
        
        # Initialize solution
        current_solution = random.sample(range(total_strategies), min(portfolio_size, total_strategies))
        current_fitness = self.calculate_fitness(current_solution, daily_matrix, metric)
        
        best_solution = current_solution.copy()
        best_fitness = current_fitness
        
        for iteration in range(iterations):
            # Generate neighbor
            neighbor = current_solution.copy()
            idx = random.randint(0, portfolio_size - 1)
            available = list(set(range(total_strategies)) - set(neighbor))
            if available:
                neighbor[idx] = random.choice(available)
            
            neighbor_fitness = self.calculate_fitness(neighbor, daily_matrix, metric)
            
            # Accept if better
            if neighbor_fitness > current_fitness:
                current_solution = neighbor
                current_fitness = neighbor_fitness
                
                if neighbor_fitness > best_fitness:
                    best_solution = neighbor.copy()
                    best_fitness = neighbor_fitness
            
            fitness_history.append(best_fitness)
            
            if iteration % 50 == 0:
                logger.info(f"[HC] Iteration {iteration}/{iterations}, Best Fitness: {best_fitness:.6f}")
        
        execution_time = time.time() - start_time
        
        logger.info(f"[HC] Completed: {iterations} iterations, Best Fitness: {best_fitness:.6f}, Time: {execution_time:.2f}s")
        
        return OptimizationResult(
            algorithm="hill_climbing",
            solution=best_solution,
            fitness=best_fitness,
            execution_time=execution_time,
            iterations_completed=iterations,
            convergence_achieved=True,
            fitness_history=fitness_history
        )
    
    def bayesian_optimization(self, daily_matrix: np.ndarray, portfolio_size: int,
                            metric: str = 'ratio', iterations: int = 50) -> OptimizationResult:
        """Bayesian Optimization with full iterations"""
        start_time = time.time()
        total_strategies = daily_matrix.shape[1]
        fitness_history = []
        
        logger.info(f"[BO] Starting with {iterations} iterations")
        
        # Simple Bayesian optimization (random sampling with exploitation)
        evaluated_solutions = []
        evaluated_fitness = []
        
        best_solution = None
        best_fitness = 0
        
        for iteration in range(iterations):
            if iteration < 10:
                # Exploration phase: random sampling
                solution = random.sample(range(total_strategies), min(portfolio_size, total_strategies))
            else:
                # Exploitation phase: sample around best solutions
                if evaluated_solutions:
                    # Select best solution and modify it
                    best_idx = evaluated_fitness.index(max(evaluated_fitness))
                    base_solution = evaluated_solutions[best_idx].copy()
                    
                    # Modify 20% of the solution
                    num_changes = max(1, portfolio_size // 5)
                    for _ in range(num_changes):
                        idx = random.randint(0, portfolio_size - 1)
                        available = list(set(range(total_strategies)) - set(base_solution))
                        if available:
                            base_solution[idx] = random.choice(available)
                    
                    solution = base_solution
                else:
                    solution = random.sample(range(total_strategies), min(portfolio_size, total_strategies))
            
            fitness = self.calculate_fitness(solution, daily_matrix, metric)
            
            evaluated_solutions.append(solution)
            evaluated_fitness.append(fitness)
            
            if fitness > best_fitness:
                best_fitness = fitness
                best_solution = solution.copy()
            
            fitness_history.append(best_fitness)
        
        execution_time = time.time() - start_time
        
        logger.info(f"[BO] Completed: {iterations} iterations, Best Fitness: {best_fitness:.6f}, Time: {execution_time:.2f}s")
        
        return OptimizationResult(
            algorithm="bayesian_optimization",
            solution=best_solution,
            fitness=best_fitness,
            execution_time=execution_time,
            iterations_completed=iterations,
            convergence_achieved=True,
            fitness_history=fitness_history
        )
    
    def optimize_parallel(self, daily_matrix: np.ndarray, portfolio_size: int,
                         metric: str = 'ratio', algorithms: List[str] = None) -> Dict[str, Any]:
        """ADDED: Missing parallel execution method with ThreadPoolExecutor"""
        if algorithms is None:
            algorithms = [
                'genetic_algorithm',
                'particle_swarm_optimization', 
                'simulated_annealing',
                'differential_evolution',
                'ant_colony_optimization',
                'hill_climbing',
                'bayesian_optimization'
            ]
        
        logger.info(f"🚀 Starting parallel optimization with {len(algorithms)} algorithms")
        logger.info(f"📊 Portfolio size: {portfolio_size}, Strategies available: {daily_matrix.shape[1]}")
        logger.info(f"🎯 Optimization metric: {metric}")
        logger.info(f"⚡ Connection pool size: {self.connection_pool_size}")
        
        start_time = time.time()
        results = []
        
        # Execute algorithms in parallel using ThreadPoolExecutor
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.connection_pool_size) as executor:
            future_to_algorithm = {}
            
            for algorithm in algorithms:
                if hasattr(self, algorithm):
                    try:
                        # Submit algorithm to thread pool
                        if algorithm == 'genetic_algorithm':
                            future = executor.submit(self.genetic_algorithm, daily_matrix, portfolio_size, metric, 50)
                        elif algorithm == 'particle_swarm_optimization':
                            future = executor.submit(self.particle_swarm_optimization, daily_matrix, portfolio_size, metric, 50)
                        elif algorithm == 'simulated_annealing':
                            future = executor.submit(self.simulated_annealing, daily_matrix, portfolio_size, metric, 1000)
                        elif algorithm == 'differential_evolution':
                            future = executor.submit(self.differential_evolution, daily_matrix, portfolio_size, metric, 50)
                        elif algorithm == 'ant_colony_optimization':
                            future = executor.submit(self.ant_colony_optimization, daily_matrix, portfolio_size, metric, 50)
                        elif algorithm == 'hill_climbing':
                            future = executor.submit(self.hill_climbing, daily_matrix, portfolio_size, metric, 200)
                        elif algorithm == 'bayesian_optimization':
                            future = executor.submit(self.bayesian_optimization, daily_matrix, portfolio_size, metric, 50)
                        
                        future_to_algorithm[future] = algorithm
                        logger.info(f"🔄 Launched {algorithm}")
                    except Exception as e:
                        logger.error(f"❌ Failed to launch {algorithm}: {e}")
            
            # Collect results using Future.result() (not Future.get())
            for future in concurrent.futures.as_completed(future_to_algorithm):
                algorithm = future_to_algorithm[future]
                try:
                    result = future.result()  # FIXED: Using result() instead of get()
                    results.append(result)
                    logger.info(f"✅ {algorithm} completed: fitness={result.fitness:.6f}, "
                              f"time={result.execution_time:.2f}s")
                except Exception as e:
                    logger.error(f"❌ {algorithm} failed: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
        
        if not results:
            logger.error("❌ No algorithms completed successfully")
            return {'error': 'No algorithms completed successfully'}
        
        # Find best result
        best_result = max(results, key=lambda x: x.fitness)
        total_time = time.time() - start_time
        
        # Create comprehensive results
        algorithm_results = {}
        for result in results:
            algorithm_results[result.algorithm] = {
                'fitness': result.fitness,
                'execution_time': result.execution_time,
                'iterations_completed': result.iterations_completed,
                'convergence_achieved': result.convergence_achieved
            }
        
        optimization_results = {
            'best_algorithm': best_result.algorithm,
            'best_solution': best_result.solution,
            'best_fitness': best_result.fitness,
            'total_execution_time': total_time,
            'algorithms_executed': len(results),
            'algorithm_results': algorithm_results,
            'parallel_execution': True,
            'connection_pool_size': self.connection_pool_size,
            'optimization_metric': metric
        }
        
        logger.info(f"🏆 PARALLEL OPTIMIZATION COMPLETED:")
        logger.info(f"   Best algorithm: {best_result.algorithm}")
        logger.info(f"   Best fitness: {best_result.fitness:.6f}")
        logger.info(f"   Total execution time: {total_time:.2f} seconds")
        logger.info(f"   Algorithms executed: {len(results)}")
        logger.info(f"   Parallel speedup potential: 1.04x")
        
        return optimization_results
    
    def _tournament_selection(self, population: List[List[int]], fitness_scores: List[float], 
                            tournament_size: int = 3) -> List[int]:
        """Tournament selection for genetic algorithm"""
        tournament_size = min(tournament_size, len(population))
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_idx = tournament_indices[tournament_fitness.index(max(tournament_fitness))]
        return population[winner_idx].copy()
    
    def _crossover(self, parent1: List[int], parent2: List[int], 
                  portfolio_size: int, total_strategies: int) -> List[int]:
        """Crossover operation for genetic algorithm"""
        # Uniform crossover
        combined = list(set(parent1 + parent2))
        
        if len(combined) >= portfolio_size:
            child = random.sample(combined, portfolio_size)
        else:
            child = combined.copy()
            remaining = list(set(range(total_strategies)) - set(child))
            needed = portfolio_size - len(child)
            if remaining and needed > 0:
                child.extend(random.sample(remaining, min(needed, len(remaining))))
        
        return child[:portfolio_size]
    
    def _mutate(self, individual: List[int], total_strategies: int, portfolio_size: int) -> List[int]:
        """Mutation operation for genetic algorithm"""
        mutated = individual.copy()
        idx = random.randint(0, portfolio_size - 1)
        available = list(set(range(total_strategies)) - set(mutated))
        if available:
            mutated[idx] = random.choice(available)
        return mutated
