"""
Algorithm iteration monitoring module
Ensures algorithms iterate properly with logging
"""

import logging
import time
from typing import Dict, Any, Callable
import functools

# Set up algorithm monitoring logger
logger = logging.getLogger('algorithm_monitor')
logger.setLevel(logging.INFO)

# Add console handler if not already present
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class AlgorithmMonitor:
    """Monitor algorithm iterations and performance"""
    
    def __init__(self):
        self.iteration_counts = {}
        self.fitness_evaluations = {}
        self.start_times = {}
        self.parameters = {}
    
    def log_algorithm_start(self, algorithm_name: str, parameters: Dict[str, Any]):
        """Log when an algorithm starts"""
        self.start_times[algorithm_name] = time.time()
        self.iteration_counts[algorithm_name] = 0
        self.fitness_evaluations[algorithm_name] = 0
        self.parameters[algorithm_name] = parameters
        
        logger.info(f"🚀 {algorithm_name} started with parameters:")
        for key, value in parameters.items():
            logger.info(f"   - {key}: {value}")
    
    def log_iteration(self, algorithm_name: str, iteration: int, best_fitness: float = None):
        """Log algorithm iteration"""
        self.iteration_counts[algorithm_name] = iteration
        
        if best_fitness is not None:
            logger.debug(f"   {algorithm_name} - Iteration {iteration}: Best fitness = {best_fitness:.6f}")
    
    def log_fitness_evaluation(self, algorithm_name: str):
        """Count fitness function evaluations"""
        if algorithm_name not in self.fitness_evaluations:
            self.fitness_evaluations[algorithm_name] = 0
        self.fitness_evaluations[algorithm_name] += 1
    
    def log_algorithm_complete(self, algorithm_name: str, result: Dict[str, Any]):
        """Log when an algorithm completes"""
        elapsed = time.time() - self.start_times.get(algorithm_name, 0)
        iterations = self.iteration_counts.get(algorithm_name, 0)
        evaluations = self.fitness_evaluations.get(algorithm_name, 0)
        
        logger.info(f"✅ {algorithm_name} completed:")
        logger.info(f"   - Total iterations: {iterations}")
        logger.info(f"   - Fitness evaluations: {evaluations}")
        logger.info(f"   - Execution time: {elapsed:.2f}s")
        logger.info(f"   - Best fitness: {result.get('best_fitness', 0):.6f}")
        
        # Check if iterations match expected
        params = self.parameters.get(algorithm_name, {})
        expected_iterations = params.get('expected_iterations', 0)
        
        if expected_iterations > 0 and iterations < expected_iterations * 0.9:
            logger.warning(f"⚠️ {algorithm_name} completed with fewer iterations than expected ({iterations}/{expected_iterations})")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all algorithms"""
        return {
            'iteration_counts': self.iteration_counts,
            'fitness_evaluations': self.fitness_evaluations,
            'parameters': self.parameters
        }

# Global monitor instance
_monitor = AlgorithmMonitor()

def monitor_algorithm(algorithm_name: str, expected_iterations: int = None):
    """Decorator to monitor algorithm execution"""
    def decorator(optimize_func: Callable):
        @functools.wraps(optimize_func)
        def wrapper(self, *args, **kwargs):
            # Get algorithm parameters
            params = {
                'expected_iterations': expected_iterations or getattr(self, 'iterations', getattr(self, 'generations', getattr(self, 'max_iterations', 0))),
                'population_size': getattr(self, 'population_size', getattr(self, 'swarm_size', getattr(self, 'colony_size', 1))),
            }
            
            # Add specific parameters
            if hasattr(self, 'mutation_rate'):
                params['mutation_rate'] = self.mutation_rate
            if hasattr(self, 'crossover_rate'):
                params['crossover_rate'] = self.crossover_rate
            if hasattr(self, 'temperature'):
                params['initial_temperature'] = getattr(self, 'initial_temperature', self.temperature)
            
            _monitor.log_algorithm_start(algorithm_name, params)
            
            # Wrap fitness function to count evaluations
            fitness_func = kwargs.get('fitness_function', args[2] if len(args) > 2 else None)
            
            def monitored_fitness(indices):
                _monitor.log_fitness_evaluation(algorithm_name)
                return fitness_func(indices)
            
            # Replace fitness function
            if 'fitness_function' in kwargs:
                kwargs['fitness_function'] = monitored_fitness
            else:
                args = list(args)
                args[2] = monitored_fitness
            
            # Run algorithm
            result = optimize_func(self, *args, **kwargs)
            
            # Log completion
            _monitor.log_algorithm_complete(algorithm_name, result)
            
            return result
        
        return wrapper
    return decorator

def get_algorithm_monitor():
    """Get the global algorithm monitor instance"""
    return _monitor

# Utility function to verify algorithm iterations
def verify_algorithm_iterations(algorithm_name: str, min_iterations: int) -> bool:
    """Verify an algorithm ran sufficient iterations"""
    actual = _monitor.iteration_counts.get(algorithm_name, 0)
    if actual >= min_iterations:
        logger.info(f"✅ {algorithm_name} iterations verified: {actual} >= {min_iterations}")
        return True
    else:
        logger.warning(f"❌ {algorithm_name} insufficient iterations: {actual} < {min_iterations}")
        return False