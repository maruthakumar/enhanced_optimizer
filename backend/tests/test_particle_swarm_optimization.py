#!/usr/bin/env python3
"""
Unit Tests for Particle Swarm Optimization

Tests the modular PSO implementation including:
- Basic functionality
- Swarm dynamics
- Velocity updates
- Configuration parameters
"""

import unittest
import numpy as np
from pathlib import Path

# Add backend to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from algorithms.particle_swarm_optimization import ParticleSwarmOptimization
from tests.test_base_algorithm import BaseAlgorithmTest


class TestParticleSwarmOptimization(BaseAlgorithmTest):
    """Test cases for Particle Swarm Optimization"""
    
    def test_basic_functionality_small_data(self):
        """Test PSO with small dataset"""
        result = self.run_algorithm_test(
            ParticleSwarmOptimization,
            "PSO - Small Data",
            self.small_data,
            self.small_portfolio
        )
        
        # PSO should find a reasonable solution
        self.assertGreater(result['best_fitness'], -1.0)
        self.assertLess(result['best_fitness'], 10.0)
    
    def test_basic_functionality_medium_data(self):
        """Test PSO with medium dataset"""
        result = self.run_algorithm_test(
            ParticleSwarmOptimization,
            "PSO - Medium Data",
            self.medium_data,
            self.medium_portfolio
        )
        
        # Check algorithm-specific metrics
        self.assertIn('iterations', result)
        self.assertIn('swarm_size', result)
        self.assertIn('iteration_stats', result)
    
    def test_basic_functionality_large_data(self):
        """Test PSO with large dataset"""
        result = self.run_algorithm_test(
            ParticleSwarmOptimization,
            "PSO - Large Data",
            self.large_data,
            self.large_portfolio
        )
        
        # Verify iteration statistics
        iter_stats = result.get('iteration_stats', [])
        self.assertGreater(len(iter_stats), 0)
        
        # Check that best fitness improves
        if len(iter_stats) > 1:
            first_iter_fitness = iter_stats[0]['best_score']
            last_iter_fitness = iter_stats[-1]['best_score']
            self.assertGreaterEqual(last_iter_fitness, first_iter_fitness)
    
    def test_configuration_parameters(self):
        """Test PSO with custom configuration"""
        import tempfile
        import os
        
        custom_config = """[PSO]
swarm_size = 15
iterations = 40
inertia = 0.9
acceleration_coef_1 = 2.0
acceleration_coef_2 = 2.0
velocity_max = 5.0
"""
        fd, config_path = tempfile.mkstemp(suffix='.ini')
        with os.fdopen(fd, 'w') as f:
            f.write(custom_config)
        
        try:
            # Create algorithm with custom config
            pso = ParticleSwarmOptimization(config_path)
            
            # Verify parameters were loaded
            self.assertEqual(pso.swarm_size, 15)
            self.assertEqual(pso.iterations, 40)
            self.assertEqual(pso.inertia, 0.9)
            
            # Run optimization
            result = pso.optimize(
                daily_matrix=self.small_data,
                portfolio_size=self.small_portfolio,
                fitness_function=self.calculate_fitness
            )
            
            # Check that custom parameters were used
            self.assertEqual(result['iterations'], 40)
            self.assertEqual(result['swarm_size'], 15)
            
        finally:
            os.unlink(config_path)
    
    def test_edge_case_empty_data(self):
        """Test PSO with empty data"""
        self.test_edge_case_empty_data(ParticleSwarmOptimization)
    
    def test_edge_case_single_strategy(self):
        """Test PSO with single strategy"""
        self.test_edge_case_single_strategy(ParticleSwarmOptimization)
    
    def test_edge_case_portfolio_equals_strategies(self):
        """Test PSO when portfolio size equals number of strategies"""
        self.test_edge_case_portfolio_equals_strategies(ParticleSwarmOptimization)
    
    def test_zone_optimization(self):
        """Test PSO with zone constraints"""
        self.test_zone_optimization(ParticleSwarmOptimization)
    
    def test_variable_portfolio_size(self):
        """Test PSO with variable portfolio size"""
        self.test_variable_portfolio_size(ParticleSwarmOptimization)
    
    def test_swarm_initialization(self):
        """Test that swarm is properly initialized"""
        pso = ParticleSwarmOptimization()
        
        # Initialize swarm
        particles, velocities = pso._initialize_swarm(
            num_strategies=20,
            portfolio_size=5
        )
        
        # Check swarm size
        self.assertEqual(len(particles), pso.swarm_size)
        self.assertEqual(len(velocities), pso.swarm_size)
        
        # Check particle properties
        for particle in particles:
            self.assertEqual(len(particle), 5)
            self.assertEqual(len(set(particle)), 5)  # No duplicates
            self.assertTrue(all(0 <= s < 20 for s in particle))
        
        # Check velocity properties
        for velocity in velocities:
            self.assertEqual(len(velocity), 5)
            self.assertTrue(all(-pso.velocity_max <= v <= pso.velocity_max for v in velocity))
    
    def test_velocity_update(self):
        """Test PSO velocity update equations"""
        pso = ParticleSwarmOptimization()
        
        # Create test data
        particles = [np.array([0, 1, 2, 3, 4]), np.array([5, 6, 7, 8, 9])]
        velocities = [np.random.uniform(-1, 1, 5), np.random.uniform(-1, 1, 5)]
        personal_best = [np.array([1, 2, 3, 4, 5]), np.array([6, 7, 8, 9, 10])]
        global_best = np.array([2, 3, 4, 5, 6])
        
        # Update swarm
        new_particles, new_velocities = pso._update_swarm(
            particles, velocities, personal_best, global_best, 15, None
        )
        
        # Check that velocities are bounded
        for velocity in new_velocities:
            self.assertTrue(all(-pso.velocity_max <= v <= pso.velocity_max for v in velocity))
        
        # Check that particles are valid
        for particle in new_particles:
            self.assertEqual(len(particle), 5)
            self.assertEqual(len(set(particle)), 5)  # No duplicates
    
    def test_discrete_position_update(self):
        """Test discrete position update mechanism"""
        pso = ParticleSwarmOptimization()
        
        position = np.array([0, 1, 2, 3, 4])
        velocity = np.array([2.0, -1.0, 0.5, -0.5, 1.5])
        
        # Update position
        new_position = pso._update_discrete_position(position, velocity, 10)
        
        # Check validity
        self.assertEqual(len(new_position), 5)
        self.assertEqual(len(set(new_position)), 5)  # No duplicates
        self.assertTrue(all(0 <= s < 10 for s in new_position))
    
    def test_convergence_behavior(self):
        """Test PSO convergence characteristics"""
        pso = ParticleSwarmOptimization()
        pso.iterations = 50
        
        result = pso.optimize(
            daily_matrix=self.medium_data,
            portfolio_size=10,
            fitness_function=self.calculate_fitness
        )
        
        iter_stats = result['iteration_stats']
        
        # Check that average fitness improves
        early_avg = np.mean([s['avg_score'] for s in iter_stats[:10]])
        late_avg = np.mean([s['avg_score'] for s in iter_stats[-10:]])
        
        self.assertGreater(late_avg, early_avg,
                         "Average swarm fitness did not improve")
        
        # Check that standard deviation decreases (convergence)
        early_std = np.mean([s['std_score'] for s in iter_stats[:10]])
        late_std = np.mean([s['std_score'] for s in iter_stats[-10:]])
        
        # Allow some tolerance for stochastic behavior
        self.assertLess(late_std, early_std * 1.5,
                       "Swarm not converging properly")
    
    def test_inertia_effect(self):
        """Test effect of inertia parameter"""
        # High inertia PSO
        pso_high_inertia = ParticleSwarmOptimization()
        pso_high_inertia.inertia = 0.9
        pso_high_inertia.iterations = 30
        
        # Low inertia PSO
        pso_low_inertia = ParticleSwarmOptimization()
        pso_low_inertia.inertia = 0.4
        pso_low_inertia.iterations = 30
        
        # Run both
        result_high = pso_high_inertia.optimize(
            daily_matrix=self.small_data,
            portfolio_size=5,
            fitness_function=self.calculate_fitness
        )
        
        result_low = pso_low_inertia.optimize(
            daily_matrix=self.small_data,
            portfolio_size=5,
            fitness_function=self.calculate_fitness
        )
        
        # Both should produce valid results
        self.validate_algorithm_result(result_high, "PSO-HighInertia")
        self.validate_algorithm_result(result_low, "PSO-LowInertia")
        
        # High inertia might explore more (higher variance in early iterations)
        # Low inertia might converge faster
        # Just check both work properly
        self.assertGreater(result_high['best_fitness'], -10)
        self.assertGreater(result_low['best_fitness'], -10)
    
    def test_personal_vs_global_best_tracking(self):
        """Test that PSO properly tracks personal and global bests"""
        pso = ParticleSwarmOptimization()
        
        result = pso.optimize(
            daily_matrix=self.medium_data,
            portfolio_size=10,
            fitness_function=self.calculate_fitness
        )
        
        iter_stats = result['iteration_stats']
        
        # Global best should never decrease
        global_best_values = [s['best_score'] for s in iter_stats]
        for i in range(1, len(global_best_values)):
            self.assertGreaterEqual(global_best_values[i], global_best_values[i-1],
                                  "Global best fitness decreased")
        
        # Final best should match last iteration's best
        self.assertEqual(result['best_fitness'], iter_stats[-1]['best_score'])
    
    def test_zone_constraints_with_swarm(self):
        """Test PSO respects zone constraints throughout optimization"""
        pso = ParticleSwarmOptimization()
        
        zone_data = {
            'allowed_strategies': list(range(5, 15)),
            'min_strategies_per_zone': 3
        }
        
        result = pso.optimize(
            daily_matrix=self.medium_data,
            portfolio_size=5,
            fitness_function=self.calculate_fitness,
            zone_data=zone_data
        )
        
        # Final portfolio should respect zone constraints
        portfolio = result['best_portfolio']
        zone_strategies = [s for s in portfolio if s in zone_data['allowed_strategies']]
        
        self.assertGreaterEqual(len(zone_strategies), zone_data['min_strategies_per_zone'])


class TestPSOComparison(BaseAlgorithmTest):
    """Comparison tests for PSO behavior"""
    
    def test_swarm_intelligence_behavior(self):
        """Test that PSO exhibits swarm intelligence characteristics"""
        pso = ParticleSwarmOptimization()
        
        # Run with detailed stats
        result = pso.optimize(
            daily_matrix=self.medium_data,
            portfolio_size=10,
            fitness_function=self.calculate_fitness
        )
        
        iter_stats = result['iteration_stats']
        
        # Check swarm behavior: average fitness should improve faster than random
        improvements = []
        for i in range(1, len(iter_stats)):
            improvement = iter_stats[i]['avg_score'] - iter_stats[i-1]['avg_score']
            improvements.append(improvement)
        
        # Most iterations should show improvement or small decline
        positive_improvements = sum(1 for imp in improvements if imp > -0.1)
        improvement_rate = positive_improvements / len(improvements)
        
        self.assertGreater(improvement_rate, 0.5,
                         "Swarm not showing collaborative improvement")


if __name__ == '__main__':
    unittest.main()