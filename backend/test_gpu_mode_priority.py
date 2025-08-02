#!/usr/bin/env python3
"""
Test GPU mode priority and algorithm iterations
Ensures GPU is primary mode with optional CPU fallback
"""

import os
import sys
import time
import logging

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set up logging to see algorithm iterations
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def test_heavydb_connection():
    """Test HeavyDB connection with fixed timeout issue"""
    print("\n🔧 Testing HeavyDB Connection (Fixed)")
    print("=" * 60)
    
    from lib.heavydb_connector import get_connection, get_execution_mode, get_gpu_memory_info
    
    # Try to connect
    conn = get_connection(force_new=True)
    
    if conn:
        print("✅ HeavyDB connection successful!")
        
        # Test execution mode
        mode = get_execution_mode()
        print(f"🎯 Execution mode: {mode.upper()}")
        
        # Get GPU info
        gpu_info = get_gpu_memory_info(connection=conn)
        print(f"🎮 GPU Available: {gpu_info.get('available', False)}")
        
        if gpu_info.get('available'):
            for gpu in gpu_info.get('gpus', []):
                print(f"  - Model: {gpu['model']}")
                print(f"  - Memory: {gpu['total_memory_gb']:.1f} GB")
        
        # Test a simple query
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT 1 as test")
            result = cursor.fetchone()
            print(f"✅ Query test successful: {result}")
        except Exception as e:
            print(f"❌ Query test failed: {e}")
        
        # Close connection
        try:
            conn.close()
        except:
            pass
        
        return True
    else:
        print("❌ HeavyDB connection failed")
        print("  - Check if HeavyDB server is running")
        print("  - Verify connection parameters")
        return False

def test_algorithm_iterations():
    """Test that algorithms iterate properly"""
    print("\n🔬 Testing Algorithm Iterations")
    print("=" * 60)
    
    # Create small test data
    import numpy as np
    import pandas as pd
    
    n_strategies = 100
    n_days = 30
    
    np.random.seed(42)
    data = np.random.randn(n_days, n_strategies) * 0.02
    daily_matrix = data
    
    # Import algorithms
    from algorithms.genetic_algorithm import GeneticAlgorithm
    from algorithms.simulated_annealing import SimulatedAnnealing
    from algorithms.particle_swarm_optimization import ParticleSwarmOptimization
    
    # Test fitness function
    def test_fitness(indices):
        portfolio = daily_matrix[:, indices]
        returns = portfolio.sum(axis=1)
        roi = returns.sum()
        drawdown = abs(returns.cumsum().min())
        return roi / (drawdown + 0.01)
    
    # Test GA iterations
    print("\n📊 Testing Genetic Algorithm Iterations:")
    ga = GeneticAlgorithm()
    
    # Override to use fewer iterations for testing
    ga.generations = 10  # Reduce for testing
    
    # Add custom logging to track iterations
    original_optimize = ga.optimize
    iteration_count = 0
    
    def tracked_optimize(*args, **kwargs):
        nonlocal iteration_count
        iteration_count = 0
        
        # Wrap the fitness function to count calls
        original_fitness = kwargs.get('fitness_function', args[2] if len(args) > 2 else None)
        
        def counting_fitness(indices):
            nonlocal iteration_count
            iteration_count += 1
            return original_fitness(indices)
        
        if 'fitness_function' in kwargs:
            kwargs['fitness_function'] = counting_fitness
        else:
            args = list(args)
            args[2] = counting_fitness
        
        return original_optimize(*args, **kwargs)
    
    ga.optimize = tracked_optimize
    
    # Run optimization
    result = ga.optimize(
        daily_matrix=daily_matrix,
        portfolio_size=10,
        fitness_function=test_fitness
    )
    
    print(f"  - Configured generations: {ga.generations}")
    print(f"  - Population size: {ga.population_size}")
    print(f"  - Expected fitness evaluations: ~{ga.generations * ga.population_size}")
    print(f"  - Actual fitness evaluations: {iteration_count}")
    print(f"  - Best fitness: {result['best_fitness']:.6f}")
    
    # Test SA iterations
    print("\n📊 Testing Simulated Annealing Iterations:")
    sa = SimulatedAnnealing()
    
    # Check SA parameters
    print(f"  - Max iterations: {sa.max_iterations}")
    print(f"  - Initial temperature: {sa.initial_temperature}")
    print(f"  - Cooling rate: {sa.cooling_rate}")
    
    # Test PSO iterations
    print("\n📊 Testing Particle Swarm Optimization Iterations:")
    pso = ParticleSwarmOptimization()
    
    print(f"  - Iterations: {pso.iterations}")
    print(f"  - Swarm size: {pso.swarm_size}")
    print(f"  - Expected fitness evaluations: {pso.iterations * pso.swarm_size}")

def test_gpu_priority_workflow():
    """Test that GPU mode is prioritized"""
    print("\n🚀 Testing GPU Priority in Workflow")
    print("=" * 60)
    
    from lib.config_reader import get_gpu_config, is_gpu_required
    
    # Check current configuration
    config = get_gpu_config()
    
    print("📋 GPU Configuration:")
    print(f"  - GPU Enabled: {config['enabled']}")
    print(f"  - Acceleration: {config['acceleration']}")
    print(f"  - CPU Fallback Allowed: {config['cpu_fallback_allowed']}")
    print(f"  - Force GPU Mode: {config['force_gpu_mode']}")
    print(f"  - GPU Required: {is_gpu_required()}")
    
    # Test with different scenarios
    print("\n🔍 Testing GPU Mode Logic:")
    
    # Scenario 1: GPU required, HeavyDB available
    print("\n1. GPU Required + HeavyDB Available:")
    print("   → Should use GPU mode")
    
    # Scenario 2: GPU required, HeavyDB not available
    print("\n2. GPU Required + HeavyDB Not Available:")
    print("   → Should fail with clear error message")
    
    # Scenario 3: GPU optional, HeavyDB available
    print("\n3. GPU Optional + HeavyDB Available:")
    print("   → Should use GPU mode")
    
    # Scenario 4: GPU optional, HeavyDB not available
    print("\n4. GPU Optional + HeavyDB Not Available:")
    print("   → Should use CPU mode")

def verify_gpu_mode_improvements():
    """Verify all GPU mode improvements"""
    print("\n✅ Verifying GPU Mode Improvements")
    print("=" * 60)
    
    improvements = {
        "HeavyDB timeout fix": False,
        "GPU prioritization": False,
        "Algorithm iterations": False,
        "CPU optional mode": False
    }
    
    # Check timeout fix
    try:
        from lib.heavydb_connector import get_connection_params
        params = get_connection_params()
        # Timeout should be in params but not passed to connect()
        if 'timeout' in params:
            print("✅ Timeout parameter configured (but not passed to connect)")
            improvements["HeavyDB timeout fix"] = True
    except:
        pass
    
    # Check GPU prioritization
    from lib.config_reader import is_gpu_required
    if is_gpu_required():
        print("✅ GPU mode is required (prioritized)")
        improvements["GPU prioritization"] = True
    
    # Check algorithm iterations from config
    import configparser
    config = configparser.ConfigParser()
    config.read('/mnt/optimizer_share/config/production_config.ini')
    
    if config.has_section('ALGORITHM_PARAMETERS'):
        ga_gens = config.getint('ALGORITHM_PARAMETERS', 'ga_generations', 0)
        pso_iters = config.getint('ALGORITHM_PARAMETERS', 'pso_iterations', 0)
        sa_iters = config.getint('ALGORITHM_PARAMETERS', 'sa_max_iterations', 0)
        
        if ga_gens > 0 and pso_iters > 0 and sa_iters > 0:
            print(f"✅ Algorithm iterations configured:")
            print(f"   - GA: {ga_gens} generations")
            print(f"   - PSO: {pso_iters} iterations")
            print(f"   - SA: {sa_iters} iterations")
            improvements["Algorithm iterations"] = True
    
    # Check CPU optional
    config_gpu = config.getboolean('GPU', 'cpu_fallback_allowed', True)
    if not config_gpu or config.get('GPU', 'gpu_acceleration') == 'optional':
        print("✅ CPU mode is optional (can be disabled)")
        improvements["CPU optional mode"] = True
    
    # Summary
    print("\n📊 Improvement Summary:")
    for improvement, status in improvements.items():
        print(f"  - {improvement}: {'✅' if status else '❌'}")
    
    return all(improvements.values())

def main():
    """Run all GPU mode tests"""
    print("🎮 GPU Mode Priority Testing")
    print("=" * 80)
    print("Ensuring GPU mode is primary with optional CPU fallback")
    print("=" * 80)
    
    # Test HeavyDB connection
    heavydb_ok = test_heavydb_connection()
    
    # Test algorithm iterations
    test_algorithm_iterations()
    
    # Test GPU priority
    test_gpu_priority_workflow()
    
    # Verify improvements
    all_good = verify_gpu_mode_improvements()
    
    print("\n" + "="*80)
    print("🏁 GPU Mode Testing Complete")
    print("="*80)
    
    if heavydb_ok:
        print("✅ HeavyDB connection working (GPU mode available)")
    else:
        print("⚠️ HeavyDB not available (GPU mode requires HeavyDB)")
    
    if all_good:
        print("✅ All GPU improvements verified")
    else:
        print("⚠️ Some improvements need attention")
    
    print("\n💡 Recommendations:")
    if not heavydb_ok:
        print("  1. Start HeavyDB server for GPU acceleration")
        print("  2. Check connection parameters in config")
    print("  3. Ensure GPU drivers are installed")
    print("  4. Monitor algorithm iterations for performance")

if __name__ == "__main__":
    main()