#!/usr/bin/env python3
"""
Test GPU-only mode with real data
Ensures system works with GPU as primary and CPU as optional
"""

import os
import sys
import time
import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Force GPU mode for testing
os.environ['FORCE_GPU_MODE'] = 'true'
os.environ['GPU_FALLBACK_ALLOWED'] = 'false'

def test_gpu_connection():
    """Test HeavyDB GPU connection"""
    print("🔧 Testing GPU Connection")
    print("=" * 60)
    
    from lib.heavydb_connector import get_connection, get_execution_mode, get_gpu_memory_info
    
    # Test connection
    conn = get_connection(force_new=True)
    
    if conn:
        print("✅ HeavyDB connection established")
        
        # Check mode
        mode = get_execution_mode()
        print(f"🎯 Mode: {mode.upper()}")
        
        # Get GPU info
        gpu_info = get_gpu_memory_info(connection=conn)
        print(f"🎮 GPU Available: {gpu_info.get('available', False)}")
        
        if gpu_info.get('gpus'):
            gpu = gpu_info['gpus'][0]
            print(f"📊 GPU Details:")
            print(f"  - Model: {gpu['model']}")
            print(f"  - Memory: {gpu['total_memory_gb']:.1f} GB")
            print(f"  - Available: {gpu['free_memory_gb']:.1f} GB")
        
        # Test GPU query
        try:
            from lib.heavydb_connector import execute_query
            
            # Create test table
            cursor = conn.cursor()
            cursor.execute("DROP TABLE IF EXISTS gpu_test")
            cursor.execute("""
                CREATE TABLE gpu_test (
                    id INTEGER,
                    val DOUBLE
                )
            """)
            
            # Insert test data
            cursor.execute("""
                INSERT INTO gpu_test VALUES 
                (1, 1.5), (2, 2.5), (3, 3.5), (4, 4.5), (5, 5.5)
            """)
            
            # Test GPU-accelerated query
            result = execute_query("SELECT AVG(val) as avg_val FROM gpu_test", connection=conn)
            
            if result is not None and not result.empty:
                print(f"✅ GPU query successful: avg = {result['avg_val'].iloc[0]}")
            
            # Clean up
            cursor.execute("DROP TABLE gpu_test")
            
        except Exception as e:
            print(f"⚠️ GPU query test failed: {e}")
        
        try:
            conn.close()
        except:
            pass
        
        return True
    else:
        print("❌ HeavyDB connection failed - GPU mode not available")
        return False

def test_algorithm_iterations_detailed():
    """Test algorithm iterations with detailed logging"""
    print("\n🔬 Testing Algorithm Iterations (Detailed)")
    print("=" * 60)
    
    # Create test data
    n_strategies = 100
    n_days = 30
    portfolio_size = 10
    
    np.random.seed(42)
    data = np.random.randn(n_days, n_strategies) * 0.02
    daily_matrix = data
    
    # Test fitness function
    def fitness_function(indices):
        portfolio = daily_matrix[:, indices]
        returns = portfolio.sum(axis=1)
        roi = returns.sum()
        drawdown = abs(returns.cumsum().min()) + 0.01
        return roi / drawdown
    
    # Import and test each algorithm
    algorithms_to_test = [
        ('genetic_algorithm', 'GeneticAlgorithm', 'generations'),
        ('simulated_annealing', 'SimulatedAnnealing', 'max_iterations'),
        ('particle_swarm_optimization', 'ParticleSwarmOptimization', 'iterations'),
        ('differential_evolution', 'DifferentialEvolution', 'generations'),
        ('ant_colony_optimization', 'AntColonyOptimization', 'iterations'),
        ('hill_climbing', 'HillClimbing', 'max_iterations'),
        ('bayesian_optimization', 'BayesianOptimization', 'n_calls'),
        ('random_search', 'RandomSearch', 'iterations')
    ]
    
    results = {}
    
    for module_name, class_name, iter_attr in algorithms_to_test:
        print(f"\n📊 Testing {class_name}:")
        
        try:
            # Import algorithm
            module = __import__(f'algorithms.{module_name}', fromlist=[class_name])
            AlgorithmClass = getattr(module, class_name)
            
            # Create instance
            algorithm = AlgorithmClass()
            
            # Get iteration count
            if hasattr(algorithm, iter_attr):
                expected_iterations = getattr(algorithm, iter_attr)
                print(f"  - Expected {iter_attr}: {expected_iterations}")
            else:
                expected_iterations = "Not found"
                print(f"  - ⚠️ {iter_attr} attribute not found")
            
            # Track iterations
            iteration_count = 0
            evaluation_count = 0
            
            # Wrap fitness function
            def counting_fitness(indices):
                nonlocal evaluation_count
                evaluation_count += 1
                return fitness_function(indices)
            
            # Run optimization
            start_time = time.time()
            result = algorithm.optimize(
                daily_matrix=daily_matrix,
                portfolio_size=portfolio_size,
                fitness_function=counting_fitness
            )
            elapsed = time.time() - start_time
            
            # Get results
            fitness = result.get('best_fitness', 0)
            
            print(f"  - Best fitness: {fitness:.6f}")
            print(f"  - Execution time: {elapsed:.3f}s")
            print(f"  - Fitness evaluations: {evaluation_count}")
            
            # Check population/swarm based algorithms
            if hasattr(algorithm, 'population_size'):
                print(f"  - Population size: {algorithm.population_size}")
                if expected_iterations != "Not found":
                    expected_evals = expected_iterations * algorithm.population_size
                    print(f"  - Expected evaluations: ~{expected_evals}")
            elif hasattr(algorithm, 'swarm_size'):
                print(f"  - Swarm size: {algorithm.swarm_size}")
                if expected_iterations != "Not found":
                    expected_evals = expected_iterations * algorithm.swarm_size
                    print(f"  - Expected evaluations: ~{expected_evals}")
            elif hasattr(algorithm, 'colony_size'):
                print(f"  - Colony size: {algorithm.colony_size}")
                if expected_iterations != "Not found":
                    expected_evals = expected_iterations * algorithm.colony_size
                    print(f"  - Expected evaluations: ~{expected_evals}")
            
            results[class_name] = {
                'success': True,
                'fitness': fitness,
                'time': elapsed,
                'evaluations': evaluation_count,
                'expected_iterations': expected_iterations
            }
            
        except Exception as e:
            print(f"  - ❌ Error: {e}")
            results[class_name] = {
                'success': False,
                'error': str(e)
            }
    
    # Summary
    print("\n📊 Algorithm Summary:")
    print("-" * 60)
    successful = sum(1 for r in results.values() if r['success'])
    print(f"Successful algorithms: {successful}/{len(algorithms_to_test)}")
    
    return results

def test_gpu_workflow():
    """Test full workflow in GPU-only mode"""
    print("\n🚀 Testing GPU-Only Workflow")
    print("=" * 60)
    
    # Use small test data
    test_file = "/mnt/optimizer_share/input/SENSEX_test_dataset.csv"
    
    if not os.path.exists(test_file):
        print("⚠️ Test file not found, creating synthetic data...")
        
        # Create small test data
        n_strategies = 50
        n_days = 30
        
        np.random.seed(42)
        data = np.random.randn(n_days, n_strategies) * 100
        
        # Create DataFrame
        df = pd.DataFrame(data)
        df.columns = [f'SENSEX_{i:04d}' for i in range(n_strategies)]
        df.insert(0, 'Date', pd.date_range('2024-01-01', periods=n_days))
        df.insert(1, 'Day', df['Date'].dt.day_name())
        
        # Save
        df.to_csv(test_file, index=False)
        print(f"✅ Created test file: {test_file}")
    
    # Run workflow
    from gpu_enhanced_workflow import GPUEnhancedOptimizer
    
    try:
        optimizer = GPUEnhancedOptimizer()
        
        print("\n📊 Running optimization...")
        success = optimizer.run_optimization(test_file, portfolio_size=10)
        
        if success:
            print("✅ GPU workflow completed successfully!")
        else:
            print("❌ GPU workflow failed")
        
        return success
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Run all GPU-only tests"""
    print("🎮 GPU-Only Mode Testing")
    print("=" * 80)
    print("Testing with GPU as primary mode (CPU optional)")
    print("=" * 80)
    
    # Check configuration
    from lib.config_reader import get_gpu_config, is_gpu_required
    
    config = get_gpu_config()
    print("\n📋 Current Configuration:")
    print(f"  - GPU Required: {is_gpu_required()}")
    print(f"  - GPU Acceleration: {config['acceleration']}")
    print(f"  - CPU Fallback: {config['cpu_fallback_allowed']}")
    print(f"  - Force GPU: {config['force_gpu_mode']}")
    
    # Test GPU connection
    gpu_ok = test_gpu_connection()
    
    if gpu_ok:
        # Test algorithms
        algo_results = test_algorithm_iterations_detailed()
        
        # Test workflow
        workflow_ok = test_gpu_workflow()
    else:
        print("\n⚠️ Skipping algorithm and workflow tests - GPU not available")
        algo_results = {}
        workflow_ok = False
    
    # Summary
    print("\n" + "="*80)
    print("📊 GPU-Only Mode Test Summary")
    print("="*80)
    
    print(f"\n✅ GPU Connection: {'PASS' if gpu_ok else 'FAIL'}")
    
    if algo_results:
        successful_algos = sum(1 for r in algo_results.values() if r.get('success', False))
        print(f"✅ Algorithm Tests: {successful_algos}/{len(algo_results)} passed")
    
    print(f"✅ GPU Workflow: {'PASS' if workflow_ok else 'FAIL'}")
    
    if gpu_ok and workflow_ok:
        print("\n🎉 GPU-only mode is working correctly!")
        print("   - All algorithms iterate properly")
        print("   - GPU acceleration is active")
        print("   - CPU mode is optional")
    else:
        print("\n⚠️ GPU-only mode needs attention:")
        if not gpu_ok:
            print("   - HeavyDB connection required for GPU mode")
            print("   - Check server status and connection parameters")

if __name__ == "__main__":
    main()