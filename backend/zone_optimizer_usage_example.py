#!/usr/bin/env python3
"""
Example usage of the Zone Optimizer module

Demonstrates zone-based portfolio optimization with various combination methods.
"""

import sys
import os
import logging
import pandas as pd
import numpy as np
import tempfile

# Add backend to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from zone_optimizer import ZoneOptimizer, ZoneConfiguration
from zone_optimizer_dal import ZoneOptimizerDAL
from dal import get_dal

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_sample_zone_data(n_days=100, n_strategies=50, n_zones=4):
    """Create sample data with zones"""
    np.random.seed(42)
    
    zones = [f'zone{i+1}' for i in range(n_zones)]
    
    data = {
        'Date': pd.date_range('2024-01-01', periods=n_days),
        'Time': '09:30:00',
        'Zone': np.repeat(zones, n_days // n_zones)[:n_days]
    }
    
    # Add strategy columns with zone-specific patterns
    for i in range(n_strategies):
        # Different zones have different return patterns
        returns = []
        for j, zone in enumerate(data['Zone']):
            zone_idx = zones.index(zone)
            base_return = np.random.randn() * (1 + zone_idx * 0.1)  # Zone-specific volatility
            trend = 0.001 * zone_idx * (i % 4)  # Zone-specific trends
            returns.append(base_return + trend)
        
        data[f'Strategy_{i+1}'] = pd.Series(returns).cumsum() + 1000
    
    return pd.DataFrame(data)


def demonstrate_basic_zone_optimization():
    """Demonstrate basic zone optimization"""
    logger.info("\n" + "="*60)
    logger.info("Basic Zone Optimization Demo")
    logger.info("="*60)
    
    # Create sample data
    data = create_sample_zone_data(n_days=100, n_strategies=30)
    logger.info(f"Created sample data: {len(data)} rows, {len(data.columns)} columns")
    
    # Initialize optimizer
    optimizer = ZoneOptimizer()
    
    # Extract zones
    zones = optimizer.extract_zones_from_data(data)
    logger.info(f"\nExtracted {len(zones)} zones:")
    for zone_name, zone_df in zones.items():
        logger.info(f"  {zone_name}: {len(zone_df)} rows")
    
    # Optimize each zone
    logger.info("\nOptimizing each zone independently...")
    zone_results = optimizer.optimize_all_zones(
        data, 
        portfolio_size_per_zone=5,
        algorithm="genetic"
    )
    
    # Display results
    logger.info("\nZone Optimization Results:")
    for zone_name, result in zone_results.items():
        logger.info(f"\n{zone_name.upper()}:")
        logger.info(f"  Fitness: {result.fitness_score:.4f}")
        logger.info(f"  Selected strategies: {', '.join(result.portfolio_columns[:3])}...")
        logger.info(f"  Optimization time: {result.optimization_time:.2f}s")
    
    # Combine results
    logger.info("\nCombining zone results...")
    combined = optimizer.combine_zone_results(zone_results)
    logger.info(f"Combined fitness: {combined.combined_fitness:.4f}")
    logger.info(f"Combined portfolio size: {len(combined.combined_portfolio)}")
    
    # Generate report
    with tempfile.NamedTemporaryFile(mode='w', suffix='_zone_report.txt', delete=False) as f:
        report_path = f.name
    
    optimizer.generate_zone_report(zone_results, report_path)
    logger.info(f"\nReport saved to: {report_path}")


def demonstrate_combination_methods():
    """Demonstrate different zone combination methods"""
    logger.info("\n" + "="*60)
    logger.info("Zone Combination Methods Demo")
    logger.info("="*60)
    
    # Create data
    data = create_sample_zone_data(n_days=100, n_strategies=20)
    
    # Create optimizer with custom weights
    config = ZoneConfiguration(
        zone_weights={'zone1': 0.4, 'zone2': 0.3, 'zone3': 0.2, 'zone4': 0.1}
    )
    optimizer = ZoneOptimizer()
    optimizer.config = config
    
    # Optimize zones
    zone_results = optimizer.optimize_all_zones(data, portfolio_size_per_zone=3)
    
    # Try different combination methods
    methods = ['weighted_average', 'best_zone', 'equal_weight']
    
    for method in methods:
        logger.info(f"\n{method.upper()} Combination:")
        combined = optimizer.combine_zone_results(zone_results, method=method)
        
        logger.info(f"  Combined fitness: {combined.combined_fitness:.4f}")
        logger.info(f"  Portfolio size: {len(combined.combined_portfolio)}")
        
        if method == 'best_zone':
            logger.info(f"  Selected zone: {combined.metadata.get('selected_zone')}")
        else:
            logger.info("  Zone contributions:")
            for zone, contrib in combined.zone_contributions.items():
                logger.info(f"    {zone}: {contrib:.2%}")


def demonstrate_dal_integration():
    """Demonstrate DAL integration for database operations"""
    logger.info("\n" + "="*60)
    logger.info("Zone Optimizer DAL Integration Demo")
    logger.info("="*60)
    
    # Create sample data
    data = create_sample_zone_data(n_days=120, n_strategies=40)
    
    # Save to temporary CSV
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        data.to_csv(f.name, index=False)
        csv_file = f.name
    
    try:
        # Use DAL
        with get_dal('csv') as dal:
            # Load data into "database"
            logger.info("\nLoading data into DAL...")
            dal.load_csv_to_heavydb(csv_file, 'zone_demo_table')
            
            # Create optimizer with DAL
            optimizer = ZoneOptimizerDAL(dal=dal)
            
            # Get zone statistics
            logger.info("\nZone statistics:")
            stats = optimizer.get_zone_statistics('zone_demo_table')
            if not stats.empty:
                logger.info(stats.to_string())
            
            # Optimize zones from table
            logger.info("\nOptimizing zones from database table...")
            zone_results = optimizer.optimize_zones_from_table(
                'zone_demo_table',
                portfolio_size_per_zone=5,
                save_results=True
            )
            
            # Combine and save
            logger.info("\nCombining and saving results...")
            combined = optimizer.combine_and_save_results(
                zone_results,
                'zone_combined_results'
            )
            
            logger.info(f"\nFinal combined fitness: {combined.combined_fitness:.4f}")
            
    finally:
        os.unlink(csv_file)


def demonstrate_zone_correlation_analysis():
    """Demonstrate zone correlation analysis"""
    logger.info("\n" + "="*60)
    logger.info("Zone Correlation Analysis Demo")
    logger.info("="*60)
    
    # Create data with correlated zones
    data = create_sample_zone_data(n_days=100, n_strategies=25)
    
    optimizer = ZoneOptimizer()
    
    # Optimize zones
    zone_results = optimizer.optimize_all_zones(data, portfolio_size_per_zone=8)
    
    # Analyze correlations
    correlation_analysis = optimizer.analyze_zone_correlation(zone_results)
    
    logger.info("\nZone Portfolio Correlations:")
    logger.info(f"Total unique strategies across zones: {correlation_analysis['total_unique_strategies']}")
    
    logger.info("\nStrategy overlap between zones:")
    for pair, overlap in correlation_analysis['strategy_overlap'].items():
        logger.info(f"  {pair}: {overlap:.2%} similarity")
    
    # Show similarity matrix
    logger.info("\nZone similarity matrix:")
    similarity_matrix = correlation_analysis['zone_similarity_matrix']
    zone_names = correlation_analysis['zone_names']
    
    # Format as table
    logger.info("        " + "  ".join([f"{z:>6}" for z in zone_names]))
    for i, zone in enumerate(zone_names):
        row = f"{zone:>6}: "
        row += "  ".join([f"{similarity_matrix[i,j]:>6.2f}" for j in range(len(zone_names))])
        logger.info(row)


def demonstrate_configuration_options():
    """Demonstrate various configuration options"""
    logger.info("\n" + "="*60)
    logger.info("Configuration Options Demo")
    logger.info("="*60)
    
    # Create config file
    config_content = """
[ZONE_SPECIFIC_OPTIMIZATION]
enable = True
min_size = 3
max_size = 10
population_size = 50
mutation_rate = 0.15

[OPTIMIZATION]
apply_ulta_logic = True
zone_weights = 0.35,0.25,0.25,0.15

[ALGORITHMS]
use_genetic_algorithm = True
use_particle_swarm = False
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
        f.write(config_content)
        config_file = f.name
    
    try:
        # Create optimizer with config
        optimizer = ZoneOptimizer(config_path=config_file)
        
        logger.info("\nLoaded configuration:")
        logger.info(f"  Min portfolio size: {optimizer.config.min_portfolio_size}")
        logger.info(f"  Max portfolio size: {optimizer.config.max_portfolio_size}")
        logger.info(f"  Apply ULTA: {optimizer.config.apply_ulta}")
        logger.info(f"  Zone weights: {optimizer.config.zone_weights}")
        
        # Run optimization with config
        data = create_sample_zone_data(n_days=80, n_strategies=20)
        
        zone_results = optimizer.optimize_all_zones(data)
        
        logger.info("\nOptimization completed with custom configuration")
        
    finally:
        os.unlink(config_file)


if __name__ == '__main__':
    # Run all demonstrations
    demonstrate_basic_zone_optimization()
    demonstrate_combination_methods()
    demonstrate_dal_integration()
    demonstrate_zone_correlation_analysis()
    demonstrate_configuration_options()
    
    logger.info("\n" + "="*60)
    logger.info("All zone optimization demonstrations completed!")
    logger.info("="*60)