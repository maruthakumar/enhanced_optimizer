# Configuration Management Guide

## Overview

The Heavy Optimizer Platform now features a comprehensive configuration management system that allows all system parameters to be adjusted without code changes. This guide covers the configuration system architecture, usage, and best practices.

## Architecture

### Core Components

1. **ConfigurationManager** (`/backend/config/config_manager.py`)
   - Centralized configuration loading and validation
   - Default value handling
   - Job-specific override support
   - Type-safe getter methods

2. **Algorithm Configuration**
   - All 8 algorithms now read parameters from configuration
   - Located in `/backend/algorithms/`
   - Each algorithm extends `BaseOptimizationAlgorithm`

3. **Module Integration**
   - ULTA Calculator reads from configuration
   - Zone Optimizer reads from configuration
   - Correlation module reads from configuration

## Configuration File Structure

### Base Configuration
The base configuration file is located at `/config/production_config.ini`

Key sections include:

```ini
[SYSTEM]
platform_name = Heavy Optimizer Platform
version = 4.1
environment = production

[ALGORITHM_PARAMETERS]
# Genetic Algorithm
ga_population_size = 30
ga_mutation_rate = 0.1
ga_crossover_rate = 0.8
ga_generations = 100

# Particle Swarm Optimization
pso_swarm_size = 25
pso_inertia_weight = 0.9
pso_iterations = 75

# ... parameters for all 8 algorithms

[PORTFOLIO_OPTIMIZATION]
min_portfolio_size = 10
max_portfolio_size = 100
default_portfolio_size = 35

[ZONE]
enabled = true
zone_count = 4
zone_weights = 0.25,0.25,0.25,0.25

[ULTA]
enabled = true
roi_threshold = 0.0
inversion_method = negative_daily_returns

[DATABASE]
type = heavydb
host = localhost
port = 6274
```

### Job-Specific Overrides

Job-specific configurations can override any base settings:

```ini
# /config/job_specific_example.ini
[ALGORITHM_PARAMETERS]
ga_population_size = 50  # Override for this job
ga_generations = 200

[PORTFOLIO_OPTIMIZATION]
default_portfolio_size = 50
```

## Usage

### Basic Usage

```python
from config.config_manager import get_config_manager

# Get singleton configuration manager
config = get_config_manager('/path/to/config.ini')

# Get values with type safety
population_size = config.getint('ALGORITHM_PARAMETERS', 'ga_population_size')
mutation_rate = config.getfloat('ALGORITHM_PARAMETERS', 'ga_mutation_rate')
enabled = config.getboolean('ZONE', 'enabled')
weights = config.getlist('ZONE', 'zone_weights')
```

### With Algorithms

```python
from algorithms import create_algorithm

# Create algorithm with configuration
ga = create_algorithm('ga', '/path/to/config.ini')
# Algorithm automatically loads its parameters from config
```

### Job-Specific Override

```python
config = get_config_manager('/path/to/base_config.ini')
config.load_job_config('/path/to/job_config.ini')
```

### Validation

```python
# Validate configuration
result = config.validate_config()

if not result.is_valid:
    print(f"Errors: {result.errors}")
    print(f"Missing sections: {result.missing_sections}")

# Get detailed validation report
print(config.get_validation_report())
```

## Configuration Parameters

### Algorithm Parameters

Each algorithm has specific parameters:

**Genetic Algorithm (GA)**
- `ga_population_size`: Size of population (default: 30)
- `ga_mutation_rate`: Mutation probability (default: 0.1)
- `ga_crossover_rate`: Crossover probability (default: 0.8)
- `ga_generations`: Number of generations (default: 100)
- `ga_selection_method`: Selection method (default: tournament)

**Particle Swarm Optimization (PSO)**
- `pso_swarm_size`: Number of particles (default: 25)
- `pso_inertia_weight`: Inertia weight (default: 0.9)
- `pso_cognitive_coefficient`: Cognitive coefficient (default: 2.0)
- `pso_social_coefficient`: Social coefficient (default: 2.0)
- `pso_iterations`: Number of iterations (default: 75)

**Simulated Annealing (SA)**
- `sa_initial_temperature`: Starting temperature (default: 1000.0)
- `sa_final_temperature`: Final temperature (default: 0.01)
- `sa_cooling_rate`: Cooling rate (default: 0.95)
- `sa_max_iterations`: Maximum iterations (default: 1000)

### Zone Configuration

- `enabled`: Enable zone optimization (default: true)
- `zone_count`: Number of zones (default: 4)
- `zone_weights`: Comma-separated weights (default: 0.25,0.25,0.25,0.25)
- `zone_selection_method`: Selection method (default: balanced)
- `min_strategies_per_zone`: Minimum strategies per zone (default: 5)
- `max_strategies_per_zone`: Maximum strategies per zone (default: 20)

### ULTA Configuration

- `enabled`: Enable ULTA transformation (default: true)
- `roi_threshold`: ROI threshold for inversion (default: 0.0)
- `inversion_method`: Method for inversion (default: negative_daily_returns)
- `min_negative_days`: Minimum negative days (default: 10)
- `negative_day_percentage`: Required negative day percentage (default: 0.6)

## Default Values

The system provides sensible defaults for all parameters. If a parameter is missing from the configuration file, the default value is used automatically.

## Best Practices

1. **Always validate configuration** before running workflows
2. **Use job-specific overrides** instead of modifying base config
3. **Document custom parameters** in job configuration files
4. **Keep base configuration** in version control
5. **Test configuration changes** with small datasets first

## Example Workflows

### Configuration-Driven Workflow

```bash
# Run with base configuration
python config_driven_workflow.py --input data.csv

# Run with job-specific override
python config_driven_workflow.py \
    --config /config/production_config.ini \
    --job-config /config/job_specific.ini \
    --input data.csv

# Validate configuration only
python config_driven_workflow.py \
    --config /config/production_config.ini \
    --validate-only
```

### Creating Custom Configuration

```python
from config.config_manager import ConfigurationManager

# Create new configuration
config = ConfigurationManager()

# Set custom values
config.config.add_section('CUSTOM')
config.config.set('CUSTOM', 'my_param', 'my_value')

# Save to file
config.save_config('/path/to/custom_config.ini')
```

## Troubleshooting

### Common Issues

1. **Missing Required Sections**
   - Error: "Missing required section: ALGORITHMS"
   - Solution: Ensure all required sections are in config file

2. **Invalid Parameter Types**
   - Error: "ga_population_size must be numeric"
   - Solution: Check parameter values are correct type

3. **Zone Weight Mismatch**
   - Error: "Number of zone weights must match zone_count"
   - Solution: Ensure zone_weights has correct number of values

### Debug Mode

Enable debug logging to see configuration details:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Migration Guide

### From Hardcoded Values

Before:
```python
# Hardcoded in workflow
for generation in range(100):  # Hardcoded!
    ...
```

After:
```python
# Read from configuration
generations = config.getint('ALGORITHM_PARAMETERS', 'ga_generations')
for generation in range(generations):
    ...
```

### From Old Workflows

Use the `WorkflowAlgorithmAdapter` for backward compatibility:

```python
from algorithms.workflow_adapter import WorkflowAlgorithmAdapter

adapter = WorkflowAlgorithmAdapter('/path/to/config.ini')
result = adapter.run_genetic_algorithm(data, portfolio_size)
```

## Testing

Run configuration tests:

```bash
cd /mnt/optimizer_share/backend/tests
python run_config_tests.py
```

## Summary

The configuration management system provides:

- ✅ Centralized parameter management
- ✅ No code changes needed for parameter tuning
- ✅ Job-specific overrides
- ✅ Validation and error reporting
- ✅ Type-safe access methods
- ✅ Default value handling
- ✅ Full backward compatibility

All requirements from the Configuration Management story have been successfully implemented.