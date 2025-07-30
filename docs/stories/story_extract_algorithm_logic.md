# Story: Extract and Modularize Algorithm Logic

**As a** Developer,
**I want to** extract the implementation of each of the 8 optimization algorithms (GA, PSO, SA, DE, ACO, BO, RS, HC) into its own module,
**So that** the core intellectual property is preserved and can be integrated into the new architecture.

### Legacy Logic to Preserve for EACH Algorithm:

- **GA**: All configurable parameters (`population_size`, `generations`, `mutation_rate`), the exact crossover and tournament selection methods.
- **PSO**: All configurable parameters (`swarm_size`, `inertia`, `acceleration coefficients`), the velocity update equations, and boundary handling.
- **SA**: All configurable parameters (`initial_temperature`, `cooling_rate`) and the neighbor selection logic.
- **DE, ACO, BO, RS, HC**: All specific parameters and implementation logic for each must be extracted and preserved without modification.

### Module Requirements:

- Each of the 8 algorithms must be in its own, independent Python module.
- Each module must read its specific parameters directly from the `.ini` configuration file.
- Each module must be designed to accept the pre-processed data (post-ULTA and Correlation steps) as its primary input.
- The modules must use the **exact same fitness function** from the legacy code, which incorporates the correlation penalty.
- The modules must support parallel execution as implemented in the legacy script.
- Each module must return the best portfolio it found along with its final fitness score.
- The modules must support the zone-wise optimization mode, operating on a subset of data when required.
- The modules must handle variable portfolio sizes based on the `min_size` and `max_size` parameters in the configuration.