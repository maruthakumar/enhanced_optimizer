# Story: Extract and Re-implement Zone-wise Optimization Logic

**As a** Developer,
**I want to** extract the zone-wise optimization logic from the `run_zone_specific_optimization_for_all_zones()` function,
**So that** this core feature is preserved and integrated into the new architecture.

### Legacy Logic to Preserve:

- **Zone Configuration**: The number of zones, their definitions, and their weights must be read from the configuration file. The implementation must support any number of zones, not be hardcoded to 4.
- **Independent Optimization**: The logic of `run_zone_specific_optimization_for_all_zones()`, which treats each zone as a separate, independent optimization problem, must be preserved.

### Implementation Requirements:

- The module must perform dynamic, zone-based filtering of the data in HeavyDB.
- It must support configurable methods for combining the results of the individual zone optimizations into a final, composite portfolio.