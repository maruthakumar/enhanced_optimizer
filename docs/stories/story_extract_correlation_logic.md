# Story: Extract and Re-implement Correlation Logic in HeavyDB

**As a** Developer,
**I want to** extract the correlation penalty logic from the `evaluate_fitness_with_correlation()` function and re-implement the matrix calculation as a standalone module in HeavyDB,
**So that** we can preserve the system's diversification strategy in the new architecture.

### Legacy Logic to Preserve:

1.  **Correlation Penalty Formula**: The exact formula `fitness = base_fitness * (1 - avg_pairwise_correlation * correlation_penalty_weight)` must be used.
2.  **Pairwise Calculation**: The logic from `compute_avg_pairwise_correlation()` must be precisely replicated.
3.  **Configuration Driven**: The `correlation_penalty_weight` must be read from the configuration file.

### Implementation Requirements:

- The system must handle correlation matrices of any size, based on the number of strategies in the input data.
- It must support configurable correlation thresholds for analysis.

### HeavyDB Implementation Specifics:

- Use HeavyDB's built-in, GPU-accelerated correlation functions (e.g., `CORR`) for maximum performance.
- Implement dynamic memory management to handle large matrices (e.g., 28,044²), allocating resources based on the actual matrix size.
- The computed correlation matrix must be stored or cached efficiently for reuse by all 8 optimization algorithms within a single run.