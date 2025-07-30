# Story: Extract and Re-implement ULTA Logic in HeavyDB

**As a** Developer,
**I want to** extract the ULTA strategy inversion logic from the `apply_ulta_logic()` function and re-implement it as a standalone, testable module that operates on data within the HeavyDB database,
**So that** we can preserve this critical pre-processing step in the new architecture.

### Legacy Logic to Preserve:

1.  **ROI Calculation**: For each strategy, calculate `ROI = (final_value / initial_value) - 1`.
2.  **Inversion Condition**: If `ROI < 0`, create a temporary inverted version of the strategy by multiplying its daily returns by -1.
3.  **Recalculate ROI**: Calculate the ROI for the new, inverted strategy.
4.  **Decision**: The inversion is kept **only if** `inverted_ROI > original_ROI`.

### Implementation Requirements:

- The process must run on the entire input dataset.
- The module must track exactly which strategies were inverted.
- It must calculate and store the percentage improvement for each inverted strategy.

### HeavyDB Implementation Specifics:

- Use `UPDATE` queries with `CASE` statements to perform the inversion efficiently on the GPU.
- Create and populate a separate metadata table to track the inverted strategies and their performance improvement.
- The final output must be a new table or a modified version of the input table, ready for the correlation step.
- The module must generate the `ULTA Inversion Report` as seen in the legacy system.

### Testing Requirements:

- Create unit tests that directly compare the output of the new HeavyDB module with the output of the legacy `apply_ulta_logic()` function.
- Verify that the **exact same strategies** are inverted in both implementations for a given test dataset.
- Validate that the calculated ROI improvement percentages match the legacy implementation precisely.