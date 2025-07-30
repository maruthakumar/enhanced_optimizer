# Story: HeavyDB Table Schema Optimization

**As a** Database Architect,
**I want to** optimize the HeavyDB table schema for analytical query performance,
**So that** all subsequent operations (ULTA, Correlation, Algorithms) run as fast as possible.

**Acceptance Criteria**:

1.  **Columnar Storage**: Ensure the table is created using HeavyDB's optimal columnar storage format.
2.  **Partitioning**: Implement a partitioning strategy on the table, likely by the `Date` column, to improve time-series query performance.
3.  **Data Encoding**: Analyze and apply the most efficient encoding for each column to reduce storage footprint and improve scan speed.
4.  **Index Strategy**: While HeavyDB is often index-less, verify if any specific columns (e.g., `Zone`) would benefit from a search index to accelerate data filtering.
5.  **Performance Validation**: Create a benchmark test that runs a set of typical analytical queries against both a default table and the optimized table to prove that the optimizations have a measurable positive impact.

**Technical Notes**:
- This story directly addresses the "Columnar Storage Optimized for analytics" block in the architecture diagram.
- This should be implemented as part of the `Dynamic Table Creation` process.
