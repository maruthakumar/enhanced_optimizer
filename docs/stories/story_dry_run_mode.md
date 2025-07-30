# Story: Dry Run Mode for Testing

**As a** Developer,
**I want to** add a "dry run" mode to the Pipeline Orchestrator,
**So that** I can test the data flow and configuration of the pipeline without actually performing the expensive computations.

**Acceptance Criteria**:
- The `start_optimization_job()` method in the Pipeline Orchestrator accepts a `dry_run` parameter.
- When `dry_run` is `True`, the orchestrator simulates the data flow and logs the steps that would be taken, but does not actually call the expensive methods in the Data Access Layer and Algorithm Runner.
- The dry run mode is covered by unit tests that verify its correctness.
