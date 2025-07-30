# Story: End-to-End Integration Testing

**As a** QA Engineer,
**I want** comprehensive integration tests for the full pipeline,
**So that** we can verify that all components work together correctly.

**Acceptance Criteria**:
- An integration test suite that covers the entire pipeline, from CSV input to all 8 output formats.
- Tests that validate the Samba share integration by submitting jobs from a simulated Windows client environment.
- Tests that use various data sizes (small, medium, large) and configurations.
- A specific test that validates the final results against the output of the legacy system for a benchmark dataset.
- Tests for common error scenarios and the system's ability to recover from them.
- A suite of performance benchmark tests to ensure the system meets its performance targets.
- Specific tests for the zone-wise optimization mode.
