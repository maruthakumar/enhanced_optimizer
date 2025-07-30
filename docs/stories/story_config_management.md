# Story: Centralized Configuration Management

**As a** User,
**I want to** manage all system parameters through configuration files,
**So that** I can adjust behavior without code changes.

**Acceptance Criteria**:

Support configuration for:
- All 8 algorithm-specific parameters.
- Portfolio size ranges (`min_size`, `max_size`).
- Zone definitions and weights.
- ULTA thresholds.
- Correlation penalty weights.
- Output formats and locations.
- Resource limits (memory, time).
- Database connection settings.

**Requirements**:
- Use the existing `.ini` file format for backward compatibility.
- The system must validate the configuration on load and report any errors.
- Provide sensible default values for all parameters.
- Support the ability to have a base configuration that can be overridden by a job-specific configuration.
