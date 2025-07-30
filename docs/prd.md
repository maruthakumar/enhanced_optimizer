# Heavy Optimizer Platform Enhancement PRD

## 1. Introduction and Analysis

### 1.1. Project Goal

To evolve the existing, proven file-based Python optimizer into the fully integrated, database-driven, GPU-accelerated system depicted in the `updated_optimization_zone_v1.png` architecture diagram. This involves bridging the architectural gaps, implementing missing components, and enhancing performance while preserving the core, validated logic of the original system.

### 1.2. Architectural Analysis

#### 1.2.1. Target Architecture (Per Diagram)

The target is a multi-layered, database-driven system that uses HeavyDB for high-speed data manipulation and GPU acceleration. Its data flow is:
1.  **Input**: CSV data is loaded.
2.  **HeavyDB Integration**: Data is transferred to a GPU-resident columnar database.
3.  **Pre-Processing**: ULTA inversion and Correlation Matrix calculation are performed on the data *within the database*.
4.  **Optimization**: 8 parallel algorithms (GA, PSO, etc.) run on the pre-processed data.
5.  **Selection & Analytics**: A winning portfolio is selected, and advanced analytics (Attribution, Sensitivity) are performed.
6.  **Output**: 8 distinct reports, charts, and data files are generated.

#### 1.2.2. Legacy Implementation (`Optimizer_New_patched.py`)

The legacy code is a powerful, file-based Python script that uses Pandas/NumPy for in-memory processing. It successfully implements the core optimization logic but lacks the database integration and full GPU acceleration of the target architecture.

#### 1.2.3. Integration Gaps to Address

This enhancement must bridge the following gaps between the legacy code and the target architecture:
- **Database Integration**: The legacy code is file-based. It must be re-engineered to use HeavyDB as the primary data store and processing engine, as shown in the diagram.
- **GPU Acceleration**: The legacy code's optional TensorFlow hook is minimal. The system must be updated to use CuPy/CUDA for comprehensive GPU acceleration, leveraging HeavyDB's capabilities.
- **Component Integration**: The Samba job queue processor is currently external. It must be integrated into a unified pipeline orchestrator.
- **Missing Layers**: The `Advanced Analytics` and real-time monitoring layers are present in the diagram but absent from the legacy code and must be implemented.

## 2. Core Principles and Technical Details to Preserve

### 2.1. Core Logic

- **ULTA Strategy Inversion**: The logic of inverting poorly performing strategies *before* optimization is critical for performance and must be maintained.
- **Correlation Penalty**: The fitness function's use of a correlation penalty (`base_fitness * (1 - correlation_penalty)`) is essential for diversification and must be retained.
- **Data Flow Sequence**: The exact `ULTA -> Correlation -> Algorithm` sequence is non-negotiable and must be the backbone of the new data pipeline.
- **Zone-Wise Optimization**: The capability to run the entire optimization pipeline independently for different trading zones is a core feature that must be fully supported.

### 2.2. Algorithm-Specific Parameters

The specific implementation details and parameters for each of the 8 algorithms from the legacy code must be preserved:
- **GA**: `population_size`, `mutation_rate`, `generations`
- **PSO**: `swarm_size`, inertia weight, acceleration coefficients
- **SA**: `initial_temperature`, `cooling_rate`
- **DE, ACO, BO, RS, HC**: All other specific parameters and logic from the legacy implementations.

### 2.3. Zone Configuration

- The system must support the configuration of 4 distinct trading zones with specific market hours.
- It must allow for the definition of `zone_weights` in the configuration to create a final, weighted portfolio from the results of each zone's independent optimization.

## 3. Requirements

### 3.1. Functional Requirements (FR)

| ID | Requirement |
|:---|:---|
| **FR1** | **Database Integration**: The system shall use HeavyDB as its primary data backend. All data loading, pre-processing (ULTA, Correlation), and optimization queries shall be executed within HeavyDB. The target table name is `strategies_python_multi_consolidated`. |
| **FR2** | **Pipeline Orchestration**: A unified orchestration component shall be created to manage the end-to-end data flow, from Samba job reception (from `\\204.12.223.93\optimizer_share`) to final output generation, ensuring the `ULTA -> Correlation -> Algorithm` sequence. |
| **FR3** | **ULTA in HeavyDB**: The ULTA strategy inversion logic shall be re-implemented as a series of SQL/HeavyDB operations to modify strategy data directly on the GPU. |
| **FR4** | **Correlation in HeavyDB**: The correlation matrix calculation (pairwise) shall be re-implemented to run efficiently within HeavyDB, capable of handling a 28,044² matrix and targeting an average correlation of ~0.142. |
| **FR5** | **GPU Memory Management**: The system shall implement an intelligent GPU memory manager to handle large datasets, targeting a VRAM allocation of ~21.1GB of a 40GB total. |
| **FR6** | **Real-Time Monitoring**: A real-time monitoring service shall be implemented, providing progress updates (e.g., percentage complete, current algorithm) and key metrics via a REST API. |
| **FR7** | **Advanced Analytics Layer**: The `Performance Attribution` and `Sensitivity Analysis` components from the architecture diagram shall be implemented as post-optimization analysis steps. |
| **FR8** | **Error Handling & Recovery**: The pipeline shall have robust error handling and be able to gracefully recover from and log common failures (e.g., database connection issues, failed jobs). |
| **FR9** | **Performance Profiling**: A new module shall be added to profile the performance (execution time, memory usage) of each of the 8 algorithms during optimization runs. |

### 3.2. Non-Functional Requirements (NFR)

| ID | Requirement |
|:---|:---|
| **NFR1** | **Performance Benchmark**: The total execution time for the reference dataset (`Python_Multi_Consolidated_20250726_challenge10Post_ohevM6L1WqLCFH0jSTKxkLJd`) must be maintained at or below **10.04 seconds**. CSV parsing must complete in **~2.3s**. |
| **NFR2** | **GPU Utilization**: The system should achieve a peak GPU utilization of **95%** during the optimization phase. |
| **NFR3** | **Memory Benchmark**: Peak system memory usage must not exceed **26.6GB**. |
| **NFR4** | **Backward Compatibility**: The system must remain fully compatible with the existing Windows client access via Samba share and the current CSV input format. |
| **NFR5** | **Configuration Driven**: All major parameters, including zone settings, algorithm selection, and ULTA logic, must continue to be controlled via the `.ini` configuration files. |
| **NFR6** | **Critical Success Metrics**: The final selected portfolio must have a size between 35-50 strategies, and the winner selection must be based on the highest fitness score among the 8 algorithms. |

### 3.3. Output Requirements

The system must generate the 8 specific output files as shown in the architecture diagram, including:
1.  **ULTA Inversion Report (.md)**
2.  **Zone Analysis Report (.md)**
3.  **Performance Report (.txt)**
4.  **Portfolio Composition (.csv)**
5.  **Excel Summary (.xlsx)**
6.  **Execution Summary (.json)**
7.  **Equity Curves (.png)**
8.  **Algorithm Comparison (.png)**

## 4. Risks and Mitigation

| Risk | Mitigation Strategy |
|:---|:---|
| **HeavyDB Integration Complexity** | Develop a dedicated data access layer (DAL) to abstract HeavyDB interactions. Create a suite of integration tests to validate each query and data transformation step independently. |
| **Maintaining Performance Benchmarks** | Implement continuous performance profiling (FR9). Benchmark each component before and after changes. Optimize critical queries and data transfer operations. |
| **GPU Memory Overflow** | Implement the intelligent GPU memory manager (FR5) with proactive memory estimation and garbage collection. Add configuration limits to prevent runaway memory usage. |
| **Breaking Windows Client Workflows** | Maintain the exact same Samba share path and CSV input/output formats (NFR4). Conduct end-to-end testing from a Windows client environment to validate workflows. |

## 5. Epic and Story Structure

### 5.1. Epic: Architect and Integrate the HeavyDB Optimization Pipeline

**Goal**: Refactor the file-based optimizer into a robust, database-driven platform that matches the target architecture, while preserving its core logic and meeting performance benchmarks.

### 5.2. User Stories

| ID | Story |
|:---|:---|
| 1 | As a Quant, I want the system to use HeavyDB for all data operations so that I can leverage its GPU acceleration and optimize much larger datasets than before. |
| 2 | As an Operator, I want a unified pipeline that automatically processes jobs from the Samba queue to completion so that the system requires minimal manual intervention. |
| 3 | As a Quant, I want the proven ULTA and Correlation logic to be applied correctly *before* optimization so that I can trust the results are both high-performing and diversified. |
| 4 | As an Operator, I want to monitor the optimization progress in real-time so that I can track long-running jobs and identify issues quickly. |
| 5 | As a Developer, I want to profile the performance of each algorithm so that I can identify and address bottlenecks to meet the 10-second execution target. |