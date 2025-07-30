# Story: Implement the Data Access Layer (DAL) for HeavyDB

**As a** Developer,
**I want to** implement a dedicated Data Access Layer (DAL) that abstracts all interactions with the HeavyDB database,
**So that** the rest of the application can interact with the database in a clean, consistent, and maintainable way.

### Core Methods Required:

- `load_csv_to_heavydb(filepath, table_name)`: Loads a CSV file into a specified HeavyDB table.
- `apply_ulta_transformation(table_name)`: Executes the ULTA logic on the specified table.
- `compute_correlation_matrix(table_name)`: Computes and returns the correlation matrix for the specified table.
- `get_strategy_subset(table_name, strategy_list)`: Returns a subset of the data for a given list of strategies.
- `execute_gpu_query(sql_query)`: Executes an arbitrary SQL query on the GPU.

### Dynamic Handling Requirements:

- The DAL must dynamically create the table schema based on the structure of the input CSV file.
- The column count and data types must be determined at runtime.
- Memory allocation for operations should be based on the actual size of the data being processed.

### Configuration-Driven Behavior:

- All database connection parameters (host, port, user, password) must be read from the configuration file.
- GPU memory limits and batch sizes for data loading must also be read from the configuration file.