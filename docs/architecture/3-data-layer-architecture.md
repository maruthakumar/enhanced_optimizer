# 3. Data Layer Architecture

**Parquet/Arrow/cuDF Stack:**
- Dynamic schema detection for variable CSV formats
- Parquet partitioning by date/zone for query optimization
- Arrow memory pools for efficient GPU transfers
- cuDF manages unlimited dataset sizes (vs 32GB HeavyDB limit)
