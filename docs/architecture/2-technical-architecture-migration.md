# 2. Technical Architecture Migration

**Data Stack Replacement:**
- **Remove**: HeavyDB, pymapd, SQL-based operations
- **Add**: Apache Parquet, Arrow, cuDF, RAPIDS ecosystem

**Pipeline Architecture:**
```
CSV → Parquet (compression) → Arrow (zero-copy) → cuDF (GPU) → Optimization
```
