# Heavy Optimizer Platform Brownfield Enhancement Architecture

## 1. Introduction

This document outlines the architectural migration from HeavyDB to Apache Parquet/Arrow/cuDF stack, enabling 100k+ strategy optimization with enhanced financial metrics and market regime analysis.

**Enhanced CSV Input Architecture:**

**New Columns Introduced:**
- `start_time`, `end_time`: Strategy start & end time
- `market_regime`: Bullish_Vol_Expansion, Bearish_Panic classifications
- `Regime_Confidence_%`: Confidence scoring (0-100%)
- `Market_regime_transition_threshold`: Transition probability thresholds
- `capital`: Dynamic capital allocation per strategy
- `zone`: Intraday zone - optimize strategies based on zone

**Key Architectural Changes:**
- **Data pipeline**: CSV → Parquet → Arrow → cuDF
- **Parallel workers**: Multi-process GPU execution for 100k+ strategies
- **Scalability**: Remove 32GB limitation
- **Financial optimization**: Kelly Criterion, VaR/CVaR, regime-based allocation
- **Multi-format CSV support** with auto-detection

## 2. Technical Architecture Migration

**Data Stack Replacement:**
- **Remove**: HeavyDB, pymapd, SQL-based operations
- **Add**: Apache Parquet, Arrow, cuDF, RAPIDS ecosystem

**Pipeline Architecture:**
```
CSV → Parquet (compression) → Arrow (zero-copy) → cuDF (GPU) → Optimization
```

## 3. Data Layer Architecture

**Parquet/Arrow/cuDF Stack:**
- Dynamic schema detection for variable CSV formats
- Parquet partitioning by date/zone for query optimization
- Arrow memory pools for efficient GPU transfers
- cuDF manages unlimited dataset sizes (vs 32GB HeavyDB limit)

## 4. Backward-Compatible Optimization Engine

**Enhanced Configurable Fitness:**
```ini
[FITNESS_CALCULATION]
legacy_metrics = roi,dd,ratio
enhanced_metrics = sharpe,sortino,calmar,kelly
mode = enhanced  # legacy|enhanced|hybrid

[MARKET_REGIME_CONFIG]
regime_column = market_regime
confidence_column = Regime_Confidence_%
min_confidence = 70
```

**Backward-Compatible Fitness:**
```python
# Legacy mode: roi/drawdown - penalty
if mode == "legacy":
    fitness = roi / max_drawdown - correlation_penalty
# Enhanced mode: full financial optimization
elif mode == "enhanced":
    fitness = kelly_weight * sharpe * regime_factor - var_penalty
```

## 5. Implementation Strategy

**Epic 1 Retrofit Plan:**
- Stories 1.1-1.4: Replace HeavyDB with Parquet/Arrow/cuDF
- Story 1.5: Multi-format CSV configuration
- Story 1.6: Enhanced financial optimization

## 6. Success Metrics & Validation

**Performance Validation:**
- 100k+ strategy processing capability
- 10x speed improvement (Parquet vs CSV)
- Unlimited memory scaling (vs 32GB limit)

**Technical Success Criteria:**
- Multi-format CSV auto-detection
- ULTA logic preservation through migration
- Configuration-driven optimization modes
- Enhanced output generation (8+ report types)

## 7. Enhanced Integration Pipeline

The complete data processing pipeline implements the following stages:

1. **CSV → Parquet**: Enhanced CSV (with regime/timing columns) converts to compressed columnar storage
2. **Parquet → Arrow**: Zero-copy memory mapping for efficient GPU transfer
3. **Arrow → GPU**: cuDF DataFrames loaded into GPU memory
4. **Parallel Workers**: Multi-process optimization execution layer
5. **ULTA Logic**: Strategy inversion algorithm (preserving existing implementation)
6. **Correlation Matrix**: GPU-accelerated pairwise correlation calculations
7. **8 Optimization Techniques**: 
   - Genetic Algorithm (GA)
   - Particle Swarm Optimization (PSO)
   - Simulated Annealing (SA)
   - Differential Evolution (DE)
   - Ant Colony Optimization (ACO)
   - Bayesian Optimization (BO)
   - Random Search (RS)
   - Hill Climbing (HC)
8. **Advanced Financial Metrics**:
   - Kelly Criterion for position sizing
   - VaR/CVaR for risk assessment
   - Sharpe/Sortino ratios for risk-adjusted returns
   - Calmar ratio for drawdown analysis
9. **Enhanced Output Generation**:
   - Original 8 output reports
   - Market regime analysis reports
   - Capital optimization recommendations
   - Risk-adjusted portfolio allocations

This pipeline ensures seamless migration from HeavyDB while adding advanced capabilities.

## 8. Testing Data Strategy

The Heavy Optimizer Platform follows a production-data-first testing philosophy:

1. **Unit Tests**: Use anonymized subsets of real production data
2. **Integration Tests**: Full production datasets with sensitive data masked
3. **Performance Tests**: Actual production workloads (25,544 strategies)
4. **Edge Case Testing**: Curated real data exhibiting edge conditions

**Rationale**: Financial optimization algorithms must be validated against real market conditions to ensure accuracy. Synthetic data cannot replicate the complex correlations and market regimes present in actual trading data.

**Data Security**: All test data is anonymized using strategy ID mapping and date shifting to protect proprietary information while maintaining mathematical relationships.
