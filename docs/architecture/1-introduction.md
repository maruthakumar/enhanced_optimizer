# 1. Introduction

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
