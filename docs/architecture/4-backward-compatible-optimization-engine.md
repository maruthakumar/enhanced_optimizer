# 4. Backward-Compatible Optimization Engine

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