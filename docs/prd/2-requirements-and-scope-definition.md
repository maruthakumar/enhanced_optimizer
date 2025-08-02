# 2. Requirements and Scope Definition

## Enhanced CSV Format Requirements
- **New Columns**: start_time, end_time, market_regime, Regime_Confidence_%, Market_regime_transition_threshold, capital, zone
- **Configuration**: Selective column inclusion for optimization
- **Backward Compatibility**: Support both legacy and enhanced CSV formats

## Scalability Requirements
- **Strategy Capacity**: 100k+ strategies (vs current ~25k limit)
- **Memory**: Unlimited (remove 32GB HeavyDB constraint)
- **Processing**: Monte Carlo parameter optimization capability
- **Hardware**: Multi-GPU scaling support

## Architecture Migration Requirements
- **Data Stack**: CSV → Parquet → Arrow → cuDF pipeline
- **Performance**: 10-50x processing improvement
- **Integration**: Preserve ULTA algorithm logic
- **Migration**: Retrofit Epic 1 stories for consistency
