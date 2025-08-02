# 4. Enhanced Financial Optimization Requirements

## Capital Utilization Optimization
- **Kelly Criterion**: Optimal position sizing based on win/loss probabilities
- **Risk Budgeting**: Allocate capital by risk contribution rather than dollar amounts
- **Leverage Optimization**: Dynamic leverage based on strategy performance/volatility
- **Capital at Risk**: Maximum capital exposure per strategy/regime

## Risk-Reward Metrics
- **Sharpe Ratio**: Risk-adjusted returns optimization
- **Sortino Ratio**: Downside deviation focus
- **Calmar Ratio**: Return/max drawdown optimization
- **VaR/CVaR**: Value at Risk constraints (95%/99% confidence)
- **Maximum Drawdown**: Hard constraints on portfolio drawdown

## Market Regime Optimization
- **Confidence Score-Based**: Only trade when Regime_Confidence_% > threshold
- **Dynamic Weighting**: Higher confidence = larger position sizing
- **Transition Logic**: Reduce positions during regime uncertainty

```ini
[REGIME_OPTIMIZATION]
min_confidence_threshold = 70
min_transition_threshold = 0.5
confidence_weighting = true
regime_specific_portfolios = true
```
