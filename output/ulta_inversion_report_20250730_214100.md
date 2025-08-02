# ULTA (Ultra Low Trading Algorithm) Inversion Report

**Generated:** 2025-07-30 21:41:02
**Run ID:** 20250730_214100

## Summary

**Total Strategies Inverted:** 2

The following strategies had their returns inverted based on ULTA logic:

| Strategy Name | Original ROI | Inverted ROI | Original Drawdown | Inverted Drawdown | Improvement % |
|---------------|--------------|--------------|-------------------|-------------------|---------------|
| Strategy_15 | -0.0250 | 0.0250 | -0.1500 | -0.1000 | 150.00% |
| Strategy_42 | -0.0180 | 0.0180 | -0.0800 | -0.0600 | 125.00% |

## Overall Impact

**Average Improvement:** 137.50%
**Strategies Improved:** 2
**Strategies Worsened:** 0

## Detailed Analysis

### ULTA Logic Applied

Strategies were inverted based on the following criteria:
- ROI/Drawdown ratio below threshold (typically 0.0)
- Negative returns on majority of trading days
- Consistent underperformance across multiple metrics

### Methodology

The ULTA inversion process:
1. Identifies poorly performing strategies
2. Inverts their daily returns (profit becomes loss, loss becomes profit)
3. Recalculates all performance metrics
4. Includes inverted strategies in optimization if improved

## Configuration

- **Enabled:** True
- **ROI Threshold:** 0.0
- **Inversion Method:** negative_daily_returns
- **Min Negative Days:** 10
- **Negative Day Percentage:** 0.6
