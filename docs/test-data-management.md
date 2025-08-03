# Test Data Management Guide

## Overview
The Heavy Optimizer Platform uses real production data for all testing, following these principles:

## Data Sets
1. **Small Set**: SENSEX_test_dataset.csv (731 rows) - Unit tests
2. **Medium Set**: Anonymized 5,000 strategy subset - Integration tests  
3. **Large Set**: Full production data (25,544 strategies) - Performance tests

## Anonymization Process
1. Map strategy names to generic IDs (STRATEGY_0001, etc.)
2. Shift dates to obscure actual trading periods
3. Maintain all mathematical relationships
4. Preserve edge cases and anomalies

## Security Requirements
- Test data stored in `/backend/tests/data/` with restricted access
- No commit of non-anonymized data to version control
- Regular audit of test data for sensitive information

## Edge Case Identification

### Real Data Edge Cases to Preserve
1. **Zero Return Days**: Strategies with no P&L change
2. **Maximum Drawdown Events**: Largest historical losses
3. **Winning Streaks**: Continuous profitable periods
4. **Correlation Extremes**: Highly correlated strategy pairs
5. **Regime Transitions**: Market regime change periods

### Data Selection Criteria
- **Temporal Coverage**: Include data from all market regimes
- **Performance Distribution**: Cover full range of returns
- **Correlation Diversity**: Various correlation levels
- **Drawdown Scenarios**: Different drawdown patterns

## Test Data Pipeline

```bash
# Anonymization Script Location
/backend/scripts/anonymize_test_data.py

# Usage
python anonymize_test_data.py \
  --input /input/Python_Multi_Consolidated_20250726_161921.csv \
  --output /backend/tests/data/test_data_large.csv \
  --strategy-mapping /backend/tests/data/strategy_mapping.json \
  --date-shift 180
```

## Compliance Requirements

1. **No Proprietary Information**: All strategy names must be anonymized
2. **Date Obfuscation**: Shift dates to prevent market timing identification
3. **Maintain Relationships**: Preserve correlations and patterns
4. **Audit Trail**: Log all anonymization operations

## Test Data Validation

Before using anonymized data:
1. Verify mathematical relationships preserved
2. Check correlation matrices match original
3. Validate edge cases still present
4. Ensure no identifiable information remains

## Storage and Version Control

```
/backend/tests/data/
├── test_data_small.csv      # 731 rows (SENSEX subset)
├── test_data_medium.csv     # 5,000 strategies
├── test_data_large.csv      # 25,544 strategies
├── strategy_mapping.json    # Anonymization mapping (NEVER commit)
└── edge_cases/
    ├── max_drawdown.csv
    ├── zero_returns.csv
    └── high_correlation.csv
```

## Best Practices

1. **Regular Updates**: Refresh test data quarterly to include new patterns
2. **Edge Case Monitoring**: Identify new edge cases from production
3. **Performance Baseline**: Maintain expected results for regression testing
4. **Security Reviews**: Monthly audit of test data for sensitive information

## Integration with CI/CD

The CI/CD pipeline (Story 1.7) uses these datasets:
- **Unit Tests**: `test_data_small.csv`
- **Integration Tests**: `test_data_medium.csv`
- **Performance Tests**: `test_data_large.csv`
- **Edge Case Tests**: Files from `edge_cases/` directory

## Responsibilities

- **Data Engineer**: Create and maintain anonymization pipeline
- **QA Team**: Validate test data integrity
- **Security**: Audit for sensitive information
- **DevOps**: Secure test data storage in CI/CD