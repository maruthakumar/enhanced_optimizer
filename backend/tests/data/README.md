# Test Data Directory

This directory contains anonymized production data for testing purposes.

## Available Test Datasets

1. **test_data_small.csv** (83 rows)
   - Small dataset for unit tests
   - Contains all 25,544 strategies
   - Dates shifted by 180 days
   - Created from Python_Multi_Consolidated_20250726_161921.csv

## Anonymization Process

Test data is created using the anonymization scripts:
- `/backend/scripts/anonymize_test_data.py` - Full anonymization with validation
- `/backend/scripts/simple_anonymize.py` - Fast anonymization for large datasets

### Creating Test Data

```bash
# Small dataset (fast unit tests)
python3 scripts/simple_anonymize.py \
  /input/Python_Multi_Consolidated_20250726_161921.csv \
  tests/data/test_data_small.csv \
  180

# Medium dataset (integration tests) - first extract 5000 rows
head -5001 /input/Python_Multi_Consolidated_20250726_161921.csv > temp_medium.csv
python3 scripts/simple_anonymize.py temp_medium.csv tests/data/test_data_medium.csv 180
rm temp_medium.csv
```

## Security Notes

- **NEVER commit *_mapping.json files** - These contain the mapping between real and anonymized strategy names
- All mapping files are excluded via .gitignore
- Dates are shifted by 180 days to obscure actual trading periods
- Strategy names are replaced with generic identifiers (STRATEGY_0001, etc.)

## Usage in Tests

```python
import pandas as pd

# Load test data
df = pd.read_csv('/backend/tests/data/test_data_small.csv')

# Get strategy columns (exclude metadata)
metadata_cols = ['Date', 'Day']
strategy_cols = [col for col in df.columns if col not in metadata_cols]
```

## Edge Cases

Edge case datasets will be generated from the anonymized data:
- `edge_cases/max_drawdown.csv` - Periods with maximum drawdowns
- `edge_cases/zero_returns.csv` - Days with zero returns
- `edge_cases/high_correlation.csv` - Highly correlated strategy pairs