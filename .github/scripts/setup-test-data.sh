#!/bin/bash
# Setup anonymized test data for CI/CD pipeline
# This script should be run locally and the results committed (except mapping files)

set -e

echo "Setting up anonymized test data for CI/CD..."

# Check if we're in the right directory
if [ ! -f "backend/scripts/simple_anonymize.py" ]; then
    echo "ERROR: Must run from project root directory"
    exit 1
fi

# Create test data directory
mkdir -p backend/tests/data/edge_cases

# Check if original data exists
if [ ! -f "input/Python_Multi_Consolidated_20250726_161921.csv" ]; then
    echo "ERROR: Original production data not found"
    echo "Please ensure input/Python_Multi_Consolidated_20250726_161921.csv exists"
    exit 1
fi

# Create small dataset (first 100 rows)
echo "Creating small test dataset..."
head -101 input/Python_Multi_Consolidated_20250726_161921.csv > backend/tests/data/temp_small.csv
python3 backend/scripts/simple_anonymize.py \
    backend/tests/data/temp_small.csv \
    backend/tests/data/test_data_small.csv \
    180
rm backend/tests/data/temp_small.csv

# Create medium dataset (first 5000 rows)
echo "Creating medium test dataset..."
head -5001 input/Python_Multi_Consolidated_20250726_161921.csv > backend/tests/data/temp_medium.csv
python3 backend/scripts/simple_anonymize.py \
    backend/tests/data/temp_medium.csv \
    backend/tests/data/test_data_medium.csv \
    180
rm backend/tests/data/temp_medium.csv

# Extract edge cases from small dataset
echo "Extracting edge cases..."
python3 backend/scripts/extract_edge_cases.py \
    backend/tests/data/test_data_small.csv \
    backend/tests/data/edge_cases

# Validate anonymization
echo "Validating anonymization..."
head -101 input/Python_Multi_Consolidated_20250726_161921.csv > backend/tests/data/original_small.csv
python3 backend/scripts/validate_anonymization.py \
    backend/tests/data/original_small.csv \
    backend/tests/data/test_data_small.csv \
    backend/tests/data/test_data_small_mapping.json

# Clean up
rm backend/tests/data/original_small.csv

echo ""
echo "✅ Test data setup complete!"
echo ""
echo "The following files have been created:"
echo "  - backend/tests/data/test_data_small.csv (unit tests)"
echo "  - backend/tests/data/test_data_medium.csv (integration tests)"
echo "  - backend/tests/data/edge_cases/*.csv (edge case tests)"
echo ""
echo "⚠️  IMPORTANT: Before committing:"
echo "  1. Add test data files to git: git add backend/tests/data/*.csv"
echo "  2. DO NOT add mapping files: *_mapping.json files are in .gitignore"
echo "  3. Review files to ensure no sensitive data remains"
echo ""
echo "To use in tests:"
echo "  export TEST_DATA_PATH=backend/tests/data/test_data_small.csv"