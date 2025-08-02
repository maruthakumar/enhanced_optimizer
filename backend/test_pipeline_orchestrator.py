#!/usr/bin/env python3
"""
Test script for Pipeline Orchestrator
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.pipeline_orchestrator import PipelineOrchestrator

def test_orchestrator():
    """Test the pipeline orchestrator with sample data."""
    
    print("Testing Pipeline Orchestrator")
    print("=" * 50)
    
    # Initialize orchestrator
    orchestrator = PipelineOrchestrator()
    
    # Validate configuration
    print("\n1. Validating configuration...")
    validation = orchestrator.validate_configuration()
    print(f"   Configuration valid: {validation['is_valid']}")
    if validation['warnings']:
        print(f"   Warnings: {validation['warnings']}")
    
    # Test with sample CSV
    test_csv = "/mnt/optimizer_share/input/SENSEX_test_dataset.csv"
    
    if not os.path.exists(test_csv):
        print(f"\nTest CSV not found: {test_csv}")
        print("Looking for alternative test files...")
        
        # Find any CSV in input directory
        input_dir = "/mnt/optimizer_share/input"
        csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
        
        if csv_files:
            test_csv = os.path.join(input_dir, csv_files[0])
            print(f"Using: {test_csv}")
        else:
            print("No CSV files found in input directory")
            return
    
    # Test dry-run mode first
    print(f"\n2. Testing dry-run mode with: {test_csv}")
    try:
        results = orchestrator.execute_pipeline(
            input_csv=test_csv,
            portfolio_size=10,
            execution_mode='dry-run'
        )
        
        print("\nDry-run completed successfully!")
        print(f"   Steps executed: {len(results['step_timings'])}")
        print(f"   Total time: {results['total_execution_time']:.2f}s")
        
    except Exception as e:
        print(f"Error during dry-run: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Test execution modes
    print("\n3. Available execution modes:")
    modes = orchestrator.get_supported_execution_modes()
    for mode in modes:
        print(f"   - {mode}")
    
    print("\nTest completed!")


if __name__ == "__main__":
    test_orchestrator()