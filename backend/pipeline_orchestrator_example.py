#!/usr/bin/env python3
"""
Pipeline Orchestrator Integration Example

This example demonstrates how to use the Pipeline Orchestrator to run
the complete Heavy Optimizer Platform pipeline.
"""

import os
import sys
import argparse
import logging
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.pipeline_orchestrator import PipelineOrchestrator


def main():
    """Main function demonstrating Pipeline Orchestrator usage"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Heavy Optimizer Platform - Pipeline Orchestrator Example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full optimization with default portfolio size
  python pipeline_orchestrator_example.py --input data.csv
  
  # Run with custom portfolio size
  python pipeline_orchestrator_example.py --input data.csv --portfolio-size 50
  
  # Run in dry-run mode to test without processing
  python pipeline_orchestrator_example.py --input data.csv --mode dry-run
  
  # Run zone-wise optimization
  python pipeline_orchestrator_example.py --input data.csv --mode zone-wise
  
  # Use custom configuration file
  python pipeline_orchestrator_example.py --input data.csv --config custom_config.ini
        """
    )
    
    parser.add_argument(
        '--input', 
        required=True,
        help='Path to input CSV file'
    )
    
    parser.add_argument(
        '--portfolio-size',
        type=int,
        default=35,
        help='Target portfolio size (default: 35)'
    )
    
    parser.add_argument(
        '--mode',
        choices=['full', 'zone-wise', 'dry-run'],
        default='full',
        help='Execution mode (default: full)'
    )
    
    parser.add_argument(
        '--config',
        help='Path to configuration file (optional)'
    )
    
    parser.add_argument(
        '--output-dir',
        help='Override output directory (optional)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    
    print("=" * 60)
    print("Heavy Optimizer Platform - Pipeline Orchestrator")
    print("=" * 60)
    print(f"Input File: {args.input}")
    print(f"Portfolio Size: {args.portfolio_size}")
    print(f"Execution Mode: {args.mode}")
    print(f"Configuration: {args.config or 'Default'}")
    print("=" * 60)
    
    try:
        # Initialize Pipeline Orchestrator
        print("\nInitializing Pipeline Orchestrator...")
        orchestrator = PipelineOrchestrator(config_path=args.config)
        
        # Validate configuration
        print("Validating configuration...")
        validation = orchestrator.validate_configuration()
        
        if not validation['is_valid']:
            print(f"ERROR: Configuration validation failed:")
            for issue in validation['issues']:
                print(f"  - {issue}")
            return 1
            
        if validation['warnings']:
            print("Configuration warnings:")
            for warning in validation['warnings']:
                print(f"  - {warning}")
        
        # Execute pipeline
        print(f"\nExecuting pipeline in {args.mode} mode...")
        start_time = datetime.now()
        
        results = orchestrator.execute_pipeline(
            input_csv=args.input,
            portfolio_size=args.portfolio_size,
            execution_mode=args.mode,
            output_dir=args.output_dir
        )
        
        # Display results
        print("\n" + "=" * 60)
        print("PIPELINE EXECUTION COMPLETE")
        print("=" * 60)
        print(f"Status: {results['status']}")
        print(f"Total Time: {results['total_execution_time']:.2f} seconds")
        print(f"Output Directory: {results['output_directory']}")
        
        print("\nStep Execution Times:")
        for step, timing in results['step_timings'].items():
            print(f"  {step}: {timing:.3f}s")
            
        if args.mode != 'dry-run':
            print("\nStep Results:")
            for step, result in results['step_results'].items():
                status = result.get('status', 'N/A')
                print(f"  {step}: {status}")
                
        print("\nPipeline execution completed successfully!")
        
        # Show output files if not dry-run
        if args.mode != 'dry-run' and os.path.exists(results['output_directory']):
            print(f"\nGenerated files in {results['output_directory']}:")
            for file in os.listdir(results['output_directory']):
                file_path = os.path.join(results['output_directory'], file)
                if os.path.isfile(file_path):
                    size_mb = os.path.getsize(file_path) / (1024 * 1024)
                    print(f"  - {file} ({size_mb:.2f} MB)")
                    
    except FileNotFoundError as e:
        print(f"\nERROR: File not found - {e}")
        return 1
        
    except ValueError as e:
        print(f"\nERROR: Invalid value - {e}")
        return 1
        
    except Exception as e:
        print(f"\nERROR: Pipeline execution failed - {e}")
        import traceback
        if args.verbose:
            traceback.print_exc()
        return 1
        
    return 0


def demonstrate_programmatic_usage():
    """Demonstrate programmatic usage of Pipeline Orchestrator"""
    
    print("\n" + "=" * 60)
    print("PROGRAMMATIC USAGE EXAMPLE")
    print("=" * 60)
    
    # Create orchestrator
    orchestrator = PipelineOrchestrator()
    
    # Get supported modes
    modes = orchestrator.get_supported_execution_modes()
    print(f"Supported execution modes: {modes}")
    
    # Validate configuration
    validation = orchestrator.validate_configuration()
    print(f"Configuration valid: {validation['is_valid']}")
    
    # Example: Run a dry-run to test the pipeline
    test_csv = "/mnt/optimizer_share/input/Python_Multi_Consolidated_20250726_161921.csv"
    
    if os.path.exists(test_csv):
        print(f"\nRunning dry-run with test data...")
        results = orchestrator.execute_pipeline(
            input_csv=test_csv,
            portfolio_size=10,
            execution_mode='dry-run'
        )
        
        print(f"Dry-run completed in {results['total_execution_time']:.2f}s")
        print(f"Steps executed: {len(results['step_timings'])}")
    else:
        print(f"\nTest CSV not found: {test_csv}")


if __name__ == "__main__":
    # Run the main example
    exit_code = main()
    
    # Optionally demonstrate programmatic usage
    if exit_code == 0 and '--demo' in sys.argv:
        demonstrate_programmatic_usage()
        
    sys.exit(exit_code)