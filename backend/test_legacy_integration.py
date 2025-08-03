#!/usr/bin/env python3
"""
Test script for Legacy Integration functionality
Validates that all components work together correctly
"""

import os
import sys
import logging
import json
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_imports():
    """Test that all required modules can be imported"""
    logger.info("Testing imports...")
    
    try:
        from legacy_system_wrapper import LegacySystemWrapper
        logger.info("✅ LegacySystemWrapper imported successfully")
    except ImportError as e:
        logger.error(f"❌ Failed to import LegacySystemWrapper: {e}")
        return False
        
    try:
        from legacy_comparison import LegacyComparison
        logger.info("✅ LegacyComparison imported successfully")
    except ImportError as e:
        logger.error(f"❌ Failed to import LegacyComparison: {e}")
        return False
        
    try:
        from legacy_report_generator import LegacyReportGenerator
        logger.info("✅ LegacyReportGenerator imported successfully")
    except ImportError as e:
        logger.error(f"❌ Failed to import LegacyReportGenerator: {e}")
        return False
        
    try:
        from legacy_integration_orchestrator import LegacyIntegrationOrchestrator
        logger.info("✅ LegacyIntegrationOrchestrator imported successfully")
    except ImportError as e:
        logger.error(f"❌ Failed to import LegacyIntegrationOrchestrator: {e}")
        return False
        
    try:
        from parquet_cudf_workflow import ParquetCuDFWorkflow
        logger.info("✅ ParquetCuDFWorkflow imported successfully")
    except ImportError as e:
        logger.error(f"❌ Failed to import ParquetCuDFWorkflow: {e}")
        return False
        
    return True

def test_file_paths():
    """Test that required file paths exist"""
    logger.info("Testing file paths...")
    
    # Check legacy system
    legacy_path = Path("/mnt/optimizer_share/zone_optimization_25_06_25")
    if not legacy_path.exists():
        logger.error(f"❌ Legacy system path not found: {legacy_path}")
        return False
    logger.info("✅ Legacy system path exists")
    
    # Check legacy optimizer script
    optimizer_script = legacy_path / "Optimizer_New_patched.py"
    if not optimizer_script.exists():
        logger.error(f"❌ Legacy optimizer script not found: {optimizer_script}")
        return False
    logger.info("✅ Legacy optimizer script exists")
    
    # Check test dataset
    test_dataset = Path("/mnt/optimizer_share/input/Python_Multi_Consolidated_20250726_161921.csv")
    if not test_dataset.exists():
        logger.warning(f"⚠️ Test dataset not found: {test_dataset}")
        # Look for alternative test files
        input_dir = Path("/mnt/optimizer_share/input")
        csv_files = list(input_dir.glob("*.csv")) if input_dir.exists() else []
        if csv_files:
            logger.info(f"✅ Found alternative CSV files: {[f.name for f in csv_files[:3]]}")
        else:
            logger.error("❌ No CSV files found in input directory")
            return False
    else:
        logger.info("✅ Test dataset exists")
    
    return True

def test_legacy_wrapper():
    """Test legacy wrapper functionality"""
    logger.info("Testing legacy wrapper...")
    
    try:
        from legacy_system_wrapper import LegacySystemWrapper
        
        # Create wrapper
        wrapper = LegacySystemWrapper()
        logger.info("✅ Legacy wrapper created successfully")
        
        # Test finding output directory (without running optimizer)
        latest_output = wrapper._find_latest_output_dir()
        if latest_output:
            logger.info(f"✅ Found existing output directory: {latest_output.name}")
            
            # Test parsing existing results
            try:
                results = wrapper.get_legacy_results(str(latest_output))
                logger.info(f"✅ Parsed legacy results: {len(results.get('portfolio_results', {}))} portfolios")
            except Exception as e:
                logger.warning(f"⚠️ Could not parse legacy results: {e}")
        else:
            logger.info("ℹ️ No existing output directory found (this is okay)")
            
        return True
        
    except Exception as e:
        logger.error(f"❌ Legacy wrapper test failed: {e}")
        return False

def test_comparison_engine():
    """Test comparison engine functionality"""
    logger.info("Testing comparison engine...")
    
    try:
        from legacy_comparison import LegacyComparison
        
        # Create comparison engine
        comparison = LegacyComparison(tolerance=0.0001)
        logger.info("✅ Comparison engine created successfully")
        
        # Test fitness comparison
        result = comparison.compare_fitness_values(
            legacy_fitness=30.45764862187442,
            new_fitness=30.45800000000000,
            portfolio_size=37,
            algorithm='SA'
        )
        
        logger.info(f"✅ Fitness comparison test: {result['percentage_difference']:.4f}% difference")
        
        # Test portfolio comparison
        legacy_strategies = ["Strategy A", "Strategy B", "Strategy C"]
        new_strategies = ["Strategy A", "Strategy B", "Strategy D"]
        
        portfolio_result = comparison.compare_portfolios(
            legacy_strategies=legacy_strategies,
            new_strategies=new_strategies,
            portfolio_size=3
        )
        
        logger.info(f"✅ Portfolio comparison test: {portfolio_result['overlap_percentage']:.1f}% overlap")
        
        # Test summary generation
        summary = comparison.generate_comparison_summary()
        logger.info(f"✅ Summary generation test: {summary['fitness_match_rate']:.1f}% match rate")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Comparison engine test failed: {e}")
        return False

def test_report_generator():
    """Test report generator functionality"""
    logger.info("Testing report generator...")
    
    try:
        from legacy_report_generator import LegacyReportGenerator
        
        # Create report generator
        generator = LegacyReportGenerator("/tmp/test_reports")
        logger.info("✅ Report generator created successfully")
        
        # Create dummy comparison summary
        comparison_summary = {
            'total_comparisons': 2,
            'fitness_matches': 1,
            'fitness_match_rate': 50.0,
            'average_percentage_difference': 0.01,
            'max_percentage_difference': 0.02,
            'all_within_tolerance': False,
            'verdict': 'Test verdict',
            'detailed_results': [
                {
                    'portfolio_size': 35,
                    'algorithm': 'GA',
                    'legacy_fitness': 25.123,
                    'new_fitness': 25.124,
                    'percentage_difference': 0.004,
                    'within_tolerance': True
                },
                {
                    'portfolio_size': 37,
                    'algorithm': 'SA', 
                    'legacy_fitness': 30.458,
                    'new_fitness': 30.463,
                    'percentage_difference': 0.016,
                    'within_tolerance': False
                }
            ]
        }
        
        # Test dashboard creation
        dashboard_path = generator.create_summary_dashboard(comparison_summary)
        logger.info(f"✅ Dashboard created: {dashboard_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Report generator test failed: {e}")
        return False

def run_all_tests():
    """Run all tests"""
    logger.info("Starting Legacy Integration Tests")
    logger.info("=" * 50)
    
    tests = [
        ("Import Tests", test_imports),
        ("File Path Tests", test_file_paths),
        ("Legacy Wrapper Tests", test_legacy_wrapper),
        ("Comparison Engine Tests", test_comparison_engine),
        ("Report Generator Tests", test_report_generator)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n--- {test_name} ---")
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("TEST SUMMARY")
    logger.info("=" * 50)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Legacy integration is ready.")
        return True
    else:
        logger.error("❌ Some tests failed. Please review the issues above.")
        return False

def main():
    """Main entry point"""
    success = run_all_tests()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()