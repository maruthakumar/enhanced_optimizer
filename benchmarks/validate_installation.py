#!/usr/bin/env python3
"""
Benchmark Framework Installation Validator
Verifies that all components are properly installed and functional
"""

import os
import sys
import json
import importlib
import subprocess
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_directory_structure():
    """Verify benchmark directory structure exists."""
    logger.info("🔍 Checking directory structure...")
    
    base_dir = Path(__file__).parent
    required_dirs = [
        'config',
        'scripts',
        'data', 
        'reports',
        'reports/ci_artifacts'
    ]
    
    missing_dirs = []
    for dir_name in required_dirs:
        dir_path = base_dir / dir_name
        if not dir_path.exists():
            missing_dirs.append(str(dir_path))
        else:
            logger.info(f"  ✅ {dir_name}/ exists")
    
    if missing_dirs:
        logger.error(f"❌ Missing directories: {missing_dirs}")
        return False
    
    logger.info("✅ Directory structure validation passed")
    return True

def check_required_files():
    """Verify required configuration and script files exist."""
    logger.info("🔍 Checking required files...")
    
    base_dir = Path(__file__).parent
    required_files = [
        'config/benchmark_config.json',
        'scripts/generate_test_data.py',
        'scripts/parquet_pipeline_benchmark.py',
        'scripts/generate_report.py',
        'run_benchmark.py',
        'README.md'
    ]
    
    missing_files = []
    for file_name in required_files:
        file_path = base_dir / file_name
        if not file_path.exists():
            missing_files.append(str(file_path))
        else:
            logger.info(f"  ✅ {file_name} exists")
    
    if missing_files:
        logger.error(f"❌ Missing files: {missing_files}")
        return False
    
    logger.info("✅ Required files validation passed")
    return True

def check_python_dependencies():
    """Check if required Python packages are available."""
    logger.info("🔍 Checking Python dependencies...")
    
    required_packages = [
        'pandas',
        'numpy', 
        'json',
        'datetime',
        'pathlib',
        'logging',
        'argparse',
        'subprocess',
        'tempfile',
        'psutil'
    ]
    
    optional_packages = [
        'pyarrow',
        'matplotlib',
        'cudf'  # GPU acceleration (optional)
    ]
    
    missing_required = []
    missing_optional = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            logger.info(f"  ✅ {package} available")
        except ImportError:
            missing_required.append(package)
            logger.error(f"  ❌ {package} missing (required)")
    
    for package in optional_packages:
        try:
            if package == 'cudf':
                # Special handling for cuDF - it may fail with CUDA errors
                try:
                    importlib.import_module(package)
                    logger.info(f"  ✅ {package} available")
                except Exception as e:
                    if "libcudart" in str(e) or "CUDA" in str(e):
                        logger.info(f"  ⚠️  {package} not available (no CUDA support - this is normal on CPU-only systems)")
                    else:
                        logger.warning(f"  ⚠️  {package} import failed: {e}")
                    missing_optional.append(package)
            else:
                importlib.import_module(package)
                logger.info(f"  ✅ {package} available")
        except ImportError:
            missing_optional.append(package)
            logger.warning(f"  ⚠️  {package} missing (optional)")
    
    if missing_required:
        logger.error(f"❌ Missing required packages: {missing_required}")
        return False
    
    if missing_optional:
        logger.warning(f"⚠️  Missing optional packages: {missing_optional}")
    
    logger.info("✅ Python dependencies validation passed")
    return True

def check_configuration_validity():
    """Validate benchmark configuration file."""
    logger.info("🔍 Checking configuration validity...")
    
    config_path = Path(__file__).parent / 'config' / 'benchmark_config.json'
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Check required configuration sections
        required_sections = [
            'dataset_config',
            'performance_thresholds', 
            'benchmark_scenarios'
        ]
        
        for section in required_sections:
            if section not in config:
                logger.error(f"❌ Missing configuration section: {section}")
                return False
            logger.info(f"  ✅ {section} section exists")
        
        # Validate scenarios
        scenarios = config.get('benchmark_scenarios', [])
        if len(scenarios) == 0:
            logger.error("❌ No benchmark scenarios configured")
            return False
        
        logger.info(f"  ✅ {len(scenarios)} benchmark scenarios configured")
        
        # Check scenario structure
        for i, scenario in enumerate(scenarios):
            required_fields = ['name', 'strategy_count', 'trading_days', 'target_time_ms']
            for field in required_fields:
                if field not in scenario:
                    logger.error(f"❌ Scenario {i} missing field: {field}")
                    return False
        
        logger.info("✅ Configuration validation passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Configuration validation failed: {e}")
        return False

def test_data_generator():
    """Test the data generation functionality."""
    logger.info("🔍 Testing data generator...")
    
    scripts_dir = Path(__file__).parent / 'scripts'
    
    try:
        # Test small dataset generation
        cmd = [
            sys.executable,
            str(scripts_dir / 'generate_test_data.py'),
            '--strategies', '50',
            '--days', '5',
            '--output', 'validation_test.csv'
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=scripts_dir,
            timeout=30
        )
        
        if result.returncode != 0:
            logger.error(f"❌ Data generator test failed: {result.stderr}")
            return False
        
        # Check if file was created
        data_dir = Path(__file__).parent / 'data'
        test_file = data_dir / 'validation_test.csv'
        
        if not test_file.exists():
            logger.error("❌ Test data file was not created")
            return False
        
        # Check file size
        file_size = test_file.stat().st_size
        if file_size < 1000:  # Should be at least 1KB for 50 strategies x 5 days
            logger.error(f"❌ Test data file is too small: {file_size} bytes")
            return False
        
        # Cleanup test file
        test_file.unlink()
        metadata_file = data_dir / 'validation_test_metadata.json'
        if metadata_file.exists():
            metadata_file.unlink()
        
        logger.info("✅ Data generator test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Data generator test failed: {e}")
        return False

def test_script_executability():
    """Test that scripts are executable and have basic syntax validity."""
    logger.info("🔍 Testing script executability...")
    
    scripts_dir = Path(__file__).parent / 'scripts'
    main_script = Path(__file__).parent / 'run_benchmark.py'
    
    scripts_to_test = [
        scripts_dir / 'generate_test_data.py',
        scripts_dir / 'parquet_pipeline_benchmark.py', 
        scripts_dir / 'generate_report.py',
        main_script
    ]
    
    for script in scripts_to_test:
        try:
            # Test syntax by compiling
            with open(script, 'r') as f:
                code = f.read()
            
            compile(code, str(script), 'exec')
            logger.info(f"  ✅ {script.name} syntax valid")
            
            # Test help option
            result = subprocess.run(
                [sys.executable, str(script), '--help'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode != 0:
                logger.warning(f"  ⚠️  {script.name} --help failed (may be normal)")
            else:
                logger.info(f"  ✅ {script.name} --help works")
        
        except Exception as e:
            logger.error(f"❌ Script {script.name} test failed: {e}")
            return False
    
    logger.info("✅ Script executability test passed")
    return True

def check_backend_integration():
    """Check if benchmark can integrate with backend components."""
    logger.info("🔍 Checking backend integration...")
    
    # Add backend to path
    backend_dir = Path(__file__).parent.parent / 'backend'
    if not backend_dir.exists():
        logger.info("ℹ️  Backend directory not found - integration tests skipped (normal for benchmark-only installation)")
        return True
    
    # Temporarily add backend to path
    original_path = sys.path.copy()
    sys.path.insert(0, str(backend_dir))
    
    try:
        # Check for basic backend structure first
        algorithms_dir = backend_dir / 'algorithms'
        if not algorithms_dir.exists():
            logger.info("ℹ️  Algorithms directory not found - backend integration not available")
            return True
        
        # Test importing key backend components with better error handling
        backend_components = [
            ('algorithms.base_algorithm', 'BaseOptimizationAlgorithm'),
            ('algorithms.genetic_algorithm', 'GeneticAlgorithm')
        ]
        
        available_components = []
        for module_name, class_name in backend_components:
            try:
                module = importlib.import_module(module_name)
                if hasattr(module, class_name):
                    available_components.append(module_name)
                    logger.info(f"  ✅ {module_name}.{class_name} available")
                else:
                    logger.info(f"  ℹ️  {module_name} exists but {class_name} not found")
            except ImportError as e:
                # Don't log missing imports as warnings - they're optional
                logger.info(f"  ℹ️  {module_name} not available: {str(e)[:50]}...")
            except Exception as e:
                # Handle other errors gracefully
                logger.info(f"  ℹ️  {module_name} import failed: {str(e)[:50]}...")
        
        if len(available_components) > 0:
            logger.info(f"✅ Backend integration available ({len(available_components)} components found)")
        else:
            logger.info("ℹ️  Backend components not available - benchmark will run in standalone mode")
        
        logger.info("✅ Backend integration check completed successfully")
        return True
        
    except Exception as e:
        logger.info(f"ℹ️  Backend integration check completed with minor issues: {str(e)[:100]}...")
        return True  # Always return True since backend integration is optional
    
    finally:
        # Restore original path
        sys.path[:] = original_path

def run_validation():
    """Run complete validation suite."""
    logger.info("🚀 Starting Benchmark Framework Validation")
    logger.info("=" * 60)
    
    checks = [
        ("Directory Structure", check_directory_structure),
        ("Required Files", check_required_files),
        ("Python Dependencies", check_python_dependencies),
        ("Configuration Validity", check_configuration_validity),
        ("Data Generator", test_data_generator),
        ("Script Executability", test_script_executability),
        ("Backend Integration", check_backend_integration)
    ]
    
    passed = 0
    total = len(checks)
    
    for check_name, check_func in checks:
        logger.info(f"\n📋 Running check: {check_name}")
        try:
            if check_func():
                passed += 1
            else:
                logger.error(f"❌ {check_name} failed")
        except Exception as e:
            logger.error(f"❌ {check_name} failed with exception: {e}")
    
    logger.info("\n" + "=" * 60)
    logger.info("🎯 VALIDATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Checks passed: {passed}/{total}")
    
    if passed == total:
        logger.info("🎉 ALL VALIDATIONS PASSED!")
        logger.info("✅ Benchmark framework is ready for use")
        logger.info("\nNext steps:")
        logger.info("  1. Run: python run_benchmark.py --scenario micro_dataset")
        logger.info("  2. Check reports in: benchmarks/reports/")
        logger.info("  3. Review README.md for full usage guide")
        return True
    else:
        logger.error("❌ SOME VALIDATIONS FAILED!")
        logger.error(f"Please fix the {total - passed} failed checks before using the framework")
        return False

if __name__ == "__main__":
    success = run_validation()
    sys.exit(0 if success else 1)