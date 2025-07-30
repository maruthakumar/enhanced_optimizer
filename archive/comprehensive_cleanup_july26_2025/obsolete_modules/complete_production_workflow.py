#!/usr/bin/env python3
"""
Complete Production Workflow System
Integrates all components: parallel execution, output generation, monitoring, and error handling
"""

import os
import sys
import time
import json
import logging
import traceback
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# Import our custom components
from parallel_algorithm_orchestrator import ParallelAlgorithmOrchestrator
from output_generation_engine import OutputGenerationEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/mnt/optimizer_share/output/workflow.log')
    ]
)
logger = logging.getLogger(__name__)

class CompleteProductionWorkflow:
    def __init__(self):
        self.orchestrator = ParallelAlgorithmOrchestrator()
        self.output_engine = OutputGenerationEngine()
        self.workflow_results = {
            'workflow_start_time': datetime.now().isoformat(),
            'data_processing': {},
            'optimization_results': {},
            'output_generation': {},
            'error_log': []
        }
        
    def execute_complete_workflow(self, excel_file_path: str, 
                                portfolio_size: int = 35,
                                output_prefix: str = "optimization") -> Dict[str, Any]:
        """Execute the complete production workflow"""
        logger.info("🚀 Starting Complete Production Workflow")
        logger.info("=" * 80)
        
        try:
            # Step 1: Data Processing and Validation
            logger.info("📊 Step 1: Data Processing and Validation")
            daily_matrix, strategy_columns = self.process_excel_data(excel_file_path)
            if daily_matrix is None:
                raise RuntimeError("Data processing failed")
            
            # Step 2: Dynamic HeavyDB Table Creation (if needed)
            logger.info("🏗️ Step 2: Dynamic Table Creation Validation")
            table_validation = self.validate_heavydb_integration()
            
            # Step 3: Parallel Algorithm Execution
            logger.info("⚡ Step 3: Parallel Algorithm Execution")
            optimization_results = self.orchestrator.execute_all_algorithms_parallel(
                daily_matrix, strategy_columns, portfolio_size
            )
            self.workflow_results['optimization_results'] = optimization_results
            
            # Step 4: Output Generation
            logger.info("📊 Step 4: Comprehensive Output Generation")
            output_files = self.output_engine.generate_comprehensive_output(
                optimization_results, daily_matrix, strategy_columns
            )
            self.workflow_results['output_generation'] = output_files
            
            # Step 5: Workflow Summary
            logger.info("📋 Step 5: Workflow Summary Generation")
            workflow_summary = self.generate_workflow_summary()
            
            # Step 6: Final Validation
            logger.info("✅ Step 6: Final Validation")
            validation_results = self.validate_workflow_completion()
            
            # Compile final results
            final_results = {
                'workflow_status': 'SUCCESS',
                'execution_time': time.time() - time.mktime(
                    datetime.fromisoformat(self.workflow_results['workflow_start_time']).timetuple()
                ),
                'data_processing': self.workflow_results['data_processing'],
                'optimization_results': optimization_results,
                'output_files': output_files,
                'workflow_summary': workflow_summary,
                'validation_results': validation_results
            }
            
            # Save complete workflow results
            results_file = f"/mnt/optimizer_share/output/complete_workflow_{int(time.time())}.json"
            with open(results_file, 'w') as f:
                json.dump(final_results, f, indent=2)
            
            logger.info("=" * 80)
            logger.info("🎉 COMPLETE PRODUCTION WORKFLOW SUCCESSFUL")
            logger.info(f"📄 Results saved to: {results_file}")
            logger.info(f"🏆 Best algorithm: {optimization_results.get('best_algorithm', 'None')}")
            logger.info(f"📈 Best fitness: {optimization_results.get('best_fitness', 0):.6f}")
            logger.info(f"⚡ Total execution time: {final_results['execution_time']:.2f}s")
            
            return final_results
            
        except Exception as e:
            logger.error(f"❌ Workflow failed: {e}")
            logger.error(f"❌ Traceback: {traceback.format_exc()}")
            
            error_result = {
                'workflow_status': 'FAILED',
                'error': str(e),
                'traceback': traceback.format_exc(),
                'partial_results': self.workflow_results
            }
            
            # Save error results
            error_file = f"/mnt/optimizer_share/output/workflow_error_{int(time.time())}.json"
            with open(error_file, 'w') as f:
                json.dump(error_result, f, indent=2)
            
            return error_result
    
    def process_excel_data(self, excel_file_path: str) -> tuple:
        """Process Excel data with validation and error handling"""
        logger.info(f"📊 Processing Excel file: {excel_file_path}")
        
        try:
            # Validate file exists
            if not os.path.exists(excel_file_path):
                raise FileNotFoundError(f"Excel file not found: {excel_file_path}")
            
            # Load Excel file
            df = pd.read_excel(excel_file_path)
            logger.info(f"✅ Excel file loaded: {len(df)} rows, {len(df.columns)} columns")
            
            # Extract strategy columns
            reserved_columns = ['Date', 'Day', 'Zone', 'date', 'day', 'zone']
            strategy_columns = [col for col in df.columns if col not in reserved_columns]
            
            # Validate data quality
            if len(strategy_columns) == 0:
                raise ValueError("No strategy columns found in Excel file")
            
            if len(df) == 0:
                raise ValueError("Excel file contains no data rows")
            
            # Create daily returns matrix
            daily_matrix = np.zeros((len(df), len(strategy_columns)))
            for i, col in enumerate(strategy_columns):
                daily_matrix[:, i] = pd.to_numeric(df[col], errors='coerce').fillna(0).values
            
            # Data quality checks
            non_zero_strategies = np.sum(np.any(daily_matrix != 0, axis=0))
            data_completeness = np.mean(~np.isnan(daily_matrix)) * 100
            
            self.workflow_results['data_processing'] = {
                'file_path': excel_file_path,
                'file_size_mb': os.path.getsize(excel_file_path) / (1024 * 1024),
                'total_rows': len(df),
                'total_strategies': len(strategy_columns),
                'non_zero_strategies': int(non_zero_strategies),
                'data_completeness_percent': float(data_completeness),
                'data_shape': daily_matrix.shape,
                'validation_status': 'PASSED'
            }
            
            logger.info(f"✅ Data processing completed:")
            logger.info(f"   Strategies: {len(strategy_columns):,}")
            logger.info(f"   Trading days: {len(df)}")
            logger.info(f"   Data completeness: {data_completeness:.1f}%")
            
            return daily_matrix, strategy_columns
            
        except Exception as e:
            logger.error(f"❌ Data processing failed: {e}")
            self.workflow_results['data_processing'] = {
                'file_path': excel_file_path,
                'validation_status': 'FAILED',
                'error': str(e)
            }
            return None, None
    
    def validate_heavydb_integration(self) -> Dict[str, Any]:
        """Validate HeavyDB integration and GPU mode"""
        logger.info("🔍 Validating HeavyDB integration...")
        
        try:
            # Check HeavyDB GPU mode
            import subprocess
            gpu_check = subprocess.run([
                'sudo', 'grep', '-i', 'Started in GPU mode',
                '/var/lib/heavyai/storage/log/heavydb.INFO'
            ], capture_output=True, text=True)
            
            heavydb_gpu_active = gpu_check.returncode == 0
            
            # Check A100 GPU status
            gpu_status = subprocess.run([
                'nvidia-smi', '--query-gpu=name,memory.used',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True)
            
            gpu_available = gpu_status.returncode == 0
            
            validation_result = {
                'heavydb_gpu_mode': heavydb_gpu_active,
                'a100_gpu_available': gpu_available,
                'integration_status': 'VALIDATED' if (heavydb_gpu_active and gpu_available) else 'WARNING'
            }
            
            if gpu_available:
                gpu_info = gpu_status.stdout.strip().split(', ')
                validation_result['gpu_name'] = gpu_info[0]
                validation_result['gpu_memory_used'] = int(gpu_info[1])
            
            logger.info(f"✅ HeavyDB integration validated:")
            logger.info(f"   GPU Mode: {'✅' if heavydb_gpu_active else '❌'}")
            logger.info(f"   A100 Available: {'✅' if gpu_available else '❌'}")
            
            return validation_result
            
        except Exception as e:
            logger.error(f"❌ HeavyDB validation failed: {e}")
            return {'integration_status': 'FAILED', 'error': str(e)}
    
    def generate_workflow_summary(self) -> Dict[str, Any]:
        """Generate comprehensive workflow summary"""
        logger.info("📋 Generating workflow summary...")
        
        try:
            optimization_results = self.workflow_results.get('optimization_results', {})
            data_processing = self.workflow_results.get('data_processing', {})
            
            summary = {
                'execution_metadata': {
                    'workflow_start_time': self.workflow_results['workflow_start_time'],
                    'workflow_end_time': datetime.now().isoformat(),
                    'total_execution_time': time.time() - time.mktime(
                        datetime.fromisoformat(self.workflow_results['workflow_start_time']).timetuple()
                    )
                },
                'data_summary': {
                    'strategies_processed': data_processing.get('total_strategies', 0),
                    'trading_days': data_processing.get('total_rows', 0),
                    'data_quality': data_processing.get('data_completeness_percent', 0)
                },
                'optimization_summary': {
                    'algorithms_executed': optimization_results.get('algorithms_executed', 0),
                    'success_rate': optimization_results.get('success_rate', 0),
                    'best_algorithm': optimization_results.get('best_algorithm', 'None'),
                    'best_fitness': optimization_results.get('best_fitness', 0),
                    'parallel_efficiency': optimization_results.get('parallel_efficiency', 0)
                },
                'output_summary': {
                    'files_generated': len(self.workflow_results.get('output_generation', {})),
                    'output_types': list(self.workflow_results.get('output_generation', {}).keys())
                },
                'performance_metrics': {
                    'data_processing_time': 'Included in total time',
                    'optimization_time': optimization_results.get('total_execution_time', 0),
                    'output_generation_time': 'Included in total time'
                }
            }
            
            logger.info("✅ Workflow summary generated")
            return summary
            
        except Exception as e:
            logger.error(f"❌ Workflow summary generation failed: {e}")
            return {'error': str(e)}
    
    def validate_workflow_completion(self) -> Dict[str, Any]:
        """Validate that all workflow components completed successfully"""
        logger.info("✅ Validating workflow completion...")
        
        try:
            validation_checks = {
                'data_processing_completed': 'data_processing' in self.workflow_results and 
                                           self.workflow_results['data_processing'].get('validation_status') == 'PASSED',
                'optimization_completed': 'optimization_results' in self.workflow_results and
                                        self.workflow_results['optimization_results'].get('algorithms_executed', 0) > 0,
                'output_generation_completed': 'output_generation' in self.workflow_results and
                                             len(self.workflow_results['output_generation']) > 0,
                'best_result_available': self.workflow_results.get('optimization_results', {}).get('best_algorithm') is not None
            }
            
            all_checks_passed = all(validation_checks.values())
            
            validation_result = {
                'overall_status': 'PASSED' if all_checks_passed else 'FAILED',
                'individual_checks': validation_checks,
                'completion_percentage': (sum(validation_checks.values()) / len(validation_checks)) * 100
            }
            
            logger.info(f"✅ Workflow validation completed:")
            logger.info(f"   Overall status: {'✅ PASSED' if all_checks_passed else '❌ FAILED'}")
            logger.info(f"   Completion: {validation_result['completion_percentage']:.1f}%")
            
            return validation_result
            
        except Exception as e:
            logger.error(f"❌ Workflow validation failed: {e}")
            return {'overall_status': 'ERROR', 'error': str(e)}


def main():
    """Main execution function for testing"""
    logger.info("🧪 Testing Complete Production Workflow")
    
    # Test with production SENSEX dataset
    excel_file = "/mnt/optimizer_share/input/SENSEX_test_dataset.xlsx"
    
    if os.path.exists(excel_file):
        workflow = CompleteProductionWorkflow()
        results = workflow.execute_complete_workflow(excel_file, portfolio_size=35)
        
        if results.get('workflow_status') == 'SUCCESS':
            print("\n🎉 COMPLETE WORKFLOW TEST SUCCESSFUL")
            print(f"Best Algorithm: {results['optimization_results'].get('best_algorithm', 'None')}")
            print(f"Best Fitness: {results['optimization_results'].get('best_fitness', 0):.6f}")
            print(f"Output Files: {len(results['output_files'])}")
            print(f"Total Time: {results['execution_time']:.2f}s")
        else:
            print("\n❌ WORKFLOW TEST FAILED")
            print(f"Error: {results.get('error', 'Unknown error')}")
    else:
        print("❌ Test dataset not found")

if __name__ == "__main__":
    main()
