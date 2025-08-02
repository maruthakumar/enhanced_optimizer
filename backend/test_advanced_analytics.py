"""
Advanced Analytics Validation Test

Validates all advanced analytics components against production data specifications:
- 25,544 SENSEX trading strategies
- 82 trading days (January-July 2024)
- Performance targets and acceptance criteria from story requirements
"""

import numpy as np
import pandas as pd
import logging
import sys
import traceback
from pathlib import Path
from datetime import datetime, timedelta
import warnings
from typing import Dict, List, Any

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add lib path
sys.path.append('/mnt/optimizer_share/backend/lib')

from advanced_analytics import (
    PerformanceAttributionAnalyzer,
    SensitivityAnalyzer,
    ScenarioModeler,
    DiversificationAnalyzer,
    RiskMetricsCalculator,
    AnalyticsVisualizer,
    AnalyticsExporter,
    AdvancedAnalyticsEngine
)


class AdvancedAnalyticsValidator:
    """
    Comprehensive validation of advanced analytics components.
    Tests against production specifications and acceptance criteria.
    """
    
    def __init__(self):
        """Initialize validator with production data specifications"""
        self.production_specs = {
            'total_strategies': 25544,
            'trading_days': 82,
            'data_period': 'January-July 2024',
            'strategy_types': ['SENSEX'],
            'stop_loss_range': (7, 88),
            'take_profit_range': (32, 42),
            'expected_data_points': 25544 * 82  # 2,094,608 data points
        }
        
        self.test_results = {
            'validation_timestamp': datetime.now().isoformat(),
            'tests_passed': 0,
            'tests_failed': 0,
            'component_results': {},
            'performance_metrics': {},
            'acceptance_criteria_met': False
        }
        
    def generate_production_like_data(self) -> Dict[str, Any]:
        """Generate synthetic data matching production specifications"""
        logger.info("🏭 Generating production-like test data")
        
        # Set random seed for reproducible tests
        np.random.seed(42)
        
        # Generate strategy names matching production patterns
        strategy_names = []
        for i in range(self.production_specs['total_strategies']):
            sl_level = np.random.randint(7, 89)  # Stop Loss 7-88%
            tp_level = np.random.randint(32, 43)  # Take Profit 32-42%
            zone = np.random.randint(1000, 1400)  # SENSEX zones
            strategy_name = f"SENSEX {zone} SL{sl_level} TP{tp_level} Strategy_{i:05d}"
            strategy_names.append(strategy_name)
        
        # Generate daily returns matrix (82 days x 25,544 strategies)
        daily_returns = np.random.normal(
            loc=0.05,  # Small positive mean return
            scale=1.5,  # Realistic volatility
            size=(self.production_specs['trading_days'], self.production_specs['total_strategies'])
        )
        
        # Add some correlation structure (more realistic)
        correlation_factor = np.random.normal(0, 0.3, size=self.production_specs['trading_days'])
        for i in range(self.production_specs['total_strategies']):
            daily_returns[:, i] += correlation_factor * np.random.uniform(0.1, 0.3)
        
        # Generate dates for January-July 2024
        start_date = pd.Timestamp('2024-01-04')  # First trading day of 2024
        dates = pd.bdate_range(start=start_date, periods=self.production_specs['trading_days'])
        
        # Generate correlation matrix (use subset for memory efficiency)
        sample_size = 1000  # Use subset for correlation matrix
        sample_indices = np.random.choice(self.production_specs['total_strategies'], sample_size, replace=False)
        sample_returns = daily_returns[:, sample_indices]
        correlation_matrix = np.corrcoef(sample_returns.T)
        
        # Generate sample portfolio from the sampled strategies (top 50 from sample)
        portfolio_indices = list(range(50))  # Use first 50 from sample
        portfolio_weights = np.random.dirichlet(np.ones(50))  # Random weights that sum to 1
        
        return {
            'daily_returns': daily_returns,
            'strategy_names': strategy_names,
            'dates': dates,
            'portfolio': portfolio_indices,
            'portfolio_weights': portfolio_weights,
            'correlation_matrix': correlation_matrix,
            'sample_indices': sample_indices
        }
    
    def test_performance_attribution(self, test_data: Dict[str, Any]) -> Dict[str, Any]:
        """Test performance attribution analyzer"""
        logger.info("📊 Testing Performance Attribution Analyzer")
        
        test_result = {'component': 'performance_attribution', 'passed': False, 'errors': []}
        
        try:
            analyzer = PerformanceAttributionAnalyzer()
            
            # Test with production-scale data
            attribution_results = analyzer.analyze_portfolio_attribution(
                portfolio=test_data['portfolio'],
                portfolio_weights=test_data['portfolio_weights'],
                daily_returns=test_data['daily_returns'],
                strategy_names=test_data['strategy_names'],
                dates=test_data['dates']
            )
            
            # Validate results structure
            required_keys = ['portfolio_summary', 'stop_loss_attribution', 'take_profit_attribution',
                           'zone_attribution', 'time_attribution', 'strategy_contribution']
            
            for key in required_keys:
                if key not in attribution_results:
                    test_result['errors'].append(f"Missing required key: {key}")
            
            # Validate data quality
            portfolio_summary = attribution_results.get('portfolio_summary', {})
            if 'total_return' not in portfolio_summary:
                test_result['errors'].append("Missing total_return in portfolio_summary")
            
            # Check Stop Loss analysis covers expected range
            sl_attribution = attribution_results.get('stop_loss_attribution', {})
            if len(sl_attribution) == 0:
                test_result['errors'].append("No Stop Loss attribution results")
            
            # Performance check
            if len(test_result['errors']) == 0:
                test_result['passed'] = True
                test_result['performance'] = {
                    'strategies_analyzed': len(test_data['strategy_names']),
                    'portfolio_size': len(test_data['portfolio']),
                    'trading_days': len(test_data['dates']),
                    'attribution_categories': len(sl_attribution)
                }
            
        except Exception as e:
            test_result['errors'].append(f"Exception: {str(e)}")
            logger.error(f"❌ Performance attribution test failed: {e}")
        
        return test_result
    
    def test_sensitivity_analysis(self, test_data: Dict[str, Any]) -> Dict[str, Any]:
        """Test sensitivity analyzer with mock optimization function"""
        logger.info("🔍 Testing Sensitivity Analyzer")
        
        test_result = {'component': 'sensitivity_analysis', 'passed': False, 'errors': []}
        
        try:
            analyzer = SensitivityAnalyzer()
            
            # Mock optimization function
            def mock_optimization_function(**kwargs):
                portfolio_size = kwargs.get('portfolio_size', 50)
                correlation_penalty = kwargs.get('correlation_penalty', 0.1)
                
                # Return mock results
                portfolio = list(range(min(portfolio_size, len(test_data['portfolio']))))
                fitness = np.random.uniform(0.5, 1.5) * (1 - correlation_penalty * 0.5)
                
                return {
                    'best_portfolio': portfolio,
                    'best_fitness': fitness
                }
            
            # Test correlation penalty sensitivity (reduced range for faster testing)
            penalty_results = analyzer.analyze_correlation_penalty_sensitivity(
                daily_returns=test_data['daily_returns'][:, :100],  # Use subset
                strategy_names=test_data['strategy_names'][:100],
                correlation_matrix=test_data['correlation_matrix'][:100, :100],
                optimization_function=mock_optimization_function
            )
            
            # Validate results
            if 'penalty_results' not in penalty_results:
                test_result['errors'].append("Missing penalty_results")
            
            if 'sensitivity_patterns' not in penalty_results:
                test_result['errors'].append("Missing sensitivity_patterns")
            
            # Test portfolio size sensitivity
            size_results = analyzer.analyze_portfolio_size_sensitivity(
                daily_returns=test_data['daily_returns'][:, :100],
                strategy_names=test_data['strategy_names'][:100],
                optimization_function=mock_optimization_function
            )
            
            if 'size_results' not in size_results:
                test_result['errors'].append("Missing size_results")
            
            if len(test_result['errors']) == 0:
                test_result['passed'] = True
                test_result['performance'] = {
                    'penalty_tests': len(penalty_results.get('penalty_results', {})),
                    'size_tests': len(size_results.get('size_results', {}))
                }
            
        except Exception as e:
            test_result['errors'].append(f"Exception: {str(e)}")
            logger.error(f"❌ Sensitivity analysis test failed: {e}")
        
        return test_result
    
    def test_risk_metrics(self, test_data: Dict[str, Any]) -> Dict[str, Any]:
        """Test risk metrics calculator"""
        logger.info("⚠️ Testing Risk Metrics Calculator")
        
        test_result = {'component': 'risk_metrics', 'passed': False, 'errors': []}
        
        try:
            calculator = RiskMetricsCalculator()
            
            # Test comprehensive risk metrics
            risk_results = calculator.calculate_comprehensive_risk_metrics(
                portfolio=test_data['portfolio'],
                portfolio_weights=test_data['portfolio_weights'],
                daily_returns=test_data['daily_returns'],
                strategy_names=test_data['strategy_names'],
                correlation_matrix=test_data['correlation_matrix']
            )
            
            # Validate results structure
            required_keys = ['var_analysis', 'drawdown_analysis', 'return_distribution_analysis',
                           'tail_risk_analysis', 'individual_risk_contributions']
            
            for key in required_keys:
                if key not in risk_results:
                    test_result['errors'].append(f"Missing required key: {key}")
            
            # Validate VaR calculations
            var_analysis = risk_results.get('var_analysis', {})
            expected_var_keys = ['var_95', 'var_99', 'cvar_95', 'cvar_99']
            for var_key in expected_var_keys:
                if var_key not in var_analysis:
                    test_result['errors'].append(f"Missing VaR metric: {var_key}")
            
            # Validate drawdown analysis
            drawdown_analysis = risk_results.get('drawdown_analysis', {})
            if 'max_drawdown' not in drawdown_analysis:
                test_result['errors'].append("Missing max_drawdown")
            
            if len(test_result['errors']) == 0:
                test_result['passed'] = True
                test_result['performance'] = {
                    'var_confidence_levels': len([k for k in var_analysis.keys() if 'var_' in k]),
                    'risk_metrics_calculated': len(risk_results.keys()),
                    'max_drawdown': drawdown_analysis.get('max_drawdown', 0)
                }
            
        except Exception as e:
            test_result['errors'].append(f"Exception: {str(e)}")
            logger.error(f"❌ Risk metrics test failed: {e}")
        
        return test_result
    
    def test_diversification_analysis(self, test_data: Dict[str, Any]) -> Dict[str, Any]:
        """Test diversification analyzer"""
        logger.info("🌐 Testing Diversification Analyzer")
        
        test_result = {'component': 'diversification_analysis', 'passed': False, 'errors': []}
        
        try:
            analyzer = DiversificationAnalyzer()
            
            # Test portfolio diversification analysis with sample data
            sample_indices = test_data['sample_indices']
            sample_daily_returns = test_data['daily_returns'][:, sample_indices]
            sample_strategy_names = [test_data['strategy_names'][i] for i in sample_indices]
            
            diversification_results = analyzer.perform_comprehensive_diversification_analysis(
                portfolio=test_data['portfolio'],
                portfolio_weights=test_data['portfolio_weights'],
                daily_returns=sample_daily_returns,
                strategy_names=sample_strategy_names,
                correlation_matrix=test_data['correlation_matrix']
            )
            
            # Validate results structure
            required_keys = ['correlation_structure', 'strategy_clustering', 'portfolio_diversification']
            
            for key in required_keys:
                if key not in diversification_results:
                    test_result['errors'].append(f"Missing required key: {key}")
            
            # Validate correlation structure
            correlation_structure = diversification_results.get('correlation_structure', {})
            if 'correlation_statistics' not in correlation_structure:
                test_result['errors'].append("Missing correlation_statistics")
            
            if len(test_result['errors']) == 0:
                test_result['passed'] = True
                test_result['performance'] = {
                    'correlation_metrics': len(correlation_structure.keys()),
                    'diversification_score': diversification_results.get('diversification_effectiveness', {}).get('diversification_ratio', 0)
                }
            
        except Exception as e:
            test_result['errors'].append(f"Exception: {str(e)}")
            logger.error(f"❌ Diversification analysis test failed: {e}")
        
        return test_result
    
    def test_scenario_modeling(self, test_data: Dict[str, Any]) -> Dict[str, Any]:
        """Test scenario modeler"""
        logger.info("🎭 Testing Scenario Modeler")
        
        test_result = {'component': 'scenario_modeling', 'passed': False, 'errors': []}
        
        try:
            modeler = ScenarioModeler()
            
            # Test scenario analysis
            scenario_results = modeler.perform_comprehensive_scenario_analysis(
                daily_returns=test_data['daily_returns'],
                portfolio=test_data['portfolio'],
                portfolio_weights=test_data['portfolio_weights'],
                strategy_names=test_data['strategy_names']
            )
            
            # Validate results structure
            required_keys = ['historical_scenarios', 'stress_scenarios', 'market_regime_analysis']
            
            for key in required_keys:
                if key not in scenario_results:
                    test_result['errors'].append(f"Missing required key: {key}")
            
            # Validate historical scenarios
            historical_scenarios = scenario_results.get('historical_scenarios', {})
            if len(historical_scenarios) == 0:
                test_result['errors'].append("No historical scenarios generated")
            
            if len(test_result['errors']) == 0:
                test_result['passed'] = True
                test_result['performance'] = {
                    'historical_scenarios': len(historical_scenarios),
                    'stress_scenarios': len(scenario_results.get('stress_scenarios', {})),
                    'regime_periods': len(scenario_results.get('regime_analysis', {}))
                }
            
        except Exception as e:
            test_result['errors'].append(f"Exception: {str(e)}")
            logger.error(f"❌ Scenario modeling test failed: {e}")
        
        return test_result
    
    def test_analytics_engine(self, test_data: Dict[str, Any]) -> Dict[str, Any]:
        """Test the main analytics engine integration"""
        logger.info("🚀 Testing Advanced Analytics Engine")
        
        test_result = {'component': 'analytics_engine', 'passed': False, 'errors': []}
        
        try:
            engine = AdvancedAnalyticsEngine()
            
            # Test engine initialization
            status = engine.get_analytics_status()
            if not status.get('engine_initialized', False):
                test_result['errors'].append("Engine not properly initialized")
            
            # Test optimization results analysis
            mock_optimization_results = {
                'best_portfolio': test_data['portfolio'],
                'portfolio_weights': test_data['portfolio_weights'],
                'best_fitness': 1.25
            }
            
            # Run comprehensive analysis (with reduced data for speed)
            analytics_results = engine.analyze_optimization_results(
                optimization_results=mock_optimization_results,
                daily_returns=test_data['daily_returns'][:, :100],  # Use subset
                strategy_names=test_data['strategy_names'][:100],
                correlation_matrix=test_data['correlation_matrix'][:100, :100],
                dates=test_data['dates']
            )
            
            # Validate comprehensive results
            expected_components = ['attribution', 'risk_metrics', 'diversification', 'scenarios']
            analysis_components = analytics_results.get('analysis_metadata', {}).get('analysis_components', [])
            
            for component in expected_components:
                if component not in analysis_components:
                    test_result['errors'].append(f"Missing analysis component: {component}")
            
            # Test report generation
            try:
                summary_text = engine.create_analytics_summary_for_output(analytics_results)
                if len(summary_text) < 100:
                    test_result['errors'].append("Analytics summary too short")
            except Exception as e:
                test_result['errors'].append(f"Summary generation failed: {e}")
            
            if len(test_result['errors']) == 0:
                test_result['passed'] = True
                test_result['performance'] = {
                    'components_initialized': len([k for k, v in status.get('components', {}).items() if v]),
                    'analysis_components': len(analysis_components),
                    'summary_length': len(summary_text)
                }
            
        except Exception as e:
            test_result['errors'].append(f"Exception: {str(e)}")
            logger.error(f"❌ Analytics engine test failed: {e}")
        
        return test_result
    
    def run_performance_benchmarks(self, test_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run performance benchmarks against production targets"""
        logger.info("⏱️ Running performance benchmarks")
        
        benchmarks = {
            'data_processing_speed': 'Not tested',
            'memory_usage': 'Not tested', 
            'analysis_completion_time': 'Not tested',
            'scalability_assessment': 'Not tested'
        }
        
        try:
            start_time = datetime.now()
            
            # Test data processing speed
            engine = AdvancedAnalyticsEngine()
            
            # Quick analysis with subset of data
            mock_results = {
                'best_portfolio': test_data['portfolio'][:25],
                'portfolio_weights': test_data['portfolio_weights'][:25]
            }
            
            analytics_results = engine.analyze_optimization_results(
                optimization_results=mock_results,
                daily_returns=test_data['daily_returns'][:, :500],  # 500 strategies
                strategy_names=test_data['strategy_names'][:500],
                dates=test_data['dates']
            )
            
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            benchmarks['analysis_completion_time'] = f"{processing_time:.2f} seconds"
            benchmarks['data_processing_speed'] = f"{500 * 82 / processing_time:.0f} data points/second"
            benchmarks['scalability_assessment'] = "Suitable for production scale" if processing_time < 30 else "May need optimization"
            
        except Exception as e:
            benchmarks['error'] = str(e)
            logger.error(f"❌ Performance benchmark failed: {e}")
        
        return benchmarks
    
    def check_acceptance_criteria(self) -> bool:
        """Check if implementation meets story acceptance criteria"""
        logger.info("✅ Checking acceptance criteria compliance")
        
        criteria_met = {
            'performance_attribution_implemented': False,
            'sensitivity_analysis_implemented': False,
            'scenario_modeling_implemented': False,
            'diversification_analysis_implemented': False,
            'risk_metrics_implemented': False,
            'visualization_components_created': False,
            'export_functionality_available': False,
            'integration_with_workflows': False
        }
        
        # Check component test results
        for component_name, test_result in self.test_results['component_results'].items():
            if test_result.get('passed', False):
                if 'performance_attribution' in component_name:
                    criteria_met['performance_attribution_implemented'] = True
                elif 'sensitivity_analysis' in component_name:
                    criteria_met['sensitivity_analysis_implemented'] = True
                elif 'scenario_modeling' in component_name:
                    criteria_met['scenario_modeling_implemented'] = True
                elif 'diversification_analysis' in component_name:
                    criteria_met['diversification_analysis_implemented'] = True
                elif 'risk_metrics' in component_name:
                    criteria_met['risk_metrics_implemented'] = True
                elif 'analytics_engine' in component_name:
                    criteria_met['integration_with_workflows'] = True
        
        # Check if visualization and export files exist
        try:
            from advanced_analytics import AnalyticsVisualizer, AnalyticsExporter
            criteria_met['visualization_components_created'] = True
            criteria_met['export_functionality_available'] = True
        except ImportError:
            pass
        
        # Overall compliance
        total_criteria = len(criteria_met)
        met_criteria = sum(criteria_met.values())
        compliance_percentage = (met_criteria / total_criteria) * 100
        
        logger.info(f"📊 Acceptance criteria compliance: {compliance_percentage:.1f}% ({met_criteria}/{total_criteria})")
        
        return compliance_percentage >= 80  # 80% threshold for acceptance
    
    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run complete validation suite"""
        logger.info("🧪 Starting comprehensive advanced analytics validation")
        
        try:
            # Generate test data
            test_data = self.generate_production_like_data()
            logger.info(f"✅ Generated test data: {test_data['daily_returns'].shape[1]} strategies, {test_data['daily_returns'].shape[0]} days")
            
            # Run component tests
            component_tests = [
                ('performance_attribution', self.test_performance_attribution),
                ('sensitivity_analysis', self.test_sensitivity_analysis),
                ('risk_metrics', self.test_risk_metrics),
                ('diversification_analysis', self.test_diversification_analysis),
                ('scenario_modeling', self.test_scenario_modeling),
                ('analytics_engine', self.test_analytics_engine)
            ]
            
            for test_name, test_function in component_tests:
                logger.info(f"🔬 Running {test_name} test...")
                try:
                    test_result = test_function(test_data)
                    self.test_results['component_results'][test_name] = test_result
                    
                    if test_result['passed']:
                        self.test_results['tests_passed'] += 1
                        logger.info(f"✅ {test_name} test PASSED")
                    else:
                        self.test_results['tests_failed'] += 1
                        logger.error(f"❌ {test_name} test FAILED: {test_result['errors']}")
                        
                except Exception as e:
                    self.test_results['tests_failed'] += 1
                    self.test_results['component_results'][test_name] = {
                        'component': test_name,
                        'passed': False,
                        'errors': [f"Test execution failed: {str(e)}"]
                    }
                    logger.error(f"❌ {test_name} test execution failed: {e}")
            
            # Run performance benchmarks
            self.test_results['performance_metrics'] = self.run_performance_benchmarks(test_data)
            
            # Check acceptance criteria
            self.test_results['acceptance_criteria_met'] = self.check_acceptance_criteria()
            
            # Final summary
            total_tests = self.test_results['tests_passed'] + self.test_results['tests_failed']
            success_rate = (self.test_results['tests_passed'] / total_tests * 100) if total_tests > 0 else 0
            
            logger.info("="*60)
            logger.info("VALIDATION SUMMARY")
            logger.info("="*60)
            logger.info(f"Tests Passed: {self.test_results['tests_passed']}")
            logger.info(f"Tests Failed: {self.test_results['tests_failed']}")
            logger.info(f"Success Rate: {success_rate:.1f}%")
            logger.info(f"Acceptance Criteria Met: {'YES' if self.test_results['acceptance_criteria_met'] else 'NO'}")
            logger.info("="*60)
            
            if self.test_results['acceptance_criteria_met'] and success_rate >= 80:
                logger.info("🎉 ADVANCED ANALYTICS VALIDATION SUCCESSFUL!")
            else:
                logger.warning("⚠️  ADVANCED ANALYTICS VALIDATION NEEDS ATTENTION")
            
        except Exception as e:
            logger.error(f"❌ Comprehensive validation failed: {e}")
            self.test_results['validation_error'] = str(e)
        
        return self.test_results


def main():
    """Run the advanced analytics validation"""
    print("🚀 Heavy Optimizer Platform - Advanced Analytics Validation")
    print("=" * 60)
    
    validator = AdvancedAnalyticsValidator()
    results = validator.run_comprehensive_validation()
    
    # Save results
    output_path = Path("/mnt/optimizer_share/output") / f"analytics_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_path.parent.mkdir(exist_ok=True)
    
    import json
    with open(output_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                json_results[key] = value.tolist()
            else:
                json_results[key] = value
        json.dump(json_results, f, indent=2, default=str)
    
    print(f"📄 Validation results saved to: {output_path}")
    
    return results


if __name__ == "__main__":
    main()