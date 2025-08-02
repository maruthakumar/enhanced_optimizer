"""
Advanced Analytics Engine

Main orchestrator for comprehensive portfolio analytics integration with Heavy Optimizer Platform.
Coordinates all analytics components and integrates with existing workflow systems.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from pathlib import Path
import sys
import os

# Add backend path for imports
sys.path.append('/mnt/optimizer_share/backend')

from .performance_attribution import PerformanceAttributionAnalyzer, AttributionConfig
from .sensitivity_analysis import SensitivityAnalyzer, SensitivityConfig
from .scenario_modeling import ScenarioModeler, ScenarioConfig
from .diversification_analysis import DiversificationAnalyzer, DiversificationConfig
from .risk_metrics import RiskMetricsCalculator, RiskMetricsConfig
from .analytics_visualizer import AnalyticsVisualizer
from .analytics_exporter import AnalyticsExporter

logger = logging.getLogger(__name__)


class AdvancedAnalyticsEngine:
    """
    Main engine for advanced portfolio analytics integration.
    
    Coordinates all analytics components and provides seamless integration 
    with existing Heavy Optimizer Platform workflows for 25,544 strategies.
    """
    
    def __init__(self, output_dir: str = "/mnt/optimizer_share/output",
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize advanced analytics engine
        
        Args:
            output_dir: Directory for analytics outputs
            config: Configuration dictionary for all analytics components
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.config = config or {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Initialize analytics components
        self._initialize_components()
        
        # Cache for expensive calculations
        self.calculation_cache = {}
        
    def _initialize_components(self):
        """Initialize all analytics components with configurations"""
        logger.info("🚀 Initializing advanced analytics components")
        
        # Performance attribution
        attribution_config = AttributionConfig(**self.config.get('attribution', {}))
        self.attribution_analyzer = PerformanceAttributionAnalyzer(attribution_config)
        
        # Sensitivity analysis
        sensitivity_config = SensitivityConfig(**self.config.get('sensitivity', {}))
        self.sensitivity_analyzer = SensitivityAnalyzer(sensitivity_config)
        
        # Scenario modeling
        scenario_config = ScenarioConfig(**self.config.get('scenarios', {}))
        self.scenario_modeler = ScenarioModeler(scenario_config)
        
        # Diversification analysis
        diversification_config = DiversificationConfig(**self.config.get('diversification', {}))
        self.diversification_analyzer = DiversificationAnalyzer(diversification_config)
        
        # Risk metrics
        risk_config = RiskMetricsConfig(**self.config.get('risk', {}))
        self.risk_calculator = RiskMetricsCalculator(risk_config)
        
        # Visualization and export
        self.visualizer = AnalyticsVisualizer(str(self.output_dir))
        self.exporter = AnalyticsExporter(str(self.output_dir))
        
        logger.info("✅ Advanced analytics components initialized")
    
    def analyze_optimization_results(self, optimization_results: Dict[str, Any],
                                   daily_returns: np.ndarray,
                                   strategy_names: List[str],
                                   correlation_matrix: Optional[np.ndarray] = None,
                                   dates: Optional[pd.DatetimeIndex] = None) -> Dict[str, Any]:
        """
        Perform comprehensive analytics on optimization results
        
        Args:
            optimization_results: Results from portfolio optimization
            daily_returns: Daily returns matrix (days x strategies)
            strategy_names: List of strategy names
            correlation_matrix: Pre-calculated correlation matrix
            dates: Trading dates for time-based analysis
            
        Returns:
            Comprehensive analytics results dictionary
        """
        logger.info("🔍 Starting comprehensive analytics analysis")
        
        # Extract portfolio information
        portfolio = optimization_results.get('best_portfolio', [])
        portfolio_weights = optimization_results.get('portfolio_weights')
        
        if portfolio_weights is None:
            # Use equal weights if not provided
            portfolio_weights = np.ones(len(portfolio)) / len(portfolio)
        
        # Calculate correlation matrix if not provided
        if correlation_matrix is None:
            logger.info("📊 Calculating correlation matrix for analysis")
            correlation_matrix = np.corrcoef(daily_returns.T)
        
        analytics_results = {
            'analysis_metadata': {
                'timestamp': datetime.now().isoformat(),
                'portfolio_size': len(portfolio),
                'total_strategies': len(strategy_names),
                'trading_days': daily_returns.shape[0],
                'analysis_components': []
            },
            'strategy_names': strategy_names,
            'correlation_matrix': correlation_matrix
        }
        
        try:
            # Performance Attribution Analysis
            logger.info("📊 Running performance attribution analysis")
            attribution_results = self.attribution_analyzer.analyze_portfolio_attribution(
                portfolio=portfolio,
                portfolio_weights=portfolio_weights,
                daily_returns=daily_returns,
                strategy_names=strategy_names,
                dates=dates
            )
            analytics_results['attribution'] = attribution_results
            analytics_results['analysis_metadata']['analysis_components'].append('attribution')
            
            # Risk Metrics Analysis
            logger.info("⚠️ Running risk metrics analysis")
            risk_results = self.risk_calculator.calculate_comprehensive_risk_metrics(
                daily_returns=daily_returns,
                portfolio=portfolio,
                portfolio_weights=portfolio_weights,
                strategy_names=strategy_names,
                correlation_matrix=correlation_matrix
            )
            analytics_results['risk_metrics'] = risk_results
            analytics_results['analysis_metadata']['analysis_components'].append('risk_metrics')
            
            # Diversification Analysis
            logger.info("🌐 Running diversification analysis")
            diversification_results = self.diversification_analyzer.perform_comprehensive_diversification_analysis(
                portfolio=portfolio,
                portfolio_weights=portfolio_weights,
                daily_returns=daily_returns,
                strategy_names=strategy_names,
                correlation_matrix=correlation_matrix
            )
            analytics_results['diversification'] = diversification_results
            analytics_results['analysis_metadata']['analysis_components'].append('diversification')
            
            # Scenario Modeling
            logger.info("🎭 Running scenario modeling analysis")
            scenario_results = self.scenario_modeler.perform_comprehensive_scenario_analysis(
                daily_returns=daily_returns,
                portfolio=portfolio,
                portfolio_weights=portfolio_weights,
                strategy_names=strategy_names
            )
            analytics_results['scenarios'] = scenario_results
            analytics_results['analysis_metadata']['analysis_components'].append('scenarios')
            
        except Exception as e:
            logger.error(f"❌ Error in analytics analysis: {e}")
            analytics_results['analysis_errors'] = str(e)
        
        logger.info("✅ Comprehensive analytics analysis completed")
        return analytics_results
    
    def perform_sensitivity_analysis(self, optimization_function: Callable,
                                   daily_returns: np.ndarray,
                                   strategy_names: List[str],
                                   correlation_matrix: np.ndarray) -> Dict[str, Any]:
        """
        Perform comprehensive sensitivity analysis on optimization parameters
        
        Args:
            optimization_function: Function to optimize portfolio
            daily_returns: Daily returns matrix
            strategy_names: Strategy names
            correlation_matrix: Strategy correlation matrix
            
        Returns:
            Comprehensive sensitivity analysis results
        """
        logger.info("🔍 Starting comprehensive sensitivity analysis")
        
        try:
            sensitivity_results = self.sensitivity_analyzer.perform_comprehensive_sensitivity_analysis(
                daily_returns=daily_returns,
                strategy_names=strategy_names,
                correlation_matrix=correlation_matrix,
                optimization_function=optimization_function
            )
            
            logger.info("✅ Comprehensive sensitivity analysis completed")
            return sensitivity_results
            
        except Exception as e:
            logger.error(f"❌ Error in sensitivity analysis: {e}")
            return {'error': str(e)}
    
    def generate_comprehensive_report(self, analytics_results: Dict[str, Any],
                                    include_visualizations: bool = True,
                                    include_exports: bool = True) -> Dict[str, Any]:
        """
        Generate comprehensive analytics report with visualizations and exports
        
        Args:
            analytics_results: Results from analytics analysis
            include_visualizations: Whether to generate visualization files
            include_exports: Whether to generate export files
            
        Returns:
            Dictionary with paths to generated files and summary
        """
        logger.info("📋 Generating comprehensive analytics report")
        
        report_results = {
            'report_timestamp': datetime.now().isoformat(),
            'visualizations': {},
            'exports': {},
            'summary': {}
        }
        
        try:
            # Generate visualizations
            if include_visualizations:
                logger.info("🎨 Generating analytics visualizations")
                visualization_paths = self.visualizer.generate_all_visualizations(analytics_results)
                report_results['visualizations'] = visualization_paths
            
            # Generate exports
            if include_exports:
                logger.info("📦 Generating analytics exports")
                export_results = self.exporter.export_all_formats(analytics_results)
                report_results['exports'] = export_results
            
            # Generate summary
            report_results['summary'] = self._generate_report_summary(analytics_results)
            
            logger.info("✅ Comprehensive analytics report generated")
            
        except Exception as e:
            logger.error(f"❌ Error generating comprehensive report: {e}")
            report_results['error'] = str(e)
        
        return report_results
    
    def _generate_report_summary(self, analytics_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate executive summary of analytics results"""
        summary = {
            'portfolio_overview': {},
            'risk_summary': {},
            'performance_highlights': {},
            'key_insights': []
        }
        
        try:
            # Portfolio overview
            if 'attribution' in analytics_results:
                portfolio_summary = analytics_results['attribution'].get('portfolio_summary', {})
                summary['portfolio_overview'] = {
                    'total_return': portfolio_summary.get('total_return', 0),
                    'sharpe_ratio': portfolio_summary.get('sharpe_ratio', 0),
                    'volatility': portfolio_summary.get('volatility', 0),
                    'num_strategies': portfolio_summary.get('num_strategies', 0),
                    'trading_days': portfolio_summary.get('trading_days', 0)
                }
            
            # Risk summary
            if 'risk_metrics' in analytics_results:
                risk_data = analytics_results['risk_metrics']
                var_analysis = risk_data.get('var_analysis', {})
                summary['risk_summary'] = {
                    'var_95': var_analysis.get('var_95', 0),
                    'cvar_95': var_analysis.get('cvar_95', 0),
                    'max_drawdown': risk_data.get('drawdown_analysis', {}).get('max_drawdown', 0),
                    'portfolio_volatility': risk_data.get('individual_risk_contributions', {}).get('portfolio_volatility', 0)
                }
            
            # Performance highlights
            if 'attribution' in analytics_results:
                strategy_contrib = analytics_results['attribution'].get('strategy_contribution', {})
                if strategy_contrib:
                    # Find top performer
                    top_contributor = max(strategy_contrib.items(), 
                                        key=lambda x: x[1].get('total_contribution', 0))
                    summary['performance_highlights'] = {
                        'top_contributor': top_contributor[0],
                        'top_contribution': top_contributor[1].get('total_contribution', 0)
                    }
            
            # Key insights
            insights = []
            
            # Portfolio concentration insight
            if summary['portfolio_overview'].get('num_strategies', 0) > 0:
                concentration = 1 / summary['portfolio_overview']['num_strategies']
                if concentration > 0.1:
                    insights.append(f"Portfolio is moderately concentrated with {summary['portfolio_overview']['num_strategies']} strategies")
                else:
                    insights.append(f"Portfolio is well-diversified with {summary['portfolio_overview']['num_strategies']} strategies")
            
            # Risk-return insight
            sharpe = summary['portfolio_overview'].get('sharpe_ratio', 0)
            if sharpe > 1.0:
                insights.append("Portfolio shows strong risk-adjusted returns (Sharpe > 1.0)")
            elif sharpe > 0.5:
                insights.append("Portfolio shows moderate risk-adjusted returns")
            else:
                insights.append("Portfolio shows low risk-adjusted returns - consider optimization")
            
            # VaR insight
            var_95 = abs(summary['risk_summary'].get('var_95', 0))
            if var_95 > 5:
                insights.append(f"High downside risk: 95% VaR of {var_95:.1f}%")
            elif var_95 > 2:
                insights.append(f"Moderate downside risk: 95% VaR of {var_95:.1f}%")
            else:
                insights.append(f"Low downside risk: 95% VaR of {var_95:.1f}%")
            
            summary['key_insights'] = insights
            
        except Exception as e:
            logger.warning(f"⚠️  Error generating summary: {e}")
            summary['error'] = str(e)
        
        return summary
    
    def integrate_with_workflow(self, workflow_results: Dict[str, Any],
                              run_analytics: bool = True,
                              generate_reports: bool = True) -> Dict[str, Any]:
        """
        Integrate advanced analytics with existing workflow results
        
        Args:
            workflow_results: Results from existing Heavy Optimizer workflow
            run_analytics: Whether to run full analytics suite
            generate_reports: Whether to generate reports and visualizations
            
        Returns:
            Enhanced workflow results with analytics
        """
        logger.info("🔗 Integrating advanced analytics with workflow")
        
        enhanced_results = workflow_results.copy()
        enhanced_results['advanced_analytics'] = {}
        
        try:
            # Extract necessary data from workflow results
            daily_returns = workflow_results.get('daily_returns')
            strategy_names = workflow_results.get('strategy_names', [])
            optimization_results = workflow_results.get('optimization_results', {})
            
            if daily_returns is None or len(strategy_names) == 0:
                logger.warning("⚠️  Insufficient data for advanced analytics")
                enhanced_results['advanced_analytics']['error'] = "Insufficient data"
                return enhanced_results
            
            if run_analytics:
                # Run comprehensive analytics
                analytics_results = self.analyze_optimization_results(
                    optimization_results=optimization_results,
                    daily_returns=daily_returns,
                    strategy_names=strategy_names
                )
                enhanced_results['advanced_analytics']['results'] = analytics_results
                
                if generate_reports:
                    # Generate comprehensive reports
                    report_results = self.generate_comprehensive_report(
                        analytics_results=analytics_results,
                        include_visualizations=True,
                        include_exports=True
                    )
                    enhanced_results['advanced_analytics']['reports'] = report_results
            
            logger.info("✅ Advanced analytics integration completed")
            
        except Exception as e:
            logger.error(f"❌ Error in workflow integration: {e}")
            enhanced_results['advanced_analytics']['error'] = str(e)
        
        return enhanced_results
    
    def create_analytics_summary_for_output(self, analytics_results: Dict[str, Any]) -> str:
        """
        Create a text summary of analytics results for existing output files
        
        Args:
            analytics_results: Results from analytics analysis
            
        Returns:
            Formatted text summary
        """
        try:
            summary_lines = []
            summary_lines.append("=" * 60)
            summary_lines.append("ADVANCED ANALYTICS SUMMARY")
            summary_lines.append("=" * 60)
            summary_lines.append("")
            
            # Portfolio Overview
            if 'attribution' in analytics_results:
                portfolio_summary = analytics_results['attribution'].get('portfolio_summary', {})
                summary_lines.append("PORTFOLIO OVERVIEW:")
                summary_lines.append(f"  Total Return: {portfolio_summary.get('total_return', 0):.2f}%")
                summary_lines.append(f"  Sharpe Ratio: {portfolio_summary.get('sharpe_ratio', 0):.3f}")
                summary_lines.append(f"  Volatility: {portfolio_summary.get('volatility', 0):.2f}%")
                summary_lines.append(f"  Max Drawdown: {portfolio_summary.get('max_drawdown', 0):.2f}%")
                summary_lines.append(f"  Strategies: {portfolio_summary.get('num_strategies', 0)}")
                summary_lines.append("")
            
            # Risk Metrics
            if 'risk_metrics' in analytics_results:
                risk_data = analytics_results['risk_metrics']
                var_analysis = risk_data.get('var_analysis', {})
                summary_lines.append("RISK METRICS:")
                summary_lines.append(f"  VaR (95%): {var_analysis.get('var_95', 0):.2f}%")
                summary_lines.append(f"  CVaR (95%): {var_analysis.get('cvar_95', 0):.2f}%")
                summary_lines.append(f"  VaR (99%): {var_analysis.get('var_99', 0):.2f}%")
                summary_lines.append(f"  CVaR (99%): {var_analysis.get('cvar_99', 0):.2f}%")
                summary_lines.append("")
            
            # Top Contributors
            if 'attribution' in analytics_results:
                strategy_contrib = analytics_results['attribution'].get('strategy_contribution', {})
                if strategy_contrib:
                    summary_lines.append("TOP 5 STRATEGY CONTRIBUTORS:")
                    sorted_contribs = sorted(strategy_contrib.items(), 
                                           key=lambda x: x[1].get('total_contribution', 0), 
                                           reverse=True)[:5]
                    for i, (strategy, contrib) in enumerate(sorted_contribs, 1):
                        contribution = contrib.get('total_contribution', 0)
                        weight = contrib.get('weight', 0) * 100
                        summary_lines.append(f"  {i}. {strategy[:40]}: {contribution:.2f}% (Weight: {weight:.1f}%)")
                    summary_lines.append("")
            
            # Analysis timestamp
            timestamp = analytics_results.get('analysis_metadata', {}).get('timestamp', 'Unknown')
            summary_lines.append(f"Analysis completed: {timestamp}")
            summary_lines.append("=" * 60)
            
            return "\n".join(summary_lines)
            
        except Exception as e:
            logger.error(f"❌ Error creating analytics summary: {e}")
            return f"Error generating analytics summary: {e}"
    
    def get_analytics_status(self) -> Dict[str, Any]:
        """Get status information about analytics components"""
        return {
            'engine_initialized': True,
            'timestamp': datetime.now().isoformat(),
            'components': {
                'attribution_analyzer': bool(self.attribution_analyzer),
                'sensitivity_analyzer': bool(self.sensitivity_analyzer),
                'scenario_modeler': bool(self.scenario_modeler),
                'diversification_analyzer': bool(self.diversification_analyzer),
                'risk_calculator': bool(self.risk_calculator),
                'visualizer': bool(self.visualizer),
                'exporter': bool(self.exporter)
            },
            'output_directory': str(self.output_dir),
            'cache_size': len(self.calculation_cache)
        }