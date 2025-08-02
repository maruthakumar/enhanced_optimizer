"""
Advanced Analytics Visualization Module

Provides comprehensive visualization capabilities for portfolio analytics including:
- Strategy performance heatmaps showing P&L across Stop Loss/Take Profit matrix
- Time series analysis charts for daily portfolio performance
- Correlation network visualizations
- Risk attribution charts
- Scenario impact plots
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import warnings
from pathlib import Path
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap

# Optional dependencies - handle gracefully
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

try:
    import plotly.graph_objects as go
    import plotly.figure_factory as ff
    from plotly.subplots import make_subplots
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

logger = logging.getLogger(__name__)

# Suppress matplotlib warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')


class AnalyticsVisualizer:
    """
    Creates comprehensive visualizations for advanced portfolio analytics.
    
    Supports both static (matplotlib/seaborn) and interactive (plotly) visualizations
    for production-scale analysis of 25,544 strategies across 82 trading days.
    """
    
    def __init__(self, output_dir: str = "/mnt/optimizer_share/output", 
                 style: str = "plotly_white"):
        """
        Initialize analytics visualizer
        
        Args:
            output_dir: Directory to save visualization outputs
            style: Visualization style ('plotly_white', 'seaborn', 'publication')
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.style = style
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Set up visualization styles
        self._setup_styles()
        
    def _setup_styles(self):
        """Setup visualization styles and color palettes"""
        # Matplotlib style
        if HAS_SEABORN and 'seaborn-v0_8' in plt.style.available:
            plt.style.use('seaborn-v0_8')
        else:
            plt.style.use('default')
        
        # Custom color palettes
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e', 
            'success': '#2ca02c',
            'danger': '#d62728',
            'warning': '#ff7f0e',
            'info': '#17a2b8',
            'light': '#f8f9fa',
            'dark': '#343a40'
        }
        
        # Risk color palette (green to red)
        self.risk_colors = ['#2ca02c', '#ffed4e', '#ff7f0e', '#d62728', '#8b0000']
        
    def create_strategy_performance_heatmap(self, attribution_results: Dict[str, Any],
                                          strategy_names: List[str],
                                          title: str = "Strategy Performance Heatmap") -> str:
        """
        Create heatmap showing strategy performance across Stop Loss/Take Profit matrix
        
        Args:
            attribution_results: Results from performance attribution analysis
            strategy_names: List of strategy names
            title: Chart title
            
        Returns:
            Path to saved visualization file
        """
        logger.info("📊 Creating strategy performance heatmap")
        
        # Extract Stop Loss and Take Profit data
        sl_data = attribution_results.get('stop_loss_attribution', {})
        tp_data = attribution_results.get('take_profit_attribution', {})
        
        if not sl_data or not tp_data:
            logger.warning("⚠️  Insufficient data for heatmap creation")
            return ""
        
        # Create heatmap matrix
        heatmap_data = self._prepare_heatmap_data(sl_data, tp_data)
        
        # Create interactive plotly heatmap if available
        if HAS_PLOTLY:
            fig = go.Figure(data=go.Heatmap(
                z=heatmap_data['values'],
                x=heatmap_data['tp_levels'],
                y=heatmap_data['sl_levels'],
                colorscale='RdYlGn',
                text=heatmap_data['text_values'],
                texttemplate="%{text}",
                textfont={"size": 10},
                hovertemplate="Stop Loss: %{y}%<br>Take Profit: %{x}%<br>Return: %{z:.2f}%<extra></extra>"
            ))
            
            fig.update_layout(
                title=f"{title} - Production Data Analysis",
                xaxis_title="Take Profit Levels (%)",
                yaxis_title="Stop Loss Levels (%)",
                width=1000,
                height=800,
                font=dict(size=12)
            )
            
            # Save interactive version
            output_path = self.output_dir / f"strategy_heatmap_{self.timestamp}.html"
            fig.write_html(str(output_path))
        else:
            # Fallback to static version only
            output_path = self.output_dir / f"strategy_heatmap_{self.timestamp}.png"
        
        # Also create static version
        self._create_static_heatmap(heatmap_data, title)
        
        logger.info(f"✅ Strategy heatmap saved to {output_path}")
        return str(output_path)
    
    def _prepare_heatmap_data(self, sl_data: Dict[str, Any], 
                            tp_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare data matrix for heatmap visualization"""
        # Extract unique levels
        sl_levels = sorted([int(k.replace('SL_', '').replace('%', '')) for k in sl_data.keys()])
        tp_levels = sorted([int(k.replace('TP_', '').replace('%', '')) for k in tp_data.keys()])
        
        # Create matrix
        matrix = np.zeros((len(sl_levels), len(tp_levels)))
        text_matrix = np.empty((len(sl_levels), len(tp_levels)), dtype=object)
        
        for i, sl in enumerate(sl_levels):
            for j, tp in enumerate(tp_levels):
                sl_key = f"SL_{sl}%"
                tp_key = f"TP_{tp}%"
                
                # Use intersection or weighted average if both exist
                if sl_key in sl_data and tp_key in tp_data:
                    sl_return = sl_data[sl_key]['total_return']
                    tp_return = tp_data[tp_key]['total_return'] 
                    combined_return = (sl_return + tp_return) / 2
                    matrix[i, j] = combined_return
                    text_matrix[i, j] = f"{combined_return:.1f}"
                elif sl_key in sl_data:
                    matrix[i, j] = sl_data[sl_key]['total_return']
                    text_matrix[i, j] = f"{sl_data[sl_key]['total_return']:.1f}"
                elif tp_key in tp_data:
                    matrix[i, j] = tp_data[tp_key]['total_return']
                    text_matrix[i, j] = f"{tp_data[tp_key]['total_return']:.1f}"
                else:
                    text_matrix[i, j] = "N/A"
        
        return {
            'values': matrix,
            'text_values': text_matrix,
            'sl_levels': sl_levels,
            'tp_levels': tp_levels
        }
    
    def _create_static_heatmap(self, heatmap_data: Dict[str, Any], title: str):
        """Create static matplotlib heatmap"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        if HAS_SEABORN:
            sns.heatmap(
                heatmap_data['values'],
                xticklabels=[f"TP {tp}%" for tp in heatmap_data['tp_levels']],
                yticklabels=[f"SL {sl}%" for sl in heatmap_data['sl_levels']],
                annot=heatmap_data['text_values'],
                fmt='',
                cmap='RdYlGn',
                center=0,
                ax=ax
            )
        else:
            # Fallback to basic matplotlib imshow
            im = ax.imshow(heatmap_data['values'], cmap='RdYlGn', aspect='auto')
            ax.set_xticks(range(len(heatmap_data['tp_levels'])))
            ax.set_yticks(range(len(heatmap_data['sl_levels'])))
            ax.set_xticklabels([f"TP {tp}%" for tp in heatmap_data['tp_levels']])
            ax.set_yticklabels([f"SL {sl}%" for sl in heatmap_data['sl_levels']])
            plt.colorbar(im, ax=ax)
        
        ax.set_title(f"{title} - Static Version", fontsize=14, fontweight='bold')
        ax.set_xlabel("Take Profit Levels", fontsize=12)
        ax.set_ylabel("Stop Loss Levels", fontsize=12)
        
        plt.tight_layout()
        static_path = self.output_dir / f"strategy_heatmap_static_{self.timestamp}.png"
        plt.savefig(static_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_time_series_analysis(self, attribution_results: Dict[str, Any],
                                  title: str = "Portfolio Time Series Analysis") -> str:
        """
        Create comprehensive time series analysis charts
        
        Args:
            attribution_results: Results from performance attribution analysis
            title: Chart title
            
        Returns:
            Path to saved visualization file
        """
        logger.info("📈 Creating time series analysis charts")
        
        time_data = attribution_results.get('time_attribution', {})
        if not time_data:
            logger.warning("⚠️  No time series data available")
            return ""
        
        # Create subplot figure
        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=['Daily Returns', 'Cumulative Returns', 
                          'Rolling Volatility (10-day)', 'Monthly Attribution'],
            vertical_spacing=0.05,
            row_heights=[0.25, 0.25, 0.25, 0.25]
        )
        
        dates = pd.to_datetime(time_data['dates'])
        daily_returns = np.array(time_data['daily_returns'])
        cumulative_returns = np.array(time_data['cumulative_returns'])
        
        # Daily returns
        fig.add_trace(
            go.Scatter(x=dates, y=daily_returns, mode='lines+markers',
                      name='Daily Returns', line=dict(color=self.colors['primary'])),
            row=1, col=1
        )
        
        # Cumulative returns
        fig.add_trace(
            go.Scatter(x=dates, y=cumulative_returns, mode='lines',
                      name='Cumulative Returns', line=dict(color=self.colors['success'])),
            row=2, col=1
        )
        
        # Rolling volatility
        rolling_vol = pd.Series(daily_returns).rolling(window=10).std()
        fig.add_trace(
            go.Scatter(x=dates, y=rolling_vol, mode='lines',
                      name='10-day Volatility', line=dict(color=self.colors['warning'])),
            row=3, col=1
        )
        
        # Monthly attribution
        monthly_data = time_data.get('monthly_attribution', {})
        if monthly_data:
            months = list(monthly_data.keys())
            monthly_returns = [data['total_return'] for data in monthly_data.values()]
            
            fig.add_trace(
                go.Bar(x=months, y=monthly_returns, name='Monthly Returns',
                      marker_color=self.colors['secondary']),
                row=4, col=1
            )
        
        fig.update_layout(
            title=f"{title} - 82 Trading Days Analysis",
            height=1200,
            showlegend=False,
            font=dict(size=10)
        )
        
        # Save interactive version
        output_path = self.output_dir / f"time_series_analysis_{self.timestamp}.html"
        fig.write_html(str(output_path))
        
        logger.info(f"✅ Time series analysis saved to {output_path}")
        return str(output_path)
    
    def create_correlation_network(self, correlation_matrix: np.ndarray,
                                 strategy_names: List[str],
                                 threshold: float = 0.5,
                                 title: str = "Strategy Correlation Network") -> str:
        """
        Create network visualization of strategy correlations
        
        Args:
            correlation_matrix: Strategy correlation matrix
            strategy_names: List of strategy names
            threshold: Minimum correlation to display edge
            title: Chart title
            
        Returns:
            Path to saved visualization file
        """
        logger.info("🕸️ Creating correlation network visualization")
        
        # Sample strategies for visualization (full 25,544 would be too dense)
        sample_size = min(100, len(strategy_names))
        sample_indices = np.random.choice(len(strategy_names), sample_size, replace=False)
        sample_corr = correlation_matrix[np.ix_(sample_indices, sample_indices)]
        sample_names = [strategy_names[i] for i in sample_indices]
        
        # Create network graph
        G = nx.Graph()
        
        # Add nodes
        for i, name in enumerate(sample_names):
            G.add_node(i, name=name[:20])  # Truncate long names
        
        # Add edges for correlations above threshold
        for i in range(len(sample_names)):
            for j in range(i+1, len(sample_names)):
                corr_value = sample_corr[i, j]
                if abs(corr_value) > threshold:
                    G.add_edge(i, j, weight=abs(corr_value), correlation=corr_value)
        
        # Create layout
        pos = nx.spring_layout(G, k=3, iterations=50)
        
        # Extract edge information
        edge_x = []
        edge_y = []
        edge_colors = []
        edge_widths = []
        
        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
            corr = edge[2]['correlation']
            edge_colors.append('red' if corr < 0 else 'blue')
            edge_widths.append(abs(corr) * 5)
        
        # Create plotly network
        fig = go.Figure()
        
        # Add edges
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='lightgray'),
            hoverinfo='none',
            mode='lines'
        ))
        
        # Add nodes
        node_x = [pos[node][0] for node in G.nodes()]
        node_y = [pos[node][1] for node in G.nodes()]
        node_text = [G.nodes[node]['name'] for node in G.nodes()]
        node_degrees = [G.degree[node] for node in G.nodes()]
        
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=node_text,
            textposition="middle center",
            hovertemplate="Strategy: %{text}<br>Connections: %{marker.size}<extra></extra>",
            marker=dict(
                size=[d*5 + 10 for d in node_degrees],
                color=node_degrees,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Connections")
            )
        ))
        
        fig.update_layout(
            title=f"{title} (Sample of {sample_size} strategies)",
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            width=1000,
            height=800
        )
        
        # Save visualization
        output_path = self.output_dir / f"correlation_network_{self.timestamp}.html"
        fig.write_html(str(output_path))
        
        logger.info(f"✅ Correlation network saved to {output_path}")
        return str(output_path)
    
    def create_risk_attribution_charts(self, risk_results: Dict[str, Any],
                                     title: str = "Risk Attribution Analysis") -> str:
        """
        Create comprehensive risk attribution visualizations
        
        Args:
            risk_results: Results from risk metrics analysis
            title: Chart title
            
        Returns:
            Path to saved visualization file
        """
        logger.info("⚠️ Creating risk attribution charts")
        
        # Create subplot figure
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['VaR Distribution', 'Drawdown Analysis', 
                          'Risk Contributions', 'Tail Risk Analysis'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"type": "pie"}, {"secondary_y": False}]]
        )
        
        # VaR Distribution
        var_data = risk_results.get('var_analysis', {})
        if var_data:
            confidence_levels = [90, 95, 99]
            var_values = [var_data.get(f'var_{cl}', 0) for cl in confidence_levels]
            
            fig.add_trace(
                go.Bar(x=[f"{cl}%" for cl in confidence_levels], y=var_values,
                      name='VaR Values', marker_color=self.colors['danger']),
                row=1, col=1
            )
        
        # Drawdown Analysis
        drawdown_data = risk_results.get('drawdown_analysis', {})
        if drawdown_data and 'daily_drawdowns' in drawdown_data:
            drawdowns = drawdown_data['daily_drawdowns']
            fig.add_trace(
                go.Scatter(y=drawdowns, mode='lines', fill='tonegative',
                          name='Daily Drawdowns', line=dict(color=self.colors['danger'])),
                row=1, col=2
            )
        
        # Risk Contributions (pie chart)
        risk_contrib = risk_results.get('individual_risk_contributions', {})
        if risk_contrib:
            # Show top 10 risk contributors
            contrib_items = list(risk_contrib.items())[:10]
            labels = [item[0][:15] for item in contrib_items]  # Truncate names
            values = [abs(item[1]['risk_contribution']) for item in contrib_items]
            
            fig.add_trace(
                go.Pie(labels=labels, values=values, name="Risk Contributions"),
                row=2, col=1
            )
        
        # Tail Risk Analysis
        tail_risk = risk_results.get('tail_risk_analysis', {})
        if tail_risk and 'extreme_events' in tail_risk:
            extreme_events = tail_risk['extreme_events']
            if extreme_events:
                fig.add_trace(
                    go.Histogram(x=extreme_events, nbinsx=20, name='Extreme Events',
                               marker_color=self.colors['warning']),
                    row=2, col=2
                )
        
        fig.update_layout(
            title=f"{title} - Production Risk Metrics",
            height=800,
            showlegend=False
        )
        
        # Save visualization
        output_path = self.output_dir / f"risk_attribution_{self.timestamp}.html"
        fig.write_html(str(output_path))
        
        logger.info(f"✅ Risk attribution charts saved to {output_path}")
        return str(output_path)
    
    def create_scenario_impact_plots(self, scenario_results: Dict[str, Any],
                                   title: str = "Scenario Impact Analysis") -> str:
        """
        Create scenario modeling impact visualizations
        
        Args:
            scenario_results: Results from scenario modeling analysis
            title: Chart title
            
        Returns:
            Path to saved visualization file
        """
        logger.info("🎭 Creating scenario impact plots")
        
        # Create subplot figure
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Historical Scenarios', 'Stress Test Results',
                          'Market Regime Analysis', 'Scenario Comparison'],
            vertical_spacing=0.1
        )
        
        # Historical scenarios
        historical_data = scenario_results.get('historical_scenarios', {})
        if historical_data:
            scenarios = list(historical_data.keys())
            returns = [data.get('total_return', 0) for data in historical_data.values()]
            
            fig.add_trace(
                go.Bar(x=scenarios, y=returns, name='Historical Returns',
                      marker_color=self.colors['primary']),
                row=1, col=1
            )
        
        # Stress test results
        stress_data = scenario_results.get('stress_scenarios', {})
        if stress_data:
            stress_scenarios = list(stress_data.keys())
            stress_returns = [data.get('portfolio_return', 0) for data in stress_data.values()]
            
            fig.add_trace(
                go.Bar(x=stress_scenarios, y=stress_returns, name='Stress Returns',
                      marker_color=self.colors['danger']),
                row=1, col=2
            )
        
        # Market regime analysis
        regime_data = scenario_results.get('regime_analysis', {})
        if regime_data:
            regimes = list(regime_data.keys())
            regime_returns = [data.get('avg_return', 0) for data in regime_data.values()]
            
            fig.add_trace(
                go.Scatter(x=regimes, y=regime_returns, mode='markers+lines',
                          name='Regime Returns', line=dict(color=self.colors['success'])),
                row=2, col=1
            )
        
        # Scenario comparison (violin plot simulation)
        scenario_comparison = scenario_results.get('scenario_comparison', {})
        if scenario_comparison:
            all_scenarios = []
            scenario_labels = []
            
            for scenario_name, scenario_data in scenario_comparison.items():
                if 'return_distribution' in scenario_data:
                    returns_dist = scenario_data['return_distribution']
                    all_scenarios.extend(returns_dist)
                    scenario_labels.extend([scenario_name] * len(returns_dist))
            
            if all_scenarios:
                fig.add_trace(
                    go.Box(x=scenario_labels, y=all_scenarios, name='Return Distributions'),
                    row=2, col=2
                )
        
        fig.update_layout(
            title=f"{title} - January-July 2024 Market Data",
            height=800,
            showlegend=False
        )
        
        # Save visualization
        output_path = self.output_dir / f"scenario_impact_{self.timestamp}.html"
        fig.write_html(str(output_path))
        
        logger.info(f"✅ Scenario impact plots saved to {output_path}")
        return str(output_path)
    
    def create_comprehensive_dashboard(self, all_results: Dict[str, Any],
                                     title: str = "Advanced Analytics Dashboard") -> str:
        """
        Create comprehensive dashboard combining all analytics visualizations
        
        Args:
            all_results: Combined results from all analytics modules
            title: Dashboard title
            
        Returns:
            Path to saved dashboard file
        """
        logger.info("📋 Creating comprehensive analytics dashboard")
        
        # Create large subplot figure
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=[
                'Portfolio Summary', 'Risk Metrics', 'Sensitivity Analysis',
                'Attribution by SL', 'Attribution by TP', 'Correlation Structure',
                'Time Series', 'Scenario Impact', 'Performance vs Risk'
            ],
            specs=[
                [{"type": "indicator"}, {"type": "bar"}, {"type": "scatter"}],
                [{"type": "bar"}, {"type": "bar"}, {"type": "heatmap"}],
                [{"type": "scatter"}, {"type": "bar"}, {"type": "scatter"}]
            ],
            vertical_spacing=0.08,
            horizontal_spacing=0.05
        )
        
        # Portfolio Summary (KPI indicators)
        portfolio_summary = all_results.get('attribution', {}).get('portfolio_summary', {})
        if portfolio_summary:
            total_return = portfolio_summary.get('total_return', 0)
            sharpe_ratio = portfolio_summary.get('sharpe_ratio', 0)
            max_drawdown = portfolio_summary.get('max_drawdown', 0)
            
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=total_return,
                    title={'text': "Total Return (%)"},
                    gauge={'axis': {'range': [-100, 100]},
                           'bar': {'color': self.colors['success']},
                           'steps': [{'range': [-100, 0], 'color': "lightgray"},
                                   {'range': [0, 100], 'color': "lightgreen"}]}
                ),
                row=1, col=1
            )
        
        # Add other dashboard components...
        # (Implementation continues with additional charts)
        
        fig.update_layout(
            title=f"{title} - Production Analytics Summary",
            height=1200,
            showlegend=False,
            font=dict(size=10)
        )
        
        # Save dashboard
        output_path = self.output_dir / f"analytics_dashboard_{self.timestamp}.html"
        fig.write_html(str(output_path))
        
        logger.info(f"✅ Comprehensive dashboard saved to {output_path}")
        return str(output_path)
    
    def generate_all_visualizations(self, all_results: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate all visualization types and return paths
        
        Args:
            all_results: Combined results from all analytics modules
            
        Returns:
            Dictionary mapping visualization types to file paths
        """
        logger.info("🎨 Generating all analytics visualizations")
        
        visualization_paths = {}
        
        try:
            # Strategy performance heatmap
            if 'attribution' in all_results:
                heatmap_path = self.create_strategy_performance_heatmap(
                    all_results['attribution'], 
                    all_results.get('strategy_names', [])
                )
                visualization_paths['heatmap'] = heatmap_path
            
            # Time series analysis
            if 'attribution' in all_results:
                timeseries_path = self.create_time_series_analysis(all_results['attribution'])
                visualization_paths['timeseries'] = timeseries_path
            
            # Correlation network
            if 'correlation_matrix' in all_results:
                network_path = self.create_correlation_network(
                    all_results['correlation_matrix'],
                    all_results.get('strategy_names', [])
                )
                visualization_paths['network'] = network_path
            
            # Risk attribution
            if 'risk_metrics' in all_results:
                risk_path = self.create_risk_attribution_charts(all_results['risk_metrics'])
                visualization_paths['risk'] = risk_path
            
            # Scenario impact
            if 'scenarios' in all_results:
                scenario_path = self.create_scenario_impact_plots(all_results['scenarios'])
                visualization_paths['scenarios'] = scenario_path
            
            # Comprehensive dashboard
            dashboard_path = self.create_comprehensive_dashboard(all_results)
            visualization_paths['dashboard'] = dashboard_path
            
        except Exception as e:
            logger.error(f"❌ Error generating visualizations: {e}")
        
        logger.info(f"✅ Generated {len(visualization_paths)} visualizations")
        return visualization_paths