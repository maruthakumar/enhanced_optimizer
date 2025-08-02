"""
Diversification Analysis Module

Analyzes portfolio diversification using correlation structure including:
- Correlation structure analysis of production strategy universe
- Risk contribution analysis using actual P&L correlations
- Strategy clustering based on performance patterns
- Diversification effectiveness across SENSEX strategy variations
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import warnings

logger = logging.getLogger(__name__)


@dataclass
class DiversificationConfig:
    """Configuration for diversification analysis"""
    correlation_threshold_high: float = 0.7
    correlation_threshold_low: float = 0.3
    clustering_methods: List[str] = None
    max_clusters: int = 10
    min_cluster_size: int = 5
    pca_variance_threshold: float = 0.95
    diversification_metrics: List[str] = None
    
    def __post_init__(self):
        if self.clustering_methods is None:
            self.clustering_methods = ['kmeans', 'hierarchical']
        
        if self.diversification_metrics is None:
            self.diversification_metrics = [
                'correlation_dispersion',
                'effective_number_of_assets',
                'diversification_ratio',
                'concentration_index'
            ]


class DiversificationAnalyzer:
    """
    Analyzes portfolio diversification characteristics and effectiveness.
    
    Provides analysis of:
    - Correlation structure across 25,544 strategies
    - Risk contribution patterns
    - Strategy clustering and groupings
    - Diversification effectiveness metrics
    """
    
    def __init__(self, config: Optional[DiversificationConfig] = None):
        """
        Initialize diversification analyzer
        
        Args:
            config: Diversification analysis configuration
        """
        self.config = config or DiversificationConfig()
        self.correlation_matrix = None
        self.strategy_clusters = {}
        self.diversification_cache = {}
        
    def perform_comprehensive_diversification_analysis(self, 
                                                     portfolio: List[int],
                                                     portfolio_weights: np.ndarray,
                                                     daily_returns: np.ndarray,
                                                     strategy_names: List[str],
                                                     correlation_matrix: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Perform comprehensive diversification analysis
        
        Args:
            portfolio: Portfolio strategy indices
            portfolio_weights: Portfolio weights
            daily_returns: Daily returns matrix (days x strategies)
            strategy_names: Strategy names
            correlation_matrix: Pre-calculated correlation matrix
            
        Returns:
            Dictionary with comprehensive diversification analysis
        """
        logger.info("🔍 Starting comprehensive diversification analysis")
        
        # Calculate or use provided correlation matrix
        if correlation_matrix is None:
            correlation_matrix = np.corrcoef(daily_returns.T)
        
        self.correlation_matrix = correlation_matrix
        
        diversification_results = {
            'correlation_structure': self.analyze_correlation_structure(
                correlation_matrix, strategy_names
            ),
            'strategy_clustering': self.perform_strategy_clustering(
                daily_returns, strategy_names, correlation_matrix
            ),
            'portfolio_diversification': self.analyze_portfolio_diversification(
                portfolio, portfolio_weights, daily_returns, correlation_matrix, strategy_names
            ),
            'risk_contribution': self.analyze_risk_contribution(
                portfolio, portfolio_weights, daily_returns, correlation_matrix, strategy_names
            ),
            'diversification_effectiveness': self.measure_diversification_effectiveness(
                portfolio, portfolio_weights, daily_returns, correlation_matrix
            ),
            'concentration_analysis': self.analyze_portfolio_concentration(
                portfolio, portfolio_weights, strategy_names
            )
        }
        
        # Add comprehensive summary
        diversification_results['diversification_summary'] = self._generate_diversification_summary(
            diversification_results, portfolio, strategy_names
        )
        
        logger.info("✅ Comprehensive diversification analysis completed")
        return diversification_results
    
    def analyze_correlation_structure(self, 
                                    correlation_matrix: np.ndarray,
                                    strategy_names: List[str]) -> Dict[str, Any]:
        """
        Analyze correlation structure of the strategy universe
        
        Args:
            correlation_matrix: Strategy correlation matrix
            strategy_names: Strategy names
            
        Returns:
            Correlation structure analysis
        """
        logger.info("📊 Analyzing correlation structure across strategy universe")
        
        # Extract upper triangle (excluding diagonal)
        n_strategies = correlation_matrix.shape[0]
        upper_triangle = np.triu(correlation_matrix, k=1)
        correlations = upper_triangle[upper_triangle != 0]
        
        # Basic correlation statistics
        correlation_stats = {
            'total_pairs': len(correlations),
            'average_correlation': np.mean(correlations),
            'median_correlation': np.median(correlations),
            'std_correlation': np.std(correlations),
            'min_correlation': np.min(correlations),
            'max_correlation': np.max(correlations),
            'correlation_percentiles': {
                f'{p}th': np.percentile(correlations, p)
                for p in [5, 10, 25, 50, 75, 90, 95, 99]
            }
        }
        
        # Correlation distribution analysis
        correlation_distribution = self._analyze_correlation_distribution(correlations)
        
        # High correlation pairs
        high_correlation_pairs = self._find_high_correlation_pairs(
            correlation_matrix, strategy_names, self.config.correlation_threshold_high
        )
        
        # Low correlation pairs (good for diversification)
        low_correlation_pairs = self._find_low_correlation_pairs(
            correlation_matrix, strategy_names, self.config.correlation_threshold_low
        )
        
        # Correlation network analysis
        network_analysis = self._analyze_correlation_network(correlation_matrix, strategy_names)
        
        return {
            'correlation_statistics': correlation_stats,
            'correlation_distribution': correlation_distribution,
            'high_correlation_pairs': high_correlation_pairs,
            'low_correlation_pairs': low_correlation_pairs,
            'network_analysis': network_analysis,
            'correlation_heatmap_data': self._prepare_correlation_heatmap_data(correlation_matrix)
        }
    
    def perform_strategy_clustering(self, 
                                  daily_returns: np.ndarray,
                                  strategy_names: List[str],
                                  correlation_matrix: np.ndarray) -> Dict[str, Any]:
        """
        Perform strategy clustering based on performance patterns
        
        Args:
            daily_returns: Daily returns matrix
            strategy_names: Strategy names
            correlation_matrix: Correlation matrix
            
        Returns:
            Strategy clustering results
        """
        logger.info("🎯 Performing strategy clustering based on performance patterns")
        
        clustering_results = {}
        
        # Prepare data for clustering
        # Use correlation distance for clustering
        correlation_distance = 1 - np.abs(correlation_matrix)
        
        for method in self.config.clustering_methods:
            logger.info(f"🔍 Applying {method} clustering")
            
            if method == 'kmeans':
                clustering_result = self._perform_kmeans_clustering(
                    daily_returns, strategy_names, correlation_distance
                )
            elif method == 'hierarchical':
                clustering_result = self._perform_hierarchical_clustering(
                    correlation_distance, strategy_names
                )
            else:
                logger.warning(f"Unknown clustering method: {method}")
                continue
            
            clustering_results[method] = clustering_result
        
        # Find optimal clustering
        optimal_clustering = self._find_optimal_clustering(clustering_results, correlation_matrix)
        
        # Analyze cluster characteristics
        cluster_analysis = self._analyze_cluster_characteristics(
            optimal_clustering, daily_returns, strategy_names, correlation_matrix
        )
        
        return {
            'clustering_results': clustering_results,
            'optimal_clustering': optimal_clustering,
            'cluster_analysis': cluster_analysis,
            'cluster_diversification_potential': self._assess_cluster_diversification_potential(
                optimal_clustering, correlation_matrix
            )
        }
    
    def analyze_portfolio_diversification(self, 
                                        portfolio: List[int],
                                        portfolio_weights: np.ndarray,
                                        daily_returns: np.ndarray,
                                        correlation_matrix: np.ndarray,
                                        strategy_names: List[str]) -> Dict[str, Any]:
        """
        Analyze current portfolio diversification characteristics
        
        Args:
            portfolio: Portfolio strategy indices
            portfolio_weights: Portfolio weights
            daily_returns: Daily returns matrix
            correlation_matrix: Correlation matrix
            strategy_names: Strategy names
            
        Returns:
            Portfolio diversification analysis
        """
        logger.info("📈 Analyzing portfolio diversification characteristics")
        
        # Extract portfolio correlation matrix
        portfolio_corr_matrix = correlation_matrix[np.ix_(portfolio, portfolio)]
        
        # Calculate diversification metrics
        diversification_metrics = {}
        
        for metric in self.config.diversification_metrics:
            if metric == 'correlation_dispersion':
                diversification_metrics[metric] = self._calculate_correlation_dispersion(
                    portfolio_corr_matrix, portfolio_weights
                )
            elif metric == 'effective_number_of_assets':
                diversification_metrics[metric] = self._calculate_effective_number_of_assets(
                    portfolio_weights, portfolio_corr_matrix
                )
            elif metric == 'diversification_ratio':
                diversification_metrics[metric] = self._calculate_diversification_ratio(
                    portfolio, daily_returns, portfolio_weights, correlation_matrix
                )
            elif metric == 'concentration_index':
                diversification_metrics[metric] = self._calculate_concentration_index(
                    portfolio_weights
                )
        
        # Portfolio correlation analysis
        portfolio_correlation_analysis = {
            'average_correlation': np.mean(portfolio_corr_matrix[np.triu_indices_from(portfolio_corr_matrix, k=1)]),
            'max_correlation': np.max(portfolio_corr_matrix[np.triu_indices_from(portfolio_corr_matrix, k=1)]),
            'min_correlation': np.min(portfolio_corr_matrix[np.triu_indices_from(portfolio_corr_matrix, k=1)]),
            'correlation_std': np.std(portfolio_corr_matrix[np.triu_indices_from(portfolio_corr_matrix, k=1)])
        }
        
        # Identify diversification gaps
        diversification_gaps = self._identify_diversification_gaps(
            portfolio, portfolio_weights, correlation_matrix, strategy_names
        )
        
        # Diversification improvement suggestions
        improvement_suggestions = self._generate_diversification_suggestions(
            portfolio, correlation_matrix, strategy_names, diversification_gaps
        )
        
        return {
            'diversification_metrics': diversification_metrics,
            'portfolio_correlation_analysis': portfolio_correlation_analysis,
            'diversification_gaps': diversification_gaps,
            'improvement_suggestions': improvement_suggestions,
            'portfolio_structure_analysis': self._analyze_portfolio_structure(
                portfolio, portfolio_weights, strategy_names
            )
        }
    
    def analyze_risk_contribution(self, 
                                portfolio: List[int],
                                portfolio_weights: np.ndarray,
                                daily_returns: np.ndarray,
                                correlation_matrix: np.ndarray,
                                strategy_names: List[str]) -> Dict[str, Any]:
        """
        Analyze risk contribution using actual P&L correlations
        
        Args:
            portfolio: Portfolio strategy indices
            portfolio_weights: Portfolio weights
            daily_returns: Daily returns matrix
            correlation_matrix: Correlation matrix
            strategy_names: Strategy names
            
        Returns:
            Risk contribution analysis
        """
        logger.info("⚖️ Analyzing risk contribution using P&L correlations")
        
        # Extract portfolio data
        portfolio_returns = daily_returns[:, portfolio]
        portfolio_corr_matrix = correlation_matrix[np.ix_(portfolio, portfolio)]
        
        # Calculate covariance matrix
        portfolio_volatilities = np.std(portfolio_returns, axis=0)
        portfolio_cov_matrix = portfolio_corr_matrix * np.outer(portfolio_volatilities, portfolio_volatilities)
        
        # Calculate portfolio variance
        portfolio_variance = np.dot(portfolio_weights, np.dot(portfolio_cov_matrix, portfolio_weights))
        
        # Marginal risk contributions
        marginal_contributions = np.dot(portfolio_cov_matrix, portfolio_weights)
        
        # Component risk contributions
        component_contributions = portfolio_weights * marginal_contributions
        relative_contributions = component_contributions / portfolio_variance
        
        # Risk contribution analysis
        risk_contributions = {}
        for i, strategy_idx in enumerate(portfolio):
            strategy_name = strategy_names[strategy_idx]
            risk_contributions[strategy_name] = {
                'weight': portfolio_weights[i],
                'marginal_contribution': marginal_contributions[i],
                'component_contribution': component_contributions[i],
                'relative_contribution': relative_contributions[i],
                'risk_contribution_pct': relative_contributions[i] * 100
            }
        
        # Risk concentration analysis
        risk_concentration = self._analyze_risk_concentration(relative_contributions, portfolio_weights)
        
        # Correlation-based risk analysis
        correlation_risk_analysis = self._analyze_correlation_based_risk(
            portfolio_corr_matrix, portfolio_weights, strategy_names, portfolio
        )
        
        return {
            'individual_risk_contributions': risk_contributions,
            'risk_concentration': risk_concentration,
            'correlation_risk_analysis': correlation_risk_analysis,
            'portfolio_risk_metrics': {
                'portfolio_variance': portfolio_variance,
                'portfolio_volatility': np.sqrt(portfolio_variance),
                'risk_weighted_average_correlation': self._calculate_risk_weighted_correlation(
                    portfolio_corr_matrix, portfolio_weights
                )
            }
        }
    
    def measure_diversification_effectiveness(self, 
                                            portfolio: List[int],
                                            portfolio_weights: np.ndarray,
                                            daily_returns: np.ndarray,
                                            correlation_matrix: np.ndarray) -> Dict[str, Any]:
        """
        Measure diversification effectiveness across SENSEX strategy variations
        
        Args:
            portfolio: Portfolio strategy indices
            portfolio_weights: Portfolio weights
            daily_returns: Daily returns matrix
            correlation_matrix: Correlation matrix
            
        Returns:
            Diversification effectiveness metrics
        """
        logger.info("📏 Measuring diversification effectiveness")
        
        # Portfolio returns and volatility
        portfolio_returns = daily_returns[:, portfolio]
        portfolio_daily_returns = np.sum(portfolio_returns * portfolio_weights, axis=1)
        portfolio_volatility = np.std(portfolio_daily_returns)
        
        # Individual strategy volatilities
        individual_volatilities = np.std(portfolio_returns, axis=0)
        
        # Weighted average volatility (no diversification case)
        weighted_avg_volatility = np.sum(portfolio_weights * individual_volatilities)
        
        # Diversification effectiveness metrics
        diversification_ratio = weighted_avg_volatility / portfolio_volatility
        diversification_benefit = (weighted_avg_volatility - portfolio_volatility) / weighted_avg_volatility
        
        # Effective number of independent assets
        portfolio_variance = portfolio_volatility ** 2
        avg_individual_variance = np.sum(portfolio_weights * (individual_volatilities ** 2))
        effective_n = avg_individual_variance / portfolio_variance
        
        # Correlation impact on diversification
        avg_correlation = np.mean(correlation_matrix[np.ix_(portfolio, portfolio)][np.triu_indices_from(correlation_matrix[np.ix_(portfolio, portfolio)], k=1)])
        
        # Theoretical minimum volatility (zero correlation case)
        zero_corr_volatility = np.sqrt(np.sum((portfolio_weights * individual_volatilities) ** 2))
        
        # Theoretical maximum volatility (perfect correlation case)
        perfect_corr_volatility = weighted_avg_volatility
        
        # Diversification efficiency
        diversification_efficiency = (perfect_corr_volatility - portfolio_volatility) / (perfect_corr_volatility - zero_corr_volatility)
        
        return {
            'diversification_ratio': diversification_ratio,
            'diversification_benefit': diversification_benefit,
            'effective_number_of_assets': effective_n,
            'diversification_efficiency': diversification_efficiency,
            'volatility_comparison': {
                'portfolio_volatility': portfolio_volatility,
                'weighted_average_volatility': weighted_avg_volatility,
                'zero_correlation_volatility': zero_corr_volatility,
                'perfect_correlation_volatility': perfect_corr_volatility
            },
            'correlation_impact': {
                'average_correlation': avg_correlation,
                'correlation_penalty': portfolio_volatility - zero_corr_volatility,
                'correlation_penalty_pct': (portfolio_volatility - zero_corr_volatility) / zero_corr_volatility * 100
            }
        }
    
    def analyze_portfolio_concentration(self, 
                                      portfolio: List[int],
                                      portfolio_weights: np.ndarray,
                                      strategy_names: List[str]) -> Dict[str, Any]:
        """
        Analyze portfolio concentration patterns
        
        Args:
            portfolio: Portfolio strategy indices
            portfolio_weights: Portfolio weights
            strategy_names: Strategy names
            
        Returns:
            Portfolio concentration analysis
        """
        logger.info("🎯 Analyzing portfolio concentration patterns")
        
        # Concentration metrics
        concentration_metrics = {
            'herfindahl_index': np.sum(portfolio_weights ** 2),
            'effective_number_of_holdings': 1 / np.sum(portfolio_weights ** 2),
            'gini_coefficient': self._calculate_gini_coefficient(portfolio_weights),
            'max_weight': np.max(portfolio_weights),
            'min_weight': np.min(portfolio_weights),
            'weight_std': np.std(portfolio_weights),
            'weight_range': np.max(portfolio_weights) - np.min(portfolio_weights)
        }
        
        # Top holdings analysis
        sorted_indices = np.argsort(portfolio_weights)[::-1]
        top_holdings = []
        cumulative_weight = 0
        
        for i in range(min(10, len(portfolio))):  # Top 10 holdings
            idx = sorted_indices[i]
            strategy_idx = portfolio[idx]
            weight = portfolio_weights[idx]
            cumulative_weight += weight
            
            top_holdings.append({
                'rank': i + 1,
                'strategy_name': strategy_names[strategy_idx],
                'weight': weight,
                'weight_pct': weight * 100,
                'cumulative_weight_pct': cumulative_weight * 100
            })
        
        # Concentration risk analysis
        concentration_risk = self._analyze_concentration_risk(
            portfolio_weights, [strategy_names[portfolio[i]] for i in range(len(portfolio))]
        )
        
        return {
            'concentration_metrics': concentration_metrics,
            'top_holdings': top_holdings,
            'concentration_risk': concentration_risk,
            'weight_distribution': self._analyze_weight_distribution(portfolio_weights)
        }
    
    def _analyze_correlation_distribution(self, correlations: np.ndarray) -> Dict[str, Any]:
        """Analyze the distribution of correlation values"""
        # Define correlation buckets
        buckets = {
            'very_low': (-1.0, -0.5),
            'low': (-0.5, -0.2),
            'weak_negative': (-0.2, 0.0),
            'weak_positive': (0.0, 0.2),
            'moderate': (0.2, 0.5),
            'high': (0.5, 0.8),
            'very_high': (0.8, 1.0)
        }
        
        distribution = {}
        for bucket_name, (low, high) in buckets.items():
            count = np.sum((correlations >= low) & (correlations < high))
            percentage = count / len(correlations) * 100
            distribution[bucket_name] = {
                'count': count,
                'percentage': percentage,
                'range': f'{low:.1f} to {high:.1f}'
            }
        
        return distribution
    
    def _find_high_correlation_pairs(self, correlation_matrix: np.ndarray,
                                   strategy_names: List[str],
                                   threshold: float) -> List[Dict[str, Any]]:
        """Find pairs with high correlation"""
        high_corr_pairs = []
        n = correlation_matrix.shape[0]
        
        for i in range(n):
            for j in range(i + 1, n):
                corr = correlation_matrix[i, j]
                if abs(corr) > threshold:
                    high_corr_pairs.append({
                        'strategy1': strategy_names[i],
                        'strategy2': strategy_names[j],
                        'correlation': corr,
                        'abs_correlation': abs(corr)
                    })
        
        # Sort by absolute correlation
        high_corr_pairs.sort(key=lambda x: x['abs_correlation'], reverse=True)
        return high_corr_pairs[:100]  # Return top 100
    
    def _find_low_correlation_pairs(self, correlation_matrix: np.ndarray,
                                  strategy_names: List[str],
                                  threshold: float) -> List[Dict[str, Any]]:
        """Find pairs with low correlation (good for diversification)"""
        low_corr_pairs = []
        n = correlation_matrix.shape[0]
        
        for i in range(n):
            for j in range(i + 1, n):
                corr = correlation_matrix[i, j]
                if abs(corr) < threshold:
                    low_corr_pairs.append({
                        'strategy1': strategy_names[i],
                        'strategy2': strategy_names[j],
                        'correlation': corr,
                        'abs_correlation': abs(corr)
                    })
        
        # Sort by absolute correlation (lowest first)
        low_corr_pairs.sort(key=lambda x: x['abs_correlation'])
        return low_corr_pairs[:100]  # Return top 100
    
    def _analyze_correlation_network(self, correlation_matrix: np.ndarray,
                                   strategy_names: List[str]) -> Dict[str, Any]:
        """Analyze correlation network structure"""
        # Calculate network density
        high_corr_threshold = self.config.correlation_threshold_high
        high_corr_matrix = np.abs(correlation_matrix) > high_corr_threshold
        np.fill_diagonal(high_corr_matrix, False)  # Remove self-connections
        
        network_density = np.sum(high_corr_matrix) / (correlation_matrix.shape[0] * (correlation_matrix.shape[0] - 1))
        
        # Find highly connected strategies (hubs)
        connection_counts = np.sum(high_corr_matrix, axis=1)
        hub_threshold = np.percentile(connection_counts, 90)  # Top 10%
        
        hubs = []
        for i, count in enumerate(connection_counts):
            if count >= hub_threshold:
                hubs.append({
                    'strategy_name': strategy_names[i],
                    'connection_count': count,
                    'connection_percentage': count / len(strategy_names) * 100
                })
        
        # Sort hubs by connection count
        hubs.sort(key=lambda x: x['connection_count'], reverse=True)
        
        return {
            'network_density': network_density,
            'high_correlation_threshold': high_corr_threshold,
            'total_high_correlations': np.sum(high_corr_matrix),
            'hubs': hubs[:20],  # Top 20 hubs
            'average_connections_per_strategy': np.mean(connection_counts),
            'max_connections': np.max(connection_counts),
            'min_connections': np.min(connection_counts)
        }
    
    def _prepare_correlation_heatmap_data(self, correlation_matrix: np.ndarray) -> Dict[str, Any]:
        """Prepare data for correlation heatmap visualization"""
        # For large matrices, sample a subset for visualization
        max_size = 100  # Maximum size for heatmap
        
        if correlation_matrix.shape[0] > max_size:
            # Sample random strategies
            indices = np.random.choice(correlation_matrix.shape[0], max_size, replace=False)
            sampled_matrix = correlation_matrix[np.ix_(indices, indices)]
        else:
            sampled_matrix = correlation_matrix
            indices = np.arange(correlation_matrix.shape[0])
        
        return {
            'correlation_matrix': sampled_matrix.tolist(),
            'strategy_indices': indices.tolist(),
            'matrix_size': sampled_matrix.shape[0],
            'is_sampled': correlation_matrix.shape[0] > max_size
        }
    
    def _perform_kmeans_clustering(self, daily_returns: np.ndarray,
                                 strategy_names: List[str],
                                 correlation_distance: np.ndarray) -> Dict[str, Any]:
        """Perform K-means clustering on strategies"""
        # Determine optimal number of clusters
        silhouette_scores = []
        k_range = range(2, min(self.config.max_clusters + 1, len(strategy_names) // self.config.min_cluster_size))
        
        best_k = 2
        best_score = -1
        
        for k in k_range:
            if k >= len(strategy_names):
                break
                
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(daily_returns.T)
            
            try:
                score = silhouette_score(daily_returns.T, cluster_labels)
                silhouette_scores.append(score)
                
                if score > best_score:
                    best_score = score
                    best_k = k
            except:
                silhouette_scores.append(-1)
        
        # Perform final clustering with best k
        kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(daily_returns.T)
        
        # Organize results
        clusters = {}
        for i, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append({
                'strategy_index': i,
                'strategy_name': strategy_names[i]
            })
        
        return {
            'method': 'kmeans',
            'n_clusters': best_k,
            'cluster_labels': cluster_labels.tolist(),
            'clusters': clusters,
            'silhouette_score': best_score,
            'silhouette_scores_by_k': dict(zip(k_range, silhouette_scores))
        }
    
    def _perform_hierarchical_clustering(self, correlation_distance: np.ndarray,
                                       strategy_names: List[str]) -> Dict[str, Any]:
        """Perform hierarchical clustering on strategies"""
        # Use correlation distance for hierarchical clustering
        linkage_method = 'ward'
        
        # Determine optimal number of clusters
        silhouette_scores = []
        k_range = range(2, min(self.config.max_clusters + 1, len(strategy_names) // self.config.min_cluster_size))
        
        best_k = 2
        best_score = -1
        
        for k in k_range:
            if k >= len(strategy_names):
                break
                
            hierarchical = AgglomerativeClustering(n_clusters=k, linkage=linkage_method)
            cluster_labels = hierarchical.fit_predict(correlation_distance)
            
            try:
                score = silhouette_score(correlation_distance, cluster_labels, metric='precomputed')
                silhouette_scores.append(score)
                
                if score > best_score:
                    best_score = score
                    best_k = k
            except:
                silhouette_scores.append(-1)
        
        # Perform final clustering with best k
        hierarchical = AgglomerativeClustering(n_clusters=best_k, linkage=linkage_method)
        cluster_labels = hierarchical.fit_predict(correlation_distance)
        
        # Organize results
        clusters = {}
        for i, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append({
                'strategy_index': i,
                'strategy_name': strategy_names[i]
            })
        
        return {
            'method': 'hierarchical',
            'linkage_method': linkage_method,
            'n_clusters': best_k,
            'cluster_labels': cluster_labels.tolist(),
            'clusters': clusters,
            'silhouette_score': best_score,
            'silhouette_scores_by_k': dict(zip(k_range, silhouette_scores))
        }
    
    def _find_optimal_clustering(self, clustering_results: Dict[str, Dict[str, Any]],
                               correlation_matrix: np.ndarray) -> Dict[str, Any]:
        """Find the optimal clustering method and parameters"""
        best_method = None
        best_score = -1
        
        for method, result in clustering_results.items():
            score = result.get('silhouette_score', -1)
            if score > best_score:
                best_score = score
                best_method = method
        
        return clustering_results[best_method] if best_method else list(clustering_results.values())[0]
    
    def _analyze_cluster_characteristics(self, optimal_clustering: Dict[str, Any],
                                       daily_returns: np.ndarray,
                                       strategy_names: List[str],
                                       correlation_matrix: np.ndarray) -> Dict[str, Any]:
        """Analyze characteristics of identified clusters"""
        cluster_analysis = {}
        
        for cluster_id, strategies in optimal_clustering['clusters'].items():
            strategy_indices = [s['strategy_index'] for s in strategies]
            
            if len(strategy_indices) < 2:
                continue
            
            # Cluster returns analysis
            cluster_returns = daily_returns[:, strategy_indices]
            cluster_correlations = correlation_matrix[np.ix_(strategy_indices, strategy_indices)]
            
            # Calculate cluster characteristics
            cluster_analysis[f'cluster_{cluster_id}'] = {
                'size': len(strategy_indices),
                'strategy_names': [s['strategy_name'] for s in strategies],
                'performance_metrics': {
                    'avg_return': np.mean(cluster_returns),
                    'avg_volatility': np.mean(np.std(cluster_returns, axis=0)),
                    'return_correlation': np.mean(cluster_correlations[np.triu_indices_from(cluster_correlations, k=1)])
                },
                'intra_cluster_correlation': {
                    'mean': np.mean(cluster_correlations[np.triu_indices_from(cluster_correlations, k=1)]),
                    'std': np.std(cluster_correlations[np.triu_indices_from(cluster_correlations, k=1)]),
                    'min': np.min(cluster_correlations[np.triu_indices_from(cluster_correlations, k=1)]),
                    'max': np.max(cluster_correlations[np.triu_indices_from(cluster_correlations, k=1)])
                }
            }
        
        return cluster_analysis
    
    def _assess_cluster_diversification_potential(self, optimal_clustering: Dict[str, Any],
                                                correlation_matrix: np.ndarray) -> Dict[str, Any]:
        """Assess diversification potential across clusters"""
        cluster_indices = {}
        
        # Group strategy indices by cluster
        for cluster_id, strategies in optimal_clustering['clusters'].items():
            cluster_indices[cluster_id] = [s['strategy_index'] for s in strategies]
        
        # Calculate inter-cluster correlations
        inter_cluster_correlations = {}
        cluster_ids = list(cluster_indices.keys())
        
        for i, cluster1 in enumerate(cluster_ids):
            for j, cluster2 in enumerate(cluster_ids):
                if i <= j:
                    continue
                
                indices1 = cluster_indices[cluster1]
                indices2 = cluster_indices[cluster2]
                
                # Calculate average correlation between clusters
                cross_correlations = []
                for idx1 in indices1:
                    for idx2 in indices2:
                        cross_correlations.append(correlation_matrix[idx1, idx2])
                
                pair_key = f'cluster_{cluster1}_vs_{cluster2}'
                inter_cluster_correlations[pair_key] = {
                    'avg_correlation': np.mean(cross_correlations),
                    'std_correlation': np.std(cross_correlations),
                    'min_correlation': np.min(cross_correlations),
                    'max_correlation': np.max(cross_correlations)
                }
        
        # Overall diversification assessment
        all_inter_correlations = [corr['avg_correlation'] for corr in inter_cluster_correlations.values()]
        
        diversification_assessment = {
            'avg_inter_cluster_correlation': np.mean(all_inter_correlations) if all_inter_correlations else 0,
            'min_inter_cluster_correlation': np.min(all_inter_correlations) if all_inter_correlations else 0,
            'max_inter_cluster_correlation': np.max(all_inter_correlations) if all_inter_correlations else 0,
            'diversification_score': 1 - np.mean(all_inter_correlations) if all_inter_correlations else 1
        }
        
        return {
            'inter_cluster_correlations': inter_cluster_correlations,
            'diversification_assessment': diversification_assessment,
            'recommended_cluster_selection': self._recommend_cluster_selection(
                cluster_indices, inter_cluster_correlations
            )
        }
    
    def _calculate_correlation_dispersion(self, correlation_matrix: np.ndarray,
                                        weights: np.ndarray) -> float:
        """Calculate correlation dispersion metric"""
        weighted_correlations = []
        n = len(weights)
        
        for i in range(n):
            for j in range(i + 1, n):
                corr = correlation_matrix[i, j]
                weight = weights[i] * weights[j]
                weighted_correlations.append(corr * weight)
        
        return np.std(weighted_correlations) if weighted_correlations else 0
    
    def _calculate_effective_number_of_assets(self, weights: np.ndarray,
                                            correlation_matrix: np.ndarray) -> float:
        """Calculate effective number of independent assets"""
        # This is an approximation based on the portfolio's correlation structure
        avg_correlation = np.mean(correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)])
        herfindahl_index = np.sum(weights ** 2)
        
        # Adjust for correlation
        effective_n = (1 / herfindahl_index) * (1 - avg_correlation) / (1 - avg_correlation / len(weights))
        return max(1, effective_n)
    
    def _calculate_diversification_ratio(self, portfolio: List[int],
                                       daily_returns: np.ndarray,
                                       weights: np.ndarray,
                                       correlation_matrix: np.ndarray) -> float:
        """Calculate diversification ratio"""
        portfolio_returns = daily_returns[:, portfolio]
        individual_volatilities = np.std(portfolio_returns, axis=0)
        
        # Weighted average volatility
        weighted_avg_volatility = np.sum(weights * individual_volatilities)
        
        # Portfolio volatility
        portfolio_daily_returns = np.sum(portfolio_returns * weights, axis=1)
        portfolio_volatility = np.std(portfolio_daily_returns)
        
        return weighted_avg_volatility / portfolio_volatility if portfolio_volatility > 0 else 1
    
    def _calculate_concentration_index(self, weights: np.ndarray) -> float:
        """Calculate Herfindahl concentration index"""
        return np.sum(weights ** 2)
    
    def _identify_diversification_gaps(self, portfolio: List[int],
                                     weights: np.ndarray,
                                     correlation_matrix: np.ndarray,
                                     strategy_names: List[str]) -> Dict[str, Any]:
        """Identify gaps in portfolio diversification"""
        portfolio_corr_matrix = correlation_matrix[np.ix_(portfolio, portfolio)]
        
        # Find highly correlated positions
        high_corr_pairs = []
        threshold = self.config.correlation_threshold_high
        
        for i in range(len(portfolio)):
            for j in range(i + 1, len(portfolio)):
                corr = portfolio_corr_matrix[i, j]
                if abs(corr) > threshold:
                    combined_weight = weights[i] + weights[j]
                    high_corr_pairs.append({
                        'strategy1': strategy_names[portfolio[i]],
                        'strategy2': strategy_names[portfolio[j]],
                        'correlation': corr,
                        'combined_weight': combined_weight,
                        'diversification_impact': combined_weight * abs(corr)
                    })
        
        # Sort by diversification impact
        high_corr_pairs.sort(key=lambda x: x['diversification_impact'], reverse=True)
        
        # Identify concentration issues
        concentration_issues = []
        max_weight_threshold = 0.1  # 10%
        
        for i, weight in enumerate(weights):
            if weight > max_weight_threshold:
                concentration_issues.append({
                    'strategy': strategy_names[portfolio[i]],
                    'weight': weight,
                    'weight_pct': weight * 100,
                    'excess_weight': weight - max_weight_threshold
                })
        
        return {
            'high_correlation_pairs': high_corr_pairs[:10],  # Top 10 issues
            'concentration_issues': concentration_issues,
            'overall_gap_score': len(high_corr_pairs) + len(concentration_issues) * 2
        }
    
    def _generate_diversification_suggestions(self, portfolio: List[int],
                                            correlation_matrix: np.ndarray,
                                            strategy_names: List[str],
                                            diversification_gaps: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate suggestions for improving diversification"""
        suggestions = []
        
        # Suggestions based on high correlation pairs
        for pair in diversification_gaps['high_correlation_pairs'][:5]:
            suggestions.append({
                'type': 'reduce_correlation',
                'priority': 'high',
                'description': f"Consider reducing exposure to highly correlated strategies: {pair['strategy1']} and {pair['strategy2']} (correlation: {pair['correlation']:.3f})",
                'strategies_involved': [pair['strategy1'], pair['strategy2']],
                'correlation': pair['correlation']
            })
        
        # Suggestions based on concentration
        for issue in diversification_gaps['concentration_issues'][:3]:
            suggestions.append({
                'type': 'reduce_concentration',
                'priority': 'medium',
                'description': f"Consider reducing weight in {issue['strategy']} (current: {issue['weight_pct']:.1f}%)",
                'strategy_involved': issue['strategy'],
                'current_weight': issue['weight']
            })
        
        # Find uncorrelated strategies not in portfolio
        available_strategies = set(range(len(strategy_names))) - set(portfolio)
        if available_strategies:
            # Calculate average correlation with portfolio for each available strategy
            uncorrelated_candidates = []
            
            for candidate_idx in list(available_strategies)[:100]:  # Limit to first 100 for performance
                avg_corr_with_portfolio = np.mean([abs(correlation_matrix[candidate_idx, p_idx]) for p_idx in portfolio])
                
                if avg_corr_with_portfolio < self.config.correlation_threshold_low:
                    uncorrelated_candidates.append({
                        'strategy_index': candidate_idx,
                        'strategy_name': strategy_names[candidate_idx],
                        'avg_correlation_with_portfolio': avg_corr_with_portfolio
                    })
            
            # Sort by lowest correlation
            uncorrelated_candidates.sort(key=lambda x: x['avg_correlation_with_portfolio'])
            
            for candidate in uncorrelated_candidates[:3]:
                suggestions.append({
                    'type': 'add_uncorrelated',
                    'priority': 'low',
                    'description': f"Consider adding {candidate['strategy_name']} for better diversification (avg correlation with portfolio: {candidate['avg_correlation_with_portfolio']:.3f})",
                    'strategy_suggested': candidate['strategy_name'],
                    'avg_correlation': candidate['avg_correlation_with_portfolio']
                })
        
        return suggestions
    
    def _analyze_portfolio_structure(self, portfolio: List[int],
                                   weights: np.ndarray,
                                   strategy_names: List[str]) -> Dict[str, Any]:
        """Analyze overall portfolio structure"""
        # Strategy type analysis (if patterns exist in names)
        strategy_types = {}
        
        for i, strategy_idx in enumerate(portfolio):
            strategy_name = strategy_names[strategy_idx]
            weight = weights[i]
            
            # Simple classification based on name patterns
            if 'SL' in strategy_name:
                sl_match = strategy_name.split('SL')[1][:2]
                strategy_type = f'SL_{sl_match}'
            elif 'TP' in strategy_name:
                tp_match = strategy_name.split('TP')[1][:2]
                strategy_type = f'TP_{tp_match}'
            else:
                strategy_type = 'other'
            
            if strategy_type not in strategy_types:
                strategy_types[strategy_type] = {'count': 0, 'total_weight': 0, 'strategies': []}
            
            strategy_types[strategy_type]['count'] += 1
            strategy_types[strategy_type]['total_weight'] += weight
            strategy_types[strategy_type]['strategies'].append(strategy_name)
        
        return {
            'portfolio_size': len(portfolio),
            'strategy_type_distribution': strategy_types,
            'weight_statistics': {
                'mean_weight': np.mean(weights),
                'weight_std': np.std(weights),
                'min_weight': np.min(weights),
                'max_weight': np.max(weights),
                'equal_weight_deviation': np.std(weights - (1/len(weights)))
            }
        }
    
    def _analyze_risk_concentration(self, relative_contributions: np.ndarray,
                                  weights: np.ndarray) -> Dict[str, Any]:
        """Analyze risk concentration vs weight concentration"""
        # Risk Herfindahl index
        risk_herfindahl = np.sum(relative_contributions ** 2)
        
        # Weight Herfindahl index
        weight_herfindahl = np.sum(weights ** 2)
        
        # Risk concentration ratio
        risk_concentration_ratio = risk_herfindahl / weight_herfindahl
        
        return {
            'risk_herfindahl_index': risk_herfindahl,
            'weight_herfindahl_index': weight_herfindahl,
            'risk_concentration_ratio': risk_concentration_ratio,
            'effective_risk_positions': 1 / risk_herfindahl,
            'effective_weight_positions': 1 / weight_herfindahl,
            'risk_vs_weight_concentration': 'higher_risk_concentration' if risk_concentration_ratio > 1 else 'higher_weight_concentration'
        }
    
    def _analyze_correlation_based_risk(self, correlation_matrix: np.ndarray,
                                      weights: np.ndarray,
                                      strategy_names: List[str],
                                      portfolio: List[int]) -> Dict[str, Any]:
        """Analyze risk arising from correlations"""
        # Risk-weighted correlation
        risk_weighted_corr = 0
        total_weight = 0
        
        for i in range(len(portfolio)):
            for j in range(i + 1, len(portfolio)):
                weight_product = weights[i] * weights[j]
                correlation = correlation_matrix[i, j]
                risk_weighted_corr += weight_product * abs(correlation)
                total_weight += weight_product
        
        avg_risk_weighted_corr = risk_weighted_corr / total_weight if total_weight > 0 else 0
        
        # Identify highest risk pairs
        risk_pairs = []
        for i in range(len(portfolio)):
            for j in range(i + 1, len(portfolio)):
                pair_risk = weights[i] * weights[j] * abs(correlation_matrix[i, j])
                risk_pairs.append({
                    'strategy1': strategy_names[portfolio[i]],
                    'strategy2': strategy_names[portfolio[j]],
                    'correlation': correlation_matrix[i, j],
                    'combined_weight': weights[i] + weights[j],
                    'pair_risk_contribution': pair_risk
                })
        
        risk_pairs.sort(key=lambda x: x['pair_risk_contribution'], reverse=True)
        
        return {
            'avg_risk_weighted_correlation': avg_risk_weighted_corr,
            'highest_risk_pairs': risk_pairs[:10],
            'correlation_risk_concentration': np.std([pair['pair_risk_contribution'] for pair in risk_pairs])
        }
    
    def _calculate_risk_weighted_correlation(self, correlation_matrix: np.ndarray,
                                           weights: np.ndarray) -> float:
        """Calculate risk-weighted average correlation"""
        risk_weighted_sum = 0
        weight_sum = 0
        
        for i in range(len(weights)):
            for j in range(i + 1, len(weights)):
                weight_product = weights[i] * weights[j]
                correlation = correlation_matrix[i, j]
                risk_weighted_sum += weight_product * correlation
                weight_sum += weight_product
        
        return risk_weighted_sum / weight_sum if weight_sum > 0 else 0
    
    def _calculate_gini_coefficient(self, weights: np.ndarray) -> float:
        """Calculate Gini coefficient for weight distribution"""
        sorted_weights = np.sort(weights)
        n = len(weights)
        index = np.arange(1, n + 1)
        return (2 * np.sum(index * sorted_weights)) / (n * np.sum(sorted_weights)) - (n + 1) / n
    
    def _analyze_concentration_risk(self, weights: np.ndarray,
                                  strategy_names: List[str]) -> Dict[str, Any]:
        """Analyze concentration risk in the portfolio"""
        # Sort by weights
        sorted_indices = np.argsort(weights)[::-1]
        
        # Calculate cumulative concentration
        cumulative_weights = []
        cumulative_pct = 0
        
        for i, idx in enumerate(sorted_indices):
            cumulative_pct += weights[idx]
            cumulative_weights.append({
                'rank': i + 1,
                'strategy': strategy_names[idx] if idx < len(strategy_names) else f'Strategy_{idx}',
                'weight': weights[idx],
                'cumulative_weight': cumulative_pct
            })
        
        # Concentration thresholds
        concentration_analysis = {
            'top_5_concentration': cumulative_weights[4]['cumulative_weight'] if len(cumulative_weights) > 4 else 1.0,
            'top_10_concentration': cumulative_weights[9]['cumulative_weight'] if len(cumulative_weights) > 9 else 1.0,
            'top_20_concentration': cumulative_weights[19]['cumulative_weight'] if len(cumulative_weights) > 19 else 1.0,
        }
        
        return {
            'cumulative_weights': cumulative_weights,
            'concentration_thresholds': concentration_analysis,
            'concentration_risk_level': self._assess_concentration_risk_level(concentration_analysis)
        }
    
    def _analyze_weight_distribution(self, weights: np.ndarray) -> Dict[str, Any]:
        """Analyze the distribution of portfolio weights"""
        equal_weight = 1 / len(weights)
        
        return {
            'equal_weight_benchmark': equal_weight,
            'deviation_from_equal_weight': np.mean(np.abs(weights - equal_weight)),
            'max_overweight': np.max(weights) - equal_weight,
            'max_underweight': equal_weight - np.min(weights),
            'weight_percentiles': {
                f'{p}th': np.percentile(weights, p)
                for p in [10, 25, 50, 75, 90]
            }
        }
    
    def _recommend_cluster_selection(self, cluster_indices: Dict[str, List[int]],
                                   inter_cluster_correlations: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Recommend optimal cluster selection for diversification"""
        # Simple heuristic: select clusters with lowest inter-cluster correlations
        cluster_scores = {}
        
        for cluster_id in cluster_indices.keys():
            # Score based on average correlation with other clusters
            correlations_with_others = []
            
            for pair_key, corr_data in inter_cluster_correlations.items():
                if f'cluster_{cluster_id}' in pair_key:
                    correlations_with_others.append(abs(corr_data['avg_correlation']))
            
            avg_correlation_with_others = np.mean(correlations_with_others) if correlations_with_others else 0
            cluster_scores[cluster_id] = {
                'diversification_score': 1 - avg_correlation_with_others,
                'avg_correlation_with_others': avg_correlation_with_others,
                'cluster_size': len(cluster_indices[cluster_id])
            }
        
        # Sort by diversification score
        ranked_clusters = sorted(cluster_scores.items(), key=lambda x: x[1]['diversification_score'], reverse=True)
        
        return {
            'cluster_rankings': ranked_clusters,
            'recommended_clusters': [cluster_id for cluster_id, _ in ranked_clusters[:3]],  # Top 3
            'selection_rationale': 'Selected clusters with lowest average inter-cluster correlations for maximum diversification benefit'
        }
    
    def _assess_concentration_risk_level(self, concentration_analysis: Dict[str, float]) -> str:
        """Assess the concentration risk level"""
        top_10_concentration = concentration_analysis['top_10_concentration']
        
        if top_10_concentration > 0.8:
            return 'high'
        elif top_10_concentration > 0.6:
            return 'medium'
        else:
            return 'low'
    
    def _generate_diversification_summary(self, diversification_results: Dict[str, Any],
                                        portfolio: List[int],
                                        strategy_names: List[str]) -> Dict[str, Any]:
        """Generate comprehensive diversification summary"""
        
        # Extract key metrics
        portfolio_div = diversification_results.get('portfolio_diversification', {})
        effectiveness = diversification_results.get('diversification_effectiveness', {})
        correlation_structure = diversification_results.get('correlation_structure', {})
        
        summary = {
            'portfolio_size': len(portfolio),
            'diversification_score': effectiveness.get('diversification_efficiency', 0),
            'average_correlation': correlation_structure.get('correlation_statistics', {}).get('average_correlation', 0),
            'effective_number_of_assets': effectiveness.get('effective_number_of_assets', len(portfolio)),
            'diversification_benefit': effectiveness.get('diversification_benefit', 0),
            'concentration_risk': diversification_results.get('concentration_analysis', {}).get('concentration_risk', {}).get('concentration_risk_level', 'unknown'),
            'key_insights': [],
            'improvement_opportunities': len(diversification_results.get('portfolio_diversification', {}).get('improvement_suggestions', []))
        }
        
        # Generate insights
        if summary['diversification_score'] > 0.8:
            summary['key_insights'].append('Portfolio shows strong diversification characteristics')
        elif summary['diversification_score'] < 0.5:
            summary['key_insights'].append('Portfolio may benefit from improved diversification')
        
        if summary['average_correlation'] > 0.5:
            summary['key_insights'].append('High average correlation may limit diversification benefits')
        
        if summary['effective_number_of_assets'] < len(portfolio) * 0.7:
            summary['key_insights'].append('Effective number of assets is significantly lower than portfolio size due to correlations')
        
        return summary