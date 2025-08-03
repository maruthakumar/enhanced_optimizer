#!/usr/bin/env python3
"""
Legacy System Comparison Engine
Compares results between legacy optimizer and new Parquet/Arrow/cuDF implementation
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LegacyComparison:
    """
    Comparison engine for validating new implementation against legacy results
    """
    
    def __init__(self, tolerance: float = 0.0001):
        """
        Initialize comparison engine
        
        Args:
            tolerance: Acceptable tolerance for fitness value comparison (0.01%)
        """
        self.tolerance = tolerance
        self.comparison_results = []
        
    def compare_fitness_values(self, 
                             legacy_fitness: float, 
                             new_fitness: float,
                             portfolio_size: int,
                             algorithm: str) -> Dict[str, Any]:
        """
        Compare fitness values between implementations
        
        Args:
            legacy_fitness: Fitness from legacy system
            new_fitness: Fitness from new system
            portfolio_size: Size of portfolio
            algorithm: Algorithm used
            
        Returns:
            Comparison result dictionary
        """
        absolute_diff = abs(legacy_fitness - new_fitness)
        relative_diff = absolute_diff / legacy_fitness if legacy_fitness != 0 else float('inf')
        
        result = {
            'portfolio_size': portfolio_size,
            'algorithm': algorithm,
            'legacy_fitness': legacy_fitness,
            'new_fitness': new_fitness,
            'absolute_difference': absolute_diff,
            'relative_difference': relative_diff,
            'percentage_difference': relative_diff * 100,
            'within_tolerance': relative_diff <= self.tolerance,
            'timestamp': datetime.now().isoformat()
        }
        
        # Log comparison
        if result['within_tolerance']:
            logger.info(f"✅ Fitness match for size {portfolio_size} ({algorithm}): "
                       f"Legacy={legacy_fitness:.6f}, New={new_fitness:.6f}, "
                       f"Diff={result['percentage_difference']:.4f}%")
        else:
            logger.warning(f"❌ Fitness mismatch for size {portfolio_size} ({algorithm}): "
                          f"Legacy={legacy_fitness:.6f}, New={new_fitness:.6f}, "
                          f"Diff={result['percentage_difference']:.4f}%")
        
        self.comparison_results.append(result)
        return result
        
    def compare_portfolios(self,
                         legacy_strategies: List[str],
                         new_strategies: List[str],
                         portfolio_size: int) -> Dict[str, Any]:
        """
        Compare portfolio composition between implementations
        
        Args:
            legacy_strategies: Strategy list from legacy system
            new_strategies: Strategy list from new system
            portfolio_size: Size of portfolio
            
        Returns:
            Portfolio comparison result
        """
        # Clean strategy names for comparison
        legacy_set = set(self._clean_strategy_names(legacy_strategies))
        new_set = set(self._clean_strategy_names(new_strategies))
        
        # Calculate overlap
        common_strategies = legacy_set.intersection(new_set)
        legacy_only = legacy_set - new_set
        new_only = new_set - legacy_set
        
        overlap_percentage = len(common_strategies) / len(legacy_set) * 100 if legacy_set else 0
        
        result = {
            'portfolio_size': portfolio_size,
            'legacy_count': len(legacy_strategies),
            'new_count': len(new_strategies),
            'common_strategies': len(common_strategies),
            'overlap_percentage': overlap_percentage,
            'legacy_only': list(legacy_only),
            'new_only': list(new_only),
            'meets_threshold': overlap_percentage >= 90.0  # 90% overlap threshold
        }
        
        logger.info(f"Portfolio overlap for size {portfolio_size}: {overlap_percentage:.1f}%")
        
        return result
        
    def compare_metrics(self,
                       legacy_metrics: Dict[str, float],
                       new_metrics: Dict[str, float],
                       portfolio_size: int) -> Dict[str, Any]:
        """
        Compare individual performance metrics
        
        Args:
            legacy_metrics: Metrics from legacy system
            new_metrics: Metrics from new system
            portfolio_size: Size of portfolio
            
        Returns:
            Metrics comparison result
        """
        metric_comparisons = {}
        
        # Define expected metric mappings
        metric_map = {
            'total_roi': 'total_roi',
            'max_drawdown': 'max_drawdown',
            'win_percentage': 'win_rate',
            'profit_factor': 'profit_factor',
            'sharpe_ratio': 'sharpe_ratio',
            'sortino_ratio': 'sortino_ratio'
        }
        
        for legacy_key, new_key in metric_map.items():
            if legacy_key in legacy_metrics and new_key in new_metrics:
                legacy_val = legacy_metrics[legacy_key]
                new_val = new_metrics[new_key]
                
                if legacy_val != 0:
                    relative_diff = abs(legacy_val - new_val) / abs(legacy_val)
                else:
                    relative_diff = float('inf') if new_val != 0 else 0
                    
                metric_comparisons[legacy_key] = {
                    'legacy': legacy_val,
                    'new': new_val,
                    'relative_difference': relative_diff,
                    'percentage_difference': relative_diff * 100,
                    'within_tolerance': relative_diff <= 0.01  # 1% tolerance for metrics
                }
                
        result = {
            'portfolio_size': portfolio_size,
            'metric_comparisons': metric_comparisons,
            'all_metrics_match': all(m['within_tolerance'] for m in metric_comparisons.values())
        }
        
        return result
        
    def compare_algorithm_rankings(self,
                                 legacy_results: Dict[str, Dict],
                                 new_results: Dict[str, Dict]) -> Dict[str, Any]:
        """
        Compare algorithm performance rankings
        
        Args:
            legacy_results: Algorithm results from legacy system
            new_results: Algorithm results from new system
            
        Returns:
            Algorithm ranking comparison
        """
        # Extract algorithm fitness scores
        legacy_scores = {alg: res.get('fitness', 0) for alg, res in legacy_results.items()}
        new_scores = {alg: res.get('fitness', 0) for alg, res in new_results.items()}
        
        # Get rankings
        legacy_ranking = sorted(legacy_scores.items(), key=lambda x: x[1], reverse=True)
        new_ranking = sorted(new_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Compare top performers
        legacy_top3 = [alg for alg, _ in legacy_ranking[:3]]
        new_top3 = [alg for alg, _ in new_ranking[:3]]
        
        result = {
            'legacy_ranking': [(alg, score) for alg, score in legacy_ranking],
            'new_ranking': [(alg, score) for alg, score in new_ranking],
            'legacy_best': legacy_ranking[0] if legacy_ranking else None,
            'new_best': new_ranking[0] if new_ranking else None,
            'top3_match': legacy_top3 == new_top3,
            'best_algorithm_match': (legacy_ranking[0][0] == new_ranking[0][0]) if legacy_ranking and new_ranking else False
        }
        
        return result
        
    def validate_precision_differences(self,
                                     legacy_value: float,
                                     new_value: float,
                                     value_type: str) -> Dict[str, Any]:
        """
        Validate and document expected precision differences
        
        Args:
            legacy_value: Value from legacy system (CPU)
            new_value: Value from new system (GPU)
            value_type: Type of value being compared
            
        Returns:
            Precision validation result
        """
        # Calculate ULP (Unit in Last Place) difference
        if legacy_value == 0 and new_value == 0:
            ulp_diff = 0
        elif legacy_value == 0 or new_value == 0:
            ulp_diff = float('inf')
        else:
            # Approximate ULP calculation
            ulp_diff = abs(legacy_value - new_value) / (np.finfo(float).eps * max(abs(legacy_value), abs(new_value)))
            
        result = {
            'value_type': value_type,
            'legacy_value': legacy_value,
            'new_value': new_value,
            'absolute_difference': abs(legacy_value - new_value),
            'ulp_difference': ulp_diff,
            'expected_precision_issue': ulp_diff < 100,  # Within 100 ULPs is acceptable
            'explanation': self._get_precision_explanation(value_type, ulp_diff)
        }
        
        return result
        
    def generate_comparison_summary(self) -> Dict[str, Any]:
        """
        Generate overall comparison summary
        
        Returns:
            Summary of all comparison results
        """
        if not self.comparison_results:
            return {'status': 'No comparisons performed'}
            
        fitness_matches = sum(1 for r in self.comparison_results if r['within_tolerance'])
        total_comparisons = len(self.comparison_results)
        
        summary = {
            'total_comparisons': total_comparisons,
            'fitness_matches': fitness_matches,
            'fitness_match_rate': fitness_matches / total_comparisons * 100 if total_comparisons > 0 else 0,
            'average_percentage_difference': np.mean([r['percentage_difference'] for r in self.comparison_results]),
            'max_percentage_difference': max(r['percentage_difference'] for r in self.comparison_results),
            'all_within_tolerance': fitness_matches == total_comparisons,
            'comparison_timestamp': datetime.now().isoformat(),
            'detailed_results': self.comparison_results
        }
        
        # Overall verdict
        if summary['all_within_tolerance']:
            summary['verdict'] = 'PASS - All fitness values match within tolerance'
            logger.info("✅ FITNESS VALIDATION PASSED")
        else:
            summary['verdict'] = 'FAIL - Some fitness values exceed tolerance'
            logger.error("❌ FITNESS VALIDATION FAILED")
            
        return summary
        
    def _clean_strategy_names(self, strategies: List[str]) -> List[str]:
        """Clean strategy names for comparison"""
        cleaned = []
        for strategy in strategies:
            # Remove leading numbers and dots
            clean = strategy.strip()
            if '. ' in clean:
                clean = clean.split('. ', 1)[1]
            cleaned.append(clean)
        return cleaned
        
    def _get_precision_explanation(self, value_type: str, ulp_diff: float) -> str:
        """Get explanation for precision differences"""
        if ulp_diff < 10:
            return "Negligible difference - within machine precision"
        elif ulp_diff < 100:
            return "Expected GPU vs CPU floating-point difference"
        elif ulp_diff < 1000:
            return "Moderate precision difference - likely due to algorithm variations"
        else:
            return "Significant difference - may indicate algorithmic divergence"
            
    def save_comparison_report(self, output_path: str):
        """
        Save comparison results to file
        
        Args:
            output_path: Path to save report
        """
        summary = self.generate_comparison_summary()
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
            
        logger.info(f"Comparison report saved to: {output_file}")
        
        # Also save as readable text
        text_file = output_file.with_suffix('.txt')
        with open(text_file, 'w') as f:
            f.write("Legacy System Comparison Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated: {summary['comparison_timestamp']}\n")
            f.write(f"Total Comparisons: {summary['total_comparisons']}\n")
            f.write(f"Fitness Match Rate: {summary['fitness_match_rate']:.2f}%\n")
            f.write(f"Average Difference: {summary['average_percentage_difference']:.4f}%\n")
            f.write(f"Maximum Difference: {summary['max_percentage_difference']:.4f}%\n")
            f.write(f"\nVerdict: {summary['verdict']}\n")
            
            if not summary['all_within_tolerance']:
                f.write("\nFailed Comparisons:\n")
                for result in summary['detailed_results']:
                    if not result['within_tolerance']:
                        f.write(f"  - Size {result['portfolio_size']} ({result['algorithm']}): "
                               f"{result['percentage_difference']:.4f}% difference\n")


def main():
    """Test the comparison engine"""
    # Example usage
    comparison = LegacyComparison(tolerance=0.0001)
    
    # Test fitness comparison
    result = comparison.compare_fitness_values(
        legacy_fitness=30.45764862187442,
        new_fitness=30.45800000000000,
        portfolio_size=37,
        algorithm='SA'
    )
    
    print(json.dumps(result, indent=2))
    
    # Generate summary
    summary = comparison.generate_comparison_summary()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()