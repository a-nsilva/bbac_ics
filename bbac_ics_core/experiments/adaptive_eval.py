#!/usr/bin/env python3
"""
BBAC ICS Framework - Adaptive Evaluation
Tests baseline adaptability and drift detection.
"""
import pandas as pd
import numpy as np
from typing import Dict, List
from pathlib import Path
import json
import matplotlib.pyplot as plt

from ..layers.baseline_manager import BaselineManager
from ..utils.config_loader import ConfigLoader
from ..utils.data_loader import DataLoader
from .metrics_calculator import MetricsCalculator


class AdaptiveEvaluation:
    """
    Evaluate adaptive baseline performance.
    
    Tests:
    1. Baseline convergence rate
    2. Drift detection accuracy
    3. Sliding window effectiveness (70/30 split)
    4. Adaptation stability
    """
    
    def __init__(self, output_dir: str = "results/adaptive"):
        """Initialize adaptive evaluation."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.config = ConfigLoader.load()
        self.metrics_calc = MetricsCalculator()
    
    def run(self) -> Dict:
        """Run adaptive evaluation tests."""
        print("=" * 50)
        print("ADAPTIVE BASELINE EVALUATION")
        print("=" * 50)
        
        results = {}
        
        # Test 1: Baseline convergence
        print("\n[1] Testing baseline convergence rate...")
        results['convergence'] = self._test_convergence()
        
        # Test 2: Sliding window effectiveness
        print("\n[2] Testing sliding window (70/30)...")
        results['sliding_window'] = self._test_sliding_window()
        
        # Test 3: Adaptation to drift
        print("\n[3] Testing drift adaptation...")
        results['drift_adaptation'] = self._test_drift_adaptation()
        
        # Save results
        self._save_results(results)
        
        return results
    
    def _test_convergence(self) -> Dict:
        """Test how quickly baseline converges."""
        
        data_loader = DataLoader()
        data_loader.load_all() 
        df = data_loader.load_split('train')
        
        # Select single agent
        agent_id = df['agent_id'].iloc[0]
        agent_data = df[df['agent_id'] == agent_id].sort_values('timestamp')
        
        baseline_manager = BaselineManager(self.config.get('baseline'))
        
        # Track baseline evolution over time
        window_sizes = [10, 50, 100, 200, 500, 1000]
        convergence_metrics = []
        
        for window_size in window_sizes:
            if len(agent_data) < window_size:
                continue
            
            data_slice = agent_data.iloc[:window_size]
            baseline = baseline_manager.compute_baseline(agent_id, data_slice, window_size=window_size)
            
            # Measure stability (variance in action frequencies)
            action_freqs = list(baseline.get('action_freq', {}).values())
            stability = np.std(action_freqs) if action_freqs else 1.0
            
            convergence_metrics.append({
                'window_size': window_size,
                'stability': stability,
                'num_actions': len(baseline.get('action_freq', {}))
            })
        
        print(f"  Convergence tested over {len(convergence_metrics)} window sizes")
        
        return {
            'agent_id': agent_id,
            'metrics': convergence_metrics
        }
    
    def _test_sliding_window(self) -> Dict:
        """Test 70/30 weighting effectiveness."""
        
        data_loader = DataLoader()
        data_loader.load_all() 
        df = data_loader.load_split('train')
        
        agent_id = df['agent_id'].iloc[0]
        agent_data = df[df['agent_id'] == agent_id].sort_values('timestamp')
        
        if len(agent_data) < 20:
            return {'error': 'insufficient_data'}
        
        # Compare different weight configurations
        weight_configs = [
            (1.0, 0.0),  # 100% recent
            (0.7, 0.3),  # 70/30 (default)
            (0.5, 0.5),  # 50/50
            (0.3, 0.7),  # 30/70
            (0.0, 1.0),  # 100% historical
        ]
        
        results = []
        
        for recent_weight, hist_weight in weight_configs:
            config = self.config.get('baseline', {}).copy()
            config['recent_weight'] = recent_weight
            
            baseline_manager = BaselineManager(config)
            baseline = baseline_manager.compute_baseline(agent_id, agent_data)
            
            # Evaluate baseline quality (how well it represents recent behavior)
            recent_data = agent_data.tail(10)
            action_match = self._compute_action_match(baseline, recent_data)
            
            results.append({
                'recent_weight': recent_weight,
                'hist_weight': hist_weight,
                'action_match_score': action_match
            })
        
        print(f"  Tested {len(weight_configs)} weight configurations")
        
        return {'results': results}
    
    def _test_drift_adaptation(self) -> Dict:
        """Test adaptation to behavioral drift."""
        
        # Simulate drift: change action distribution halfway through
        data_loader = DataLoader()
        data_loader.load_all() 
        df = data_loader.load_split('train')
        
        agent_id = df['agent_id'].iloc[0]
        agent_data = df[df['agent_id'] == agent_id].sort_values('timestamp')
        
        if len(agent_data) < 100:
            return {'error': 'insufficient_data'}
        
        # Split: pre-drift and post-drift
        split_point = len(agent_data) // 2
        pre_drift = agent_data.iloc[:split_point]
        post_drift = agent_data.iloc[split_point:]
        
        baseline_manager = BaselineManager(self.config.get('baseline'))
        
        # Baseline on pre-drift
        baseline_pre = baseline_manager.compute_baseline(agent_id, pre_drift)
        
        # Update with post-drift data
        baseline_post = baseline_manager.update_baseline(agent_id, post_drift)
        
        # Measure adaptation (change in action distribution)
        action_drift = self._compute_distribution_drift(
            baseline_pre.get('action_freq', {}),
            baseline_post.get('action_freq', {})
        )
        
        print(f"  Action distribution drift: {action_drift:.4f}")
        
        return {
            'agent_id': agent_id,
            'action_drift': action_drift,
            'pre_drift_actions': len(baseline_pre.get('action_freq', {})),
            'post_drift_actions': len(baseline_post.get('action_freq', {}))
        }
    
    def _compute_action_match(self, baseline: Dict, recent_data: pd.DataFrame) -> float:
        """Compute how well baseline matches recent behavior."""
        action_freq_baseline = baseline.get('action_freq', {})
        action_freq_recent = recent_data['action'].value_counts(normalize=True).to_dict()
        
        # Compute overlap (Jaccard similarity)
        all_actions = set(action_freq_baseline.keys()).union(action_freq_recent.keys())
        
        if not all_actions:
            return 0.0
        
        match_score = 0.0
        for action in all_actions:
            baseline_prob = action_freq_baseline.get(action, 0.0)
            recent_prob = action_freq_recent.get(action, 0.0)
            match_score += min(baseline_prob, recent_prob)
        
        return match_score
    
    def _compute_distribution_drift(self, dist1: Dict, dist2: Dict) -> float:
        """Compute KL divergence between distributions."""
        from scipy.stats import entropy
        
        all_keys = set(dist1.keys()).union(dist2.keys())
        
        if not all_keys:
            return 0.0
        
        p = [dist1.get(k, 1e-10) for k in all_keys]
        q = [dist2.get(k, 1e-10) for k in all_keys]
        
        # Normalize
        p = np.array(p) / sum(p)
        q = np.array(q) / sum(q)
        
        return float(entropy(p, q))
    
    def _save_results(self, results: Dict):
        """Save adaptive evaluation results."""
        output_file = self.output_dir / "adaptive_results.json"
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {output_file}")


def main():
    """Run adaptive evaluation."""
    evaluation = AdaptiveEvaluation()
    results = evaluation.run()


if __name__ == '__main__':
    main()
