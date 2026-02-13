#!/usr/bin/env python3
"""
BBAC ICS Framework - Behavioral Baseline Layer
Implements adaptive baseline computation using sliding window (70% recent, 30% historical).
"""
import pandas as pd
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
from scipy.stats import entropy
from ..utils.config_loader import ConfigLoader


class BaselineManager:
    """
    Manages adaptive baselines per agent.
    Baseline(t) = w * recent + (1-w) * historical
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize baseline manager.
        
        Args:
            config: Baseline configuration (from params.yaml)
        """
        if config is None:
            config = ConfigLoader.load().get('baseline', {})
        
        self.window_days = config.get('window_days', 10)
        self.recent_weight = config.get('recent_weight', 0.7)
        self.max_historical = config.get('max_historical_baselines', 10)
        
        # Storage: {agent_id: {'current': baseline_dict, 'history': [...]}}
        self.baselines: Dict[str, Dict] = {}
        
        # Paths
        paths_config = ConfigLoader.load().get('paths', {}) 
        profiles_path = paths_config.get('profiles_dir', 'profiles')
        self.profiles_dir = Path(profiles_path)
        self.profiles_dir.mkdir(exist_ok=True)    

    
    def compute_baseline(
        self,
        agent_id: str,
        events: pd.DataFrame,
        window_size: Optional[int] = None
    ) -> Dict:
        """
        Compute adaptive baseline for an agent.
        
        Args:
            agent_id: Agent identifier
            events: Historical events for this agent
            window_size: Override default window size
            
        Returns:
            Baseline statistics dictionary
        """
        if events.empty:
            return {}
        
        events = events.sort_values("timestamp").reset_index(drop=True)
        window_size = window_size or self.window_days
        
        # Split recent vs historical
        recent = events.tail(window_size)
        historical = events.iloc[:-window_size] if len(events) > window_size else pd.DataFrame()
        
        # Compute stats
        baseline_recent = self._compute_stats(recent)
        baseline_hist = self._compute_stats(historical) if not historical.empty else {}
        
        # Weighted merge
        baseline = self._merge_baselines(
            baseline_recent,
            baseline_hist,
            self.recent_weight
        )
        
        # Store
        self._store_baseline(agent_id, baseline)
        
        return baseline
    
    def get_baseline(self, agent_id: str) -> Dict:
        """Get current baseline for agent."""
        if agent_id not in self.baselines:
            return {}
        return self.baselines[agent_id].get('current', {})
    
    def update_baseline(self, agent_id: str, new_events: pd.DataFrame):
        """Update baseline with new trusted events."""
        # Get historical events
        historical = self._load_historical_events(agent_id)
        
        # Append new events
        all_events = pd.concat([historical, new_events], ignore_index=True)
        
        # Recompute baseline
        baseline = self.compute_baseline(agent_id, all_events)
        
        # Archive old baseline
        self._archive_baseline(agent_id)
        
        return baseline
    
    def _compute_stats(self, df: pd.DataFrame) -> Dict:
        """Extract behavioral statistics from window."""
        stats = {}
        
        if df.empty:
            return stats
        
        # Categorical frequencies
        stats["action_freq"] = df["action"].value_counts(normalize=True).to_dict()
        stats["resource_freq"] = df["resource"].value_counts(normalize=True).to_dict()
        stats["location_freq"] = df["location"].value_counts(normalize=True).to_dict()
        
        # Temporal gaps (assuming float timestamps)
        time_diffs = df["timestamp"].diff().dropna()
        mean_gap = time_diffs.mean() if not time_diffs.empty else 0.0
        std_gap = time_diffs.std() if not time_diffs.empty else 0.0
        #stats["mean_gap"] = float(time_diffs.mean()) if not time_diffs.empty else 0.0
        #stats["std_gap"] = float(time_diffs.std()) if not time_diffs.empty else 0.0
        stats["mean_gap"] = mean_gap.total_seconds() if hasattr(mean_gap, 'total_seconds') else float(mean_gap)
        stats["std_gap"] = std_gap.total_seconds() if hasattr(std_gap, 'total_seconds') else float(std_gap)    
        
        # Human presence probability
        stats["human_presence_prob"] = float(df["human_present"].mean())
        
        # Average attempts
        stats["avg_attempts"] = float(df["attempt_count"].mean())
        
        return stats
    
    def _merge_baselines(
        self,
        recent: Dict,
        historical: Dict,
        w: float
    ) -> Dict:
        """Weighted merge of recent and historical baselines."""
        merged = {}
        keys = set(recent.keys()).union(historical.keys())
        
        for key in keys:
            if isinstance(recent.get(key), dict):
                # Merge distributions
                merged[key] = self._merge_distributions(
                    recent.get(key, {}),
                    historical.get(key, {}),
                    w
                )
            else:
                # Merge scalars
                r = recent.get(key, 0.0)
                h = historical.get(key, 0.0)
                merged[key] = w * r + (1 - w) * h
        
        return merged
    
    def _merge_distributions(
        self,
        recent: Dict[str, float],
        historical: Dict[str, float],
        w: float
    ) -> Dict[str, float]:
        """Merge probability distributions."""
        merged = {}
        keys = set(recent.keys()).union(historical.keys())
        
        for k in keys:
            merged[k] = w * recent.get(k, 0.0) + (1 - w) * historical.get(k, 0.0)
        
        return merged
    
    def _store_baseline(self, agent_id: str, baseline: Dict):
        """Store baseline in memory."""
        if agent_id not in self.baselines:
            self.baselines[agent_id] = {'current': {}, 'history': []}
        
        self.baselines[agent_id]['current'] = baseline
    
    def _archive_baseline(self, agent_id: str):
        """Archive current baseline to history."""
        if agent_id not in self.baselines:
            return
        
        current = self.baselines[agent_id].get('current')
        if current:
            history = self.baselines[agent_id].get('history', [])
            history.append(current)
            
            # Keep only max_historical baselines
            if len(history) > self.max_historical:
                history = history[-self.max_historical:]
            
            self.baselines[agent_id]['history'] = history
    
    def _load_historical_events(self, agent_id: str) -> pd.DataFrame:
        """Load historical events from storage (placeholder)."""
        # TODO: Implement persistent storage
        return pd.DataFrame()
    
    def save(self, filepath: Optional[Path] = None):
        """Save all baselines to disk."""
        if filepath is None:
            filepath = self.profiles_dir / "baselines.pkl"
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.baselines, f)
    
    def load(self, filepath: Optional[Path] = None):
        """Load baselines from disk."""
        if filepath is None:
            filepath = self.profiles_dir / "baselines.pkl"
        
        if not filepath.exists():
            return
        
        with open(filepath, 'rb') as f:
            self.baselines = pickle.load(f)

    def detect_drift(
        self,
        agent_id: str,
        current_baseline: Dict,
        method: str = 'kl_divergence',
        threshold: float = 0.15
    ) -> Tuple[bool, float]:
        """
        Detect behavioral drift between current and historical baseline.
        
        Args:
            agent_id: Agent identifier
            current_baseline: Current baseline statistics
            method: Drift detection method ('kl_divergence' or 'ks_test')
            threshold: Drift threshold
            
        Returns:
            (drift_detected, drift_score) tuple
        """
        # Get historical baseline
        if agent_id not in self.baselines:
            return False, 0.0
        
        historical = self.baselines[agent_id].get('current', {})
        
        if not historical or not current_baseline:
            return False, 0.0
        
        if method == 'kl_divergence':
            drift_score = self._compute_kl_drift(historical, current_baseline)
        elif method == 'ks_test':
            drift_score = self._compute_ks_drift(historical, current_baseline)
        else:
            drift_score = 0.0
        
        drift_detected = drift_score > threshold
        
        return drift_detected, drift_score
    
    def _compute_kl_drift(
        self,
        baseline1: Dict,
        baseline2: Dict
    ) -> float:
        """
        Compute KL divergence between two baselines.
        
        Measures difference in action distributions.
        
        Args:
            baseline1: First baseline
            baseline2: Second baseline
            
        Returns:
            KL divergence score (higher = more drift)
        """
        # Compare action frequency distributions
        dist1 = baseline1.get('action_freq', {})
        dist2 = baseline2.get('action_freq', {})
        
        if not dist1 or not dist2:
            return 0.0
        
        # Get all actions
        all_actions = set(dist1.keys()).union(dist2.keys())
        
        if not all_actions:
            return 0.0
        
        # Build probability arrays with smoothing
        epsilon = 1e-10
        p = np.array([dist1.get(a, 0.0) + epsilon for a in all_actions])
        q = np.array([dist2.get(a, 0.0) + epsilon for a in all_actions])
        
        # Normalize
        p = p / p.sum()
        q = q / q.sum()
        
        # Compute KL divergence
        kl_div = float(entropy(p, q))
        
        return kl_div
    
    def _compute_ks_drift(
        self,
        baseline1: Dict,
        baseline2: Dict
    ) -> float:
        """
        Compute Kolmogorov-Smirnov statistic for drift.
        
        Args:
            baseline1: First baseline
            baseline2: Second baseline
            
        Returns:
            KS statistic (higher = more drift)
        """
        from scipy.stats import ks_2samp
        
        # Compare temporal gap distributions
        gap1 = baseline1.get('mean_gap', 0.0)
        std1 = baseline1.get('std_gap', 1.0)
        
        gap2 = baseline2.get('mean_gap', 0.0)
        std2 = baseline2.get('std_gap', 1.0)
        
        # Generate samples from distributions (approximation)
        if std1 > 0 and std2 > 0:
            samples1 = np.random.normal(gap1, std1, 100)
            samples2 = np.random.normal(gap2, std2, 100)
            
            # KS test
            statistic, p_value = ks_2samp(samples1, samples2)
            
            return float(statistic)
        
        return 0.0



