#!/usr/bin/env python3
"""
BBAC ICS Framework - Behavioral Baseline Layer
Implements adaptive baseline computation using sliding window (70% recent, 30% historical).
"""
import pandas as pd
import pickle
from pathlib import Path
from typing import Dict, Optional
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
        #profiles_dir = Path(config.get('profiles_dir', 'profiles'))
        profiles_path = config.get('profiles_dir', 'profiles')
        self.profiles_dir = Path(profiles_path) if isinstance(profiles_path, str) else Path('profiles')
        profiles_dir.mkdir(exist_ok=True)
        self.profiles_dir = profiles_dir
    
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
        stats["mean_gap"] = float(time_diffs.mean()) if not time_diffs.empty else 0.0
        stats["std_gap"] = float(time_diffs.std()) if not time_diffs.empty else 0.0
        
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

