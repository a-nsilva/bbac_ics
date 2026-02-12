#!/usr/bin/env python3
"""
BBAC ICS Framework - Feature Extraction Layer
Extracts features from AccessRequest using baseline statistics.
"""

import pandas as pd
from typing import Dict
from ..utils.data_structures import AccessRequest


class FeatureExtractor:
    """Extract features from access requests using baseline."""
    
    def __init__(self, baseline_manager):
        """
        Initialize feature extractor.
        
        Args:
            baseline_manager: BaselineManager instance
        """
        self.baseline_manager = baseline_manager
    
    def extract(self, request: AccessRequest) -> Dict[str, float]:
        """
        Extract feature vector from request.
        
        Args:
            request: AccessRequest object
            
        Returns:
            Dictionary of features
        """
        features = {}
        
        # Get agent baseline
        baseline = self.baseline_manager.get_baseline(request.agent_id)
        
        if not baseline:
            # No baseline yet - return default features
            return self._default_features(request)
        
        # Categorical likelihood features
        features["action_prob"] = self._lookup_prob(
            baseline.get("action_freq", {}),
            request.action.value
        )
        
        features["resource_prob"] = self._lookup_prob(
            baseline.get("resource_freq", {}),
            request.resource
        )
        
        features["location_prob"] = self._lookup_prob(
            baseline.get("location_freq", {}),
            request.location
        )
        
        # Temporal deviation (requires time_gap in request.context)
        mean_gap = baseline.get("mean_gap", 0.0)
        std_gap = baseline.get("std_gap", 1.0)
        current_gap = request.context.get("time_gap", mean_gap)
        
        features["gap_deviation"] = abs(current_gap - mean_gap) / (std_gap + 1e-6)
        
        # Human presence deviation
        expected_hp = baseline.get("human_presence_prob", 0.5)
        features["human_presence_diff"] = abs(float(request.human_present) - expected_hp)
        
        # Attempt count deviation
        avg_attempts = baseline.get("avg_attempts", 1.0)
        features["attempt_deviation"] = (request.attempt_count - avg_attempts) / (avg_attempts + 1e-6)
        
        # Context flags
        features["emergency_flag"] = float(request.emergency)
        features["auth_failed"] = float(request.auth_status.value != "success")
        
        return features
    
    def _lookup_prob(self, distribution: Dict[str, float], key: str) -> float:
        """Lookup probability in distribution."""
        if not distribution or key is None:
            return 0.0
        return float(distribution.get(key, 0.0))
    
    def _default_features(self, request: AccessRequest) -> Dict[str, float]:
        """Return default features when no baseline exists."""
        return {
            "action_prob": 0.5,
            "resource_prob": 0.5,
            "location_prob": 0.5,
            "gap_deviation": 0.0,
            "human_presence_diff": 0.0,
            "attempt_deviation": 0.0,
            "emergency_flag": float(request.emergency),
            "auth_failed": float(request.auth_status.value != "success"),
        }


def attach_temporal_gap(events: pd.DataFrame) -> pd.DataFrame:
    """
    Add time_gap column (seconds since previous event).
    For batch processing.
    
    Args:
        events: DataFrame with timestamp column (float)
        
    Returns:
        DataFrame with time_gap column
    """
    events = events.copy()
    events["time_gap"] = events["timestamp"].diff().fillna(0.0)
    return events

