#!/usr/bin/env python3
"""
BBAC ICS Framework - Statistical Detector (Behavioral Analysis Layer)
Implements statistical anomaly detection using baseline comparison.
"""
import time
from typing import Dict
from ..utils.data_structures import AccessRequest, LayerDecision
from ..layers.feature_extractor import FeatureExtractor


class StatisticalDetector:
    """Statistical anomaly detector using behavioral baseline."""
    
    def __init__(
        self,
        feature_extractor: FeatureExtractor,
        anomaly_threshold: float = 0.5
    ):
        """
        Initialize statistical detector.
        
        Args:
            feature_extractor: FeatureExtractor instance
            anomaly_threshold: Threshold for anomaly score
        """
        self.feature_extractor = feature_extractor
        self.anomaly_threshold = anomaly_threshold
    
    def analyze(self, request: AccessRequest) -> LayerDecision:
        """
        Analyze request for statistical anomalies.
        
        Args:
            request: AccessRequest object
            
        Returns:
            LayerDecision with score and decision
        """
        #start = time.time()
        start = time.perf_counter()
        
        # Extract features
        features = self.feature_extractor.extract(request)
        
        # Compute anomaly score
        anomaly_score = self._compute_anomaly_score(features, request)
        
        # Convert to legitimacy score (inverse of anomaly)
        score = 1.0 - anomaly_score
        
        # Decision based on threshold
        decision = "grant" if score >= self.anomaly_threshold else "deny"
        
        #latency_ms = (time.time() - start) * 1000
        latency_ms = (time.perf_counter() - start) * 1000
        
        return LayerDecision(
            layer_name="statistical",
            score=score,
            decision=decision,
            confidence=score,
            latency_ms=latency_ms,
            explanation={
                "anomaly_score": anomaly_score,
                "features": features,
                "threshold": self.anomaly_threshold
            }
        )
    
    def _compute_anomaly_score(
        self,
        features: Dict[str, float],
        request: AccessRequest
    ) -> float:
        """
        Compute aggregate anomaly score from features.
        
        Args:
            features: Extracted features
            request: Original request
            
        Returns:
            Anomaly score [0, 1]
        """
        anomaly = 0.0
        
        # Low probability actions/resources/locations are suspicious
        anomaly += (1.0 - features["action_prob"]) * 0.2
        anomaly += (1.0 - features["resource_prob"]) * 0.2
        anomaly += (1.0 - features["location_prob"]) * 0.1
        
        # Temporal anomalies
        if features["gap_deviation"] > 2.0:  # > 2 std deviations
            anomaly += 0.15
        
        # Human presence mismatch
        anomaly += features["human_presence_diff"] * 0.1
        
        # High attempt count
        if features["attempt_deviation"] > 2.0:
            anomaly += 0.2
        
        # Critical flags
        if features["emergency_flag"] and not request.human_present:
            anomaly += 0.3
        
        if features["auth_failed"]:
            anomaly += 0.25
        
        # Clip to [0, 1]
        return min(anomaly, 1.0)

