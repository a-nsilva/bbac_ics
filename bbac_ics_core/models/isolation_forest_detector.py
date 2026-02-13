#!/usr/bin/env python3
"""
BBAC ICS Framework - Isolation Forest Anomaly Detector
Real ML-based anomaly detection using sklearn IsolationForest.
"""
import time
import numpy as np
from sklearn.ensemble import IsolationForest
from typing import Dict, Optional
from ..utils.data_structures import AccessRequest, LayerDecision
from ..layers.feature_extractor import FeatureExtractor


class IsolationForestDetector:
    """
    Statistical anomaly detector using Isolation Forest.
    
    Uses feature extraction + trained IsolationForest model.
    """
    
    def __init__(
        self,
        feature_extractor: FeatureExtractor,
        contamination: float = 0.1,
        n_estimators: int = 100,
        random_state: int = 42,
        anomaly_threshold: float = 0.5
    ):
        """
        Initialize Isolation Forest detector.
        
        Args:
            feature_extractor: FeatureExtractor instance
            contamination: Expected proportion of anomalies (default 0.1 = 10%)
            n_estimators: Number of trees
            random_state: Random seed for reproducibility
            anomaly_threshold: Threshold for binary decision
        """
        self.feature_extractor = feature_extractor
        self.anomaly_threshold = anomaly_threshold
        
        # Initialize Isolation Forest
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1  # Use all CPU cores
        )
        
        self.is_trained = False
        self.feature_names = [
            'action_prob',
            'resource_prob',
            'location_prob',
            'gap_deviation',
            'human_presence_diff',
            'attempt_deviation',
            'emergency_flag',
            'auth_failed'
        ]
    
    def train(self, training_requests: list):
        """
        Train Isolation Forest on normal behavior.
        
        Args:
            training_requests: List of AccessRequest objects (normal behavior)
        """
        if not training_requests:
            return
        
        # Extract features
        X = []
        for request in training_requests:
            features = self.feature_extractor.extract(request)
            feature_vector = self._features_to_vector(features)
            X.append(feature_vector)
        
        X = np.array(X)
        
        # Train model
        self.model.fit(X)
        self.is_trained = True
    
    def analyze(self, request: AccessRequest) -> LayerDecision:
        """
        Analyze request for anomalies using Isolation Forest.
        
        Args:
            request: AccessRequest object
            
        Returns:
            LayerDecision with score and decision
        """
        #start = time.time()
        start = time.perf_counter()
        
        # Extract features
        features = self.feature_extractor.extract(request)
        feature_vector = self._features_to_vector(features)
        
        if not self.is_trained:
            # Fallback to heuristic scoring if not trained
            anomaly_score = self._heuristic_score(features, request)
        else:
            # Use Isolation Forest prediction
            X = np.array([feature_vector])
            
            # Get anomaly score (-1 = anomaly, 1 = normal)
            prediction = self.model.predict(X)[0]
            
            # Get decision function (lower = more anomalous)
            decision_score = self.model.decision_function(X)[0]
            
            # Normalize to [0, 1] where 0 = anomalous, 1 = normal
            # decision_score ranges approximately [-0.5, 0.5]
            anomaly_score = 1.0 / (1.0 + np.exp(decision_score * 5))  # Sigmoid
            
            # Cap between [0, 1]
            anomaly_score = np.clip(anomaly_score, 0.0, 1.0)
        
        # Convert to legitimacy score (inverse of anomaly)
        score = 1.0 - anomaly_score
        
        # Decision based on threshold
        decision = "grant" if score >= self.anomaly_threshold else "deny"
        
        #latency_ms = (time.time() - start) * 1000
        latency_ms = (time.perf_counter() - start) * 1000
        
        return LayerDecision(
            layer_name="statistical",
            score=float(score),
            decision=decision,
            confidence=float(score),
            latency_ms=latency_ms,
            explanation={
                "anomaly_score": float(anomaly_score),
                "features": features,
                "threshold": self.anomaly_threshold,
                "model_trained": self.is_trained
            }
        )
    
    def _features_to_vector(self, features: Dict[str, float]) -> np.ndarray:
        """Convert feature dict to numpy vector."""
        return np.array([
            features.get(name, 0.0) for name in self.feature_names
        ])
    
    def _heuristic_score(
        self,
        features: Dict[str, float],
        request: AccessRequest
    ) -> float:
        """
        Fallback heuristic scoring when model not trained.
        Same as original StatisticalDetector logic.
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
    
    def update_model(self, new_requests: list):
        """
        Update model with new trusted samples (incremental learning).
        
        Args:
            new_requests: List of new AccessRequest objects
        """
        # Note: sklearn IsolationForest doesn't support incremental learning
        # For now, retrain with all data
        self.train(new_requests)
