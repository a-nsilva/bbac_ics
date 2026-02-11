#!/usr/bin/env python3
"""
BBAC ICS Framework - Decision Maker Layer
Applies thresholds to fusion score and generates final access decision.
"""
import time
from typing import Dict
from ..utils.data_structures import (
    AccessRequest,
    AccessDecision,
    HybridDecision,
    DecisionType
)
from ..utils.config_loader import ConfigLoader


class DecisionMaker:
    """
    Final decision layer applying threshold logic.
    Maps fusion score to: allow, mfa, review, deny, auto_deny + alert
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize decision maker.
        
        Args:
            config: Thresholds configuration
        """
        if config is None:
            config = ConfigLoader.load().get('thresholds', {})
        
        self.t_min_deny = config.get('t_min_deny', 0.2)
        self.t1_review = config.get('t1_review', 0.4)
        self.t2_mfa = config.get('t2_mfa', 0.6)
        self.high_conf_alert = config.get('high_confidence_alert', 0.8)
    
    def decide(
        self,
        request: AccessRequest,
        hybrid_decision: HybridDecision
    ) -> AccessDecision:
        """
        Generate final access decision with thresholds.
        
        Args:
            request: Original access request
            hybrid_decision: Fused decision from layers
            
        Returns:
            AccessDecision with final verdict
        """
        start = time.time()
        
        score = hybrid_decision.score
        
        # Apply threshold logic (from flowchart)
        if score < self.t_min_deny:
            decision = "auto_deny"
            reason = "critical_anomaly_detected"
            alert = True
        elif score < self.t1_review:
            decision = "review"
            reason = "suspicious_pattern_requires_review"
            alert = False
        elif score < self.t2_mfa:
            decision = "mfa"
            reason = "additional_authentication_required"
            alert = False
        else:  # score >= t2_mfa
            decision = "allow"
            reason = "request_approved"
            alert = False
        
        # High confidence alert (monitoring)
        if score >= self.high_conf_alert and decision == "allow":
            reason = "high_confidence_approval"
        
        latency_ms = (time.time() - start) * 1000
        total_latency = hybrid_decision.total_latency_ms + latency_ms
        
        # Build layer decisions breakdown
        layer_decisions_dict = {
            layer_name: {
                'score': ld.score,
                'decision': ld.decision,
                'confidence': ld.confidence,
                'latency_ms': ld.latency_ms,
                'explanation': ld.explanation
            }
            for layer_name, ld in hybrid_decision.layer_results.items()
        }
        
        # Add fusion layer
        layer_decisions_dict['fusion'] = {
            'score': score,
            'decision': hybrid_decision.decision.value,
            'confidence': hybrid_decision.confidence,
            'latency_ms': hybrid_decision.total_latency_ms,
            'explanation': hybrid_decision.explanation
        }
        
        return AccessDecision(
            request_id=request.request_id,
            timestamp=time.time(),
            decision=decision,
            confidence=hybrid_decision.confidence,
            latency_ms=total_latency,
            reason=reason,
            layer_decisions=layer_decisions_dict
        )
