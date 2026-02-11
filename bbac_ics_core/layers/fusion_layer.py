#!/usr/bin/env python3
"""
BBAC ICS Framework - Score Fusion Layer
Combines decisions from statistical, sequence, and policy layers.
"""
import time
from typing import Dict, List
from ..utils.data_structures import (
    LayerDecision,
    HybridDecision,
    DecisionType,
    FusionStrategy
)
from ..utils.config_loader import ConfigLoader


class FusionLayer:
    """Fuse multiple layer decisions into final decision."""
    
    def __init__(self, config: Dict = None):
        """
        Initialize fusion layer.
        
        Args:
            config: Fusion configuration
        """
        if config is None:
            config = ConfigLoader.load().get('fusion', {})
        
        self.method = config.get('fusion_method', 'weighted_voting')
        self.weights = config.get('weights', {
            'rule': 0.4,
            'behavioral': 0.3,
            'ml': 0.3
        })
        self.high_conf_threshold = config.get('high_confidence_threshold', 0.9)
        self.decision_threshold = config.get('decision_threshold', 0.5)
    
    def fuse(
        self,
        request_id: str,
        layer_decisions: Dict[str, LayerDecision]
    ) -> HybridDecision:
        """
        Fuse layer decisions into final decision.
        
        Args:
            request_id: Request identifier
            layer_decisions: Dict of {layer_name: LayerDecision}
            
        Returns:
            HybridDecision with final score and decision
        """
        start = time.time()
        
        # Map layer names to weight keys
        layer_map = {
            'policy': 'rule',
            'statistical': 'behavioral',
            'sequence': 'ml'
        }
        
        if self.method == 'weighted_voting':
            score, strategy = self._weighted_voting(layer_decisions, layer_map)
        elif self.method == 'rule_priority':
            score, strategy = self._rule_priority(layer_decisions)
        elif self.method == 'high_confidence_denial':
            score, strategy = self._high_confidence_denial(layer_decisions, layer_map)
        else:
            # Default to weighted voting
            score, strategy = self._weighted_voting(layer_decisions, layer_map)
        
        # Map score to decision type
        decision = self._score_to_decision(score)
        
        # Compute confidence (average of layer confidences)
        confidences = [ld.confidence for ld in layer_decisions.values()]
        confidence = sum(confidences) / len(confidences) if confidences else 0.5
        
        total_latency = (time.time() - start) * 1000
        
        return HybridDecision(
            request_id=request_id,
            decision=decision,
            score=score,
            confidence=confidence,
            fusion_strategy=strategy,
            layer_results=layer_decisions,
            total_latency_ms=total_latency,
            explanation={
                "method": self.method,
                "weights": self.weights,
                "individual_scores": {k: v.score for k, v in layer_decisions.items()}
            }
        )
    
    def _weighted_voting(
        self,
        layer_decisions: Dict[str, LayerDecision],
        layer_map: Dict[str, str]
    ) -> tuple:
        """Weighted average of layer scores."""
        total_score = 0.0
        total_weight = 0.0
        
        for layer_name, decision in layer_decisions.items():
            weight_key = layer_map.get(layer_name, layer_name)
            weight = self.weights.get(weight_key, 0.0)
            
            total_score += decision.score * weight
            total_weight += weight
        
        # Normalize
        final_score = total_score / total_weight if total_weight > 0 else 0.5
        
        return final_score, FusionStrategy.WEIGHTED_VOTING
    
    def _rule_priority(
        self,
        layer_decisions: Dict[str, LayerDecision]
    ) -> tuple:
        """Policy layer takes precedence."""
        policy_decision = layer_decisions.get('policy')
        
        if policy_decision and policy_decision.decision == 'deny':
            return policy_decision.score, FusionStrategy.RULE_PRIORITY
        
        # Fallback to weighted voting
        return self._weighted_voting(layer_decisions, {})
    
    def _high_confidence_denial(
        self,
        layer_decisions: Dict[str, LayerDecision],
        layer_map: Dict[str, str]
    ) -> tuple:
        """If any layer has high-confidence deny, reject immediately."""
        for layer_name, decision in layer_decisions.items():
            if decision.decision == 'deny' and decision.confidence >= self.high_conf_threshold:
                return decision.score, FusionStrategy.HIGH_CONFIDENCE_DENIAL
        
        # Otherwise, weighted voting
        return self._weighted_voting(layer_decisions, layer_map)
    
    def _score_to_decision(self, score: float) -> DecisionType:
        """Map fusion score to decision type (placeholder - DecisionMaker handles thresholds)."""
        if score >= self.decision_threshold:
            return DecisionType.GRANT
        else:
            return DecisionType.DENY
