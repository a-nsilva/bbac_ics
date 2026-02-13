#!/usr/bin/env python3
"""
BBAC ICS Framework - Score Fusion Layer
Combines decisions from statistical, sequence, and policy layers.
"""
import time
from typing import Dict, List, Optional

from ..models.ensemble_predictor import EnsemblePredictor

from ..utils.config_loader import ConfigLoader
from ..utils.data_structures import (    
    DecisionType,
    FusionStrategy,
    HybridDecision,
    LayerDecision
)


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

        # Meta-classifier (ensemble)
        self.use_meta_classifier = config.get('use_meta_classifier', False)
        self.ensemble: Optional[EnsemblePredictor] = None
        
        if self.use_meta_classifier:
            model_type = config.get('meta_classifier_model', 'logistic_regression')
            self.ensemble = EnsemblePredictor(model_type=model_type)
    
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
        #start = time.time()
        start = time.perf_counter()
        
        # Map layer names to weight keys
        layer_map = {
            'policy': 'rule',
            'statistical': 'behavioral',
            'sequence': 'ml'
        }
        """
        if self.method == 'weighted_voting':
            score, strategy = self._weighted_voting(layer_decisions, layer_map)
        elif self.method == 'rule_priority':
            score, strategy = self._rule_priority(layer_decisions)
        elif self.method == 'high_confidence_denial':
            score, strategy = self._high_confidence_denial(layer_decisions, layer_map)
        else:
            # Default to weighted voting
            score, strategy = self._weighted_voting(layer_decisions, layer_map)
        """
        # Use meta-classifier if available and trained
        if self.use_meta_classifier and self.ensemble and self.ensemble.is_trained:
            score, strategy = self._meta_classifier_fusion(layer_decisions, layer_map)
        elif self.method == 'weighted_voting':
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
        
        #latency_ms = (time.time() - start) * 1000
        latency_ms = (time.perf_counter() - start) * 1000
        
        return HybridDecision(
            request_id=request_id,
            decision=decision,
            score=score,
            confidence=confidence,
            fusion_strategy=strategy,
            layer_results=layer_decisions,
            total_latency_ms=latency_ms,
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


    def _meta_classifier_fusion(
        self,
        layer_decisions: Dict[str, LayerDecision],
        layer_map: Dict[str, str]
    ) -> tuple:
        """
        Use meta-classifier (ensemble) to combine layer scores.
        
        Args:
            layer_decisions: Layer decisions
            layer_map: Mapping of layer names to weight keys
            
        Returns:
            (score, strategy) tuple
        """
        # Extract scores in order: rule, behavioral, ml
        rule_score = layer_decisions.get('policy', LayerDecision(
            layer_name='policy', score=0.5, decision='deny', confidence=0.5, latency_ms=0
        )).score
        
        behavioral_score = layer_decisions.get('statistical', LayerDecision(
            layer_name='statistical', score=0.5, decision='deny', confidence=0.5, latency_ms=0
        )).score
        
        ml_score = layer_decisions.get('sequence', LayerDecision(
            layer_name='sequence', score=0.5, decision='deny', confidence=0.5, latency_ms=0
        )).score
        
        # Predict with ensemble
        score = self.ensemble.predict_score(rule_score, behavioral_score, ml_score)
        
        return score, FusionStrategy.META_CLASSIFIER
    
    def train_ensemble(
        self,
        layer_scores_list: List[tuple],
        ground_truth: List[int]
    ):
        """
        Train meta-classifier on historical data.
        
        Args:
            layer_scores_list: List of (rule, behavioral, ml) score tuples
            ground_truth: List of binary labels (0=deny, 1=allow)
        """
        if not self.use_meta_classifier or not self.ensemble:
            raise ValueError("Meta-classifier not enabled in config")
        
        self.ensemble.train(layer_scores_list, ground_truth)
    
    def get_ensemble_importance(self) -> Dict:
        """Get feature importance from ensemble."""
        if not self.ensemble or not self.ensemble.is_trained:
            return {}
        
        return self.ensemble.get_feature_importance()



