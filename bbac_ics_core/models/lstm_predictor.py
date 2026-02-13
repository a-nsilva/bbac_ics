#!/usr/bin/env python3
"""
BBAC ICS Framework - LSTM Sequence Predictor
Implements sequence-based behavioral analysis using LSTM.
"""
import time
import numpy as np
from collections import deque
from typing import Dict, List
from ..utils.data_structures import AccessRequest, LayerDecision, ActionType
from .markov_chain import MarkovChain


class LSTMPredictor:
    """LSTM-based sequence predictor for behavioral patterns."""
    
    def __init__(
        self,
        sequence_length: int = 5,
        anomaly_threshold: float = 0.5,
        use_markov: bool = True
    ):
        """
        Initialize LSTM predictor (currently uses Markov Chain).
        
        Args:
            sequence_length: Number of past actions to consider
            anomaly_threshold: Threshold for anomaly detection
            use_markov: Use Markov Chain (True) or fallback heuristics (False)
        """
        self.sequence_length = sequence_length
        self.anomaly_threshold = anomaly_threshold
        self.use_markov = use_markov
        
        # Use Markov Chain as sequence model
        if use_markov:
            self.markov_chain = MarkovChain(
                sequence_length=sequence_length,
                anomaly_threshold=anomaly_threshold
            )
        else:
            # Fallback: track sequences manually
            self.sequences: Dict[str, deque] = {}
            self.transition_probs: Dict[tuple, float] = {}
    
    def analyze(self, request: AccessRequest) -> LayerDecision:
        """
        Analyze request using sequence model.
        
        Args:
            request: AccessRequest object
            
        Returns:
            LayerDecision with score and decision
        """
        if self.use_markov:
            # Delegate to Markov Chain
            return self.markov_chain.analyze(request)
        else:
            # Fallback to heuristic logic
            return self._analyze_heuristic(request)
    
    def _compute_sequence_anomaly(
        self,
        sequence: List[str],
        current_action: ActionType,
        request: AccessRequest
    ) -> float:
        """
        Compute anomaly score based on action sequence.
        
        Args:
            sequence: Historical action sequence
            current_action: Current action being requested
            request: Full request context
            
        Returns:
            Anomaly score [0, 1]
        """
        # If no history, use weak signal
        if not sequence:
            return 0.2
        
        anomaly = 0.0
        
        # Check transition probability (simplified Markov)
        prev_action = sequence[-1]
        transition = (prev_action, current_action.value)
        
        # Lookup or estimate transition probability
        # TODO: Replace with LSTM prediction
        prob = self.transition_probs.get(transition, 0.1)
        
        if prob < 0.1:  # Rare transition
            anomaly += 0.4
        elif prob < 0.3:  # Uncommon transition
            anomaly += 0.2
        
        # Check for suspicious patterns
        # Rapid action changes
        if len(sequence) >= 3:
            unique_recent = len(set(sequence[-3:]))
            if unique_recent >= 3:  # 3 different actions in last 3 steps
                anomaly += 0.2
        
        # Repetitive failures
        if request.auth_status.value == "failed" and len(sequence) >= 2:
            if sequence[-1] == sequence[-2] == current_action.value:
                anomaly += 0.3  # Retry pattern
        
        return min(anomaly, 1.0)
    
    def update_model(self, trusted_sequences: List[List[str]]):
        """
        Update sequence model from trusted data.
        
        Args:
            trusted_sequences: List of action sequences from trusted behavior
        """
        if self.use_markov:
            self.markov_chain.update_model(trusted_sequences)
        else:
            # Fallback: update transition probabilities
            self._update_transition_probs(trusted_sequences)

    def _analyze_heuristic(self, request: AccessRequest) -> LayerDecision:
        """Fallback heuristic analysis (original logic)."""
        start = time.time()
        
        agent_id = request.agent_id
        if agent_id not in self.sequences:
            self.sequences[agent_id] = deque(maxlen=self.sequence_length)
        
        sequence = list(self.sequences[agent_id])
        anomaly_score = self._compute_sequence_anomaly(sequence, request.action, request)
        self.sequences[agent_id].append(request.action.value)
        
        score = 1.0 - anomaly_score
        decision = "grant" if score >= self.anomaly_threshold else "deny"
        latency_ms = (time.time() - start) * 1000
        
        return LayerDecision(
            layer_name="sequence",
            score=score,
            decision=decision,
            confidence=score,
            latency_ms=latency_ms,
            explanation={
                "anomaly_score": anomaly_score,
                "sequence_length": len(sequence),
                "current_action": request.action.value,
                "threshold": self.anomaly_threshold,
                "model": "heuristic"
            }
        )
    
    def _update_transition_probs(self, trusted_sequences: List[List[str]]):
        """Update transition probabilities (fallback)."""
        transition_counts: Dict[tuple, int] = {}
        total_transitions = 0
        
        for seq in trusted_sequences:
            for i in range(len(seq) - 1):
                transition = (seq[i], seq[i + 1])
                transition_counts[transition] = transition_counts.get(transition, 0) + 1
                total_transitions += 1
        
        if total_transitions > 0:
            self.transition_probs = {
                k: v / total_transitions for k, v in transition_counts.items()
            }
