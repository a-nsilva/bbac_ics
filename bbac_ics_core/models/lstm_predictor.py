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


class LSTMPredictor:
    """LSTM-based sequence predictor for behavioral patterns."""
    
    def __init__(
        self,
        sequence_length: int = 5,
        anomaly_threshold: float = 0.5
    ):
        """
        Initialize LSTM predictor.
        
        Args:
            sequence_length: Number of past actions to consider
            anomaly_threshold: Threshold for anomaly detection
        """
        self.sequence_length = sequence_length
        self.anomaly_threshold = anomaly_threshold
        
        # Track sequences per agent: {agent_id: deque([action1, action2, ...])}
        self.sequences: Dict[str, deque] = {}
        
        # Common action transitions (Markov-like baseline)
        # TODO: Replace with actual LSTM model
        self.transition_probs: Dict[tuple, float] = {}
    
    def analyze(self, request: AccessRequest) -> LayerDecision:
        """
        Analyze request using sequence history.
        
        Args:
            request: AccessRequest object
            
        Returns:
            LayerDecision with score and decision
        """
        start = time.time()
        
        # Get agent's action sequence
        agent_id = request.agent_id
        if agent_id not in self.sequences:
            self.sequences[agent_id] = deque(maxlen=self.sequence_length)
        
        sequence = list(self.sequences[agent_id])
        
        # Compute sequence anomaly score
        anomaly_score = self._compute_sequence_anomaly(
            sequence,
            request.action,
            request
        )
        
        # Update sequence
        self.sequences[agent_id].append(request.action.value)
        
        # Convert to legitimacy score
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
                "threshold": self.anomaly_threshold
            }
        )
    
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
        Update transition probabilities from trusted data.
        
        Args:
            trusted_sequences: List of action sequences from trusted behavior
        """
        # Count transitions
        transition_counts: Dict[tuple, int] = {}
        total_transitions = 0
        
        for seq in trusted_sequences:
            for i in range(len(seq) - 1):
                transition = (seq[i], seq[i + 1])
                transition_counts[transition] = transition_counts.get(transition, 0) + 1
                total_transitions += 1
        
        # Convert to probabilities
        if total_transitions > 0:
            self.transition_probs = {
                k: v / total_transitions
                for k, v in transition_counts.items()
            }
