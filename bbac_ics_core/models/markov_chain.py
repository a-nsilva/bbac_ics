#!/usr/bin/env python3
"""
BBAC ICS Framework - Markov Chain Sequence Analyzer
First-order Markov Chain for action sequence prediction.
"""
import time
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from ..utils.data_structures import AccessRequest, LayerDecision, ActionType


class MarkovChain:
    """
    First-order Markov Chain for action sequence analysis.
    
    Models P(action_t | action_t-1) transition probabilities.
    """
    
    def __init__(
        self,
        sequence_length: int = 5,
        anomaly_threshold: float = 0.5,
        min_transition_count: int = 2
    ):
        """
        Initialize Markov Chain.
        
        Args:
            sequence_length: Number of past actions to track
            anomaly_threshold: Threshold for anomaly detection
            min_transition_count: Minimum observations for valid transition
        """
        self.sequence_length = sequence_length
        self.anomaly_threshold = anomaly_threshold
        self.min_transition_count = min_transition_count
        
        # Track sequences per agent: {agent_id: deque([action1, action2, ...])}
        from collections import deque
        self.sequences: Dict[str, deque] = {}
        
        # Transition counts: {(prev_action, current_action): count}
        self.transition_counts: Dict[Tuple[str, str], int] = defaultdict(int)
        
        # Transition probabilities: {(prev_action, current_action): probability}
        self.transition_probs: Dict[Tuple[str, str], float] = {}
        
        # Action frequencies
        self.action_counts: Dict[str, int] = defaultdict(int)
        self.total_transitions = 0
    
    def analyze(self, request: AccessRequest) -> LayerDecision:
        """
        Analyze request using Markov Chain.
        
        Args:
            request: AccessRequest object
            
        Returns:
            LayerDecision with score and decision
        """
        start = time.time()
        
        # Get agent's action sequence
        agent_id = request.agent_id
        if agent_id not in self.sequences:
            from collections import deque
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
            score=float(score),
            decision=decision,
            confidence=float(score),
            latency_ms=latency_ms,
            explanation={
                "anomaly_score": float(anomaly_score),
                "sequence_length": len(sequence),
                "current_action": request.action.value,
                "threshold": self.anomaly_threshold,
                "transition_prob": self._get_transition_prob(
                    sequence[-1] if sequence else None,
                    request.action.value
                ) if sequence else 0.0
            }
        )
    
    def _compute_sequence_anomaly(
        self,
        sequence: List[str],
        current_action: ActionType,
        request: AccessRequest
    ) -> float:
        """
        Compute anomaly score based on Markov transition probabilities.
        
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
        
        # Get transition probability
        prev_action = sequence[-1]
        transition_prob = self._get_transition_prob(prev_action, current_action.value)
        
        # Low probability transitions are anomalous
        if transition_prob == 0.0:
            anomaly += 0.5  # Never seen this transition
        elif transition_prob < 0.05:
            anomaly += 0.4  # Very rare transition
        elif transition_prob < 0.1:
            anomaly += 0.2  # Uncommon transition
        
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
    
    def _get_transition_prob(
        self,
        prev_action: Optional[str],
        current_action: str
    ) -> float:
        """
        Get transition probability P(current | prev).
        
        Args:
            prev_action: Previous action
            current_action: Current action
            
        Returns:
            Transition probability [0, 1]
        """
        if prev_action is None:
            # Prior probability
            if self.total_transitions == 0:
                return 0.5  # No data yet
            return self.action_counts.get(current_action, 0) / self.total_transitions
        
        transition = (prev_action, current_action)
        
        # Laplace smoothing
        count = self.transition_counts.get(transition, 0)
        total_from_prev = sum(
            c for (p, _), c in self.transition_counts.items() if p == prev_action
        )
        
        if total_from_prev == 0:
            return 0.0
        
        # Add-one smoothing
        num_actions = len(set(a for _, a in self.transition_counts.keys()))
        smoothed_prob = (count + 1) / (total_from_prev + num_actions)
        
        return smoothed_prob
    
    def train(self, training_sequences: List[List[str]]):
        """
        Train Markov Chain on action sequences.
        
        Args:
            training_sequences: List of action sequences
        """
        # Reset counts
        self.transition_counts.clear()
        self.action_counts.clear()
        self.total_transitions = 0
        
        # Count transitions
        for sequence in training_sequences:
            for i in range(len(sequence)):
                action = sequence[i]
                self.action_counts[action] += 1
                
                if i > 0:
                    prev_action = sequence[i - 1]
                    transition = (prev_action, action)
                    self.transition_counts[transition] += 1
                    self.total_transitions += 1
        
        # Compute probabilities
        self._compute_transition_probs()
    
    def _compute_transition_probs(self):
        """Compute transition probability matrix."""
        self.transition_probs.clear()
        
        # Group by previous action
        prev_actions = set(p for p, _ in self.transition_counts.keys())
        
        for prev_action in prev_actions:
            # Total transitions from this action
            total = sum(
                count for (p, _), count in self.transition_counts.items()
                if p == prev_action
            )
            
            # Compute probabilities
            for (p, curr), count in self.transition_counts.items():
                if p == prev_action and count >= self.min_transition_count:
                    self.transition_probs[(p, curr)] = count / total
    
    def update_model(self, trusted_sequences: List[List[str]]):
        """
        Update model with new trusted sequences (incremental).
        
        Args:
            trusted_sequences: List of action sequences from trusted behavior
        """
        # Add to existing counts
        for sequence in trusted_sequences:
            for i in range(len(sequence)):
                action = sequence[i]
                self.action_counts[action] += 1
                
                if i > 0:
                    prev_action = sequence[i - 1]
                    transition = (prev_action, action)
                    self.transition_counts[transition] += 1
                    self.total_transitions += 1
        
        # Recompute probabilities
        self._compute_transition_probs()
    
    def get_statistics(self) -> Dict:
        """Get Markov Chain statistics."""
        return {
            "total_transitions": self.total_transitions,
            "unique_actions": len(self.action_counts),
            "unique_transitions": len(self.transition_counts),
            "transition_matrix_size": len(self.transition_probs),
            "most_common_actions": sorted(
                self.action_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
        }
