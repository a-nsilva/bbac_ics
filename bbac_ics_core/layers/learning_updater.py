#!/usr/bin/env python3
"""
BBAC ICS Framework - Continuous Learning Layer
Updates baselines and models with trusted samples.
"""

import pandas as pd
from pathlib import Path
from typing import List, Dict
from collections import defaultdict
from ..utils.data_structures import AccessRequest, AccessDecision, DecisionOutput
from ..layers.baseline_manager import BaselineManager
from ..models.lstm_predictor import LSTMPredictor
from ..utils.config_loader import ConfigLoader


class LearningUpdater:
    """
    Manages continuous learning loop.
    - Buffers trusted samples
    - Updates baselines when buffer is full
    - Updates ML models
    """
    
    def __init__(
        self,
        baseline_manager: BaselineManager,
        lstm_predictor: LSTMPredictor,
        config: Dict = None
    ):
        """
        Initialize learning updater.
        
        Args:
            baseline_manager: BaselineManager instance
            lstm_predictor: LSTMPredictor instance
            config: Learning configuration
        """
        if config is None:
            config = ConfigLoader.load().get('learning', {})
        
        self.baseline_manager = baseline_manager
        self.lstm_predictor = lstm_predictor
        
        self.buffer_size = config.get('buffer_size', 1000)
        self.min_samples = config.get('min_samples_for_update', 100)
        self.trust_threshold = config.get('trust_threshold', 0.8)
        
        # Trusted buffer per agent: {agent_id: [requests]}
        self.trusted_buffer: Dict[str, List[AccessRequest]] = defaultdict(list)
        
        # Quarantine for suspicious samples
        self.quarantine: List[AccessRequest] = []
        
        # Paths
        paths_config = ConfigLoader.load().get('paths', {})
        self.quarantine_dir = Path(paths_config.get('quarantine_dir', 'quarantine'))
        self.quarantine_dir.mkdir(exist_ok=True)
    
    def process_decision(
        self,
        request: AccessRequest,
        decision: AccessDecision
    ):
        """
        Process decision for continuous learning.
        
        Args:
            request: Original request
            decision: Final decision
        """
        # Determine if trusted
        is_trusted = self._is_trusted(request, decision)
        
        if is_trusted:
            # Add to trusted buffer
            agent_id = request.agent_id
            self.trusted_buffer[agent_id].append(request)
            
            # Check if buffer is full
            if len(self.trusted_buffer[agent_id]) >= self.buffer_size:
                self._update_models(agent_id)
        else:
            # Add to quarantine
            self.quarantine.append(request)
    
    def _is_trusted(
        self,
        request: AccessRequest,
        decision: AccessDecision
    ) -> bool:
        """
        Determine if sample is trusted for learning.
        
        Args:
            request: Access request
            decision: Decision made
            
        Returns:
            True if trusted
        """
        # Only trust 'allow' decisions with high confidence
        #if decision.decision != "allow":
        if decision.decision != DecisionOutput.ALLOW.value:
            return False
        
        if decision.confidence < self.trust_threshold:
            return False
        
        # Additional checks
        if request.auth_status.value != "success":
            return False
        
        if request.attempt_count > 2:
            return False
        
        return True
    
    def _update_models(self, agent_id: str):
        """
        Update baseline and ML models for agent.
        
        Args:
            agent_id: Agent identifier
        """
        buffer = self.trusted_buffer[agent_id]
        
        if len(buffer) < self.min_samples:
            return
        
        # Convert to DataFrame
        data = []
        for req in buffer:
            data.append({
                'timestamp': req.timestamp,
                'action': req.action.value,
                'resource': req.resource,
                'location': req.location,
                'human_present': req.human_present,
                'attempt_count': req.attempt_count,
            })
        
        df = pd.DataFrame(data)
        
        # Update baseline
        self.baseline_manager.update_baseline(agent_id, df)
        
        # Update LSTM transition probabilities
        sequences = self._extract_sequences(buffer)
        self.lstm_predictor.update_model(sequences)
        
        # Clear buffer
        self.trusted_buffer[agent_id] = []
        
        print(f"[LearningUpdater] Updated models for agent {agent_id} with {len(buffer)} samples")
    
    def _extract_sequences(
        self,
        requests: List[AccessRequest]
    ) -> List[List[str]]:
        """
        Extract action sequences from requests.
        
        Args:
            requests: List of access requests
            
        Returns:
            List of action sequences
        """
        # Group by session
        sessions: Dict[str, List[str]] = defaultdict(list)
        
        for req in requests:
            session_id = req.session_id or req.agent_id
            sessions[session_id].append(req.action.value)
        
        return list(sessions.values())
    
    def save_quarantine(self):
        """Save quarantined samples to disk."""
        if not self.quarantine:
            return
        
        data = []
        for req in self.quarantine:
            data.append({
                'request_id': req.request_id,
                'timestamp': req.timestamp,
                'agent_id': req.agent_id,
                'action': req.action.value,
                'resource': req.resource,
                'reason': 'untrusted_decision'
            })
        
        df = pd.DataFrame(data)
        filepath = self.quarantine_dir / f"quarantine_{int(time.time())}.csv"
        df.to_csv(filepath, index=False)
        
        print(f"[LearningUpdater] Saved {len(self.quarantine)} quarantined samples to {filepath}")
        
        # Clear quarantine
        self.quarantine = []



