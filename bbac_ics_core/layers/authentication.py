#!/usr/bin/env python3
"""
BBAC ICS Framework - Authentication Layer
Simulates authentication validation for experimental purposes.
"""
import time
from typing import Dict, Tuple
from ..utils.data_structures import AccessRequest, AuthStatus


class AuthenticationModule:
    """Simulate authentication checks."""
    
    def __init__(self, max_attempts: int = 3):
        """
        Initialize authentication module.
        
        Args:
            max_attempts: Maximum login attempts before lockout
        """
        self.max_attempts = max_attempts
        # Track attempts per session: {session_id: count}
        self.session_attempts: Dict[str, int] = {}
    
    def authenticate(self, request: AccessRequest) -> Tuple[bool, str]:
        """
        Validate authentication for request.
        
        Args:
            request: AccessRequest object
            
        Returns:
            (is_valid, reason) tuple
        """
        # Check if auth already failed
        if request.auth_status == AuthStatus.FAILED:
            return False, "authentication_failed"
        
        # Check if locked
        if request.auth_status == AuthStatus.LOCKED:
            return False, "account_locked"
        
        # Check attempt count
        session_id = request.session_id or request.agent_id
        current_attempts = self.session_attempts.get(session_id, 0)
        
        # Update attempt count
        if request.attempt_count > 0:
            self.session_attempts[session_id] = request.attempt_count
            current_attempts = request.attempt_count
        
        # Exceeded attempts?
        if current_attempts > self.max_attempts:
            return False, "max_attempts_exceeded"
        
        # Valid authentication
        return True, "authenticated"
    
    def reset_attempts(self, session_id: str):
        """Reset attempt counter for session."""
        if session_id in self.session_attempts:
            del self.session_attempts[session_id]
