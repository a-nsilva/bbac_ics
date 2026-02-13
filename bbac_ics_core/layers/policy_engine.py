#!/usr/bin/env python3
"""
BBAC ICS Framework - Policy Engine (RuBAC)
Implements Rule-based and Behavior-Attribute-based Access Control.
"""
import time
from typing import Dict, List, Set

from ..utils.config_loader import ConfigLoader
from ..utils.data_structures import (
    AccessRequest,
    ActionType,
    AgentRole,
    LayerDecision,
    ResourceType,
)


class PolicyEngine:
    """Rule-based Access Control (RuBAC) engine."""
    
    def __init__(self, config: Dict = None):
        """
        Initialize policy engine.
        
        Args:
            config: Policy configuration
        """
        if config is None:
            config = ConfigLoader.load().get('policy', {})
        
        self.maintenance_hours = config.get('maintenance_hours', [2, 3])
        
        # Define RBAC policies
        self._init_policies()
    
    def _init_policies(self):
        """Initialize role-based access policies."""
        
        # Role → Allowed Actions
        self.role_actions: Dict[AgentRole, Set[ActionType]] = {
            AgentRole.SUPERVISOR: {
                ActionType.READ, ActionType.WRITE, ActionType.EXECUTE,
                ActionType.OVERRIDE, ActionType.EMERGENCY_STOP,
                ActionType.EMERGENCY_OVERRIDE
            },
            AgentRole.OPERATOR: {
                ActionType.READ, ActionType.WRITE, ActionType.EXECUTE,
                ActionType.MONITOR
            },
            AgentRole.TECHNICIAN: {
                ActionType.READ, ActionType.MAINTENANCE,
                ActionType.CALIBRATION, ActionType.DIAGNOSTIC
            },
            AgentRole.ASSEMBLY_ROBOT: {
                ActionType.READ, ActionType.WRITE, ActionType.EXECUTE
            },
            AgentRole.TRANSPORT_ROBOT: {
                ActionType.READ, ActionType.TRANSPORT
            },
            AgentRole.INSPECTION_ROBOT: {
                ActionType.READ, ActionType.MONITOR
            },
        }
        
        # Critical resources requiring human supervision
        self.critical_resources: Set[ResourceType] = {
            ResourceType.EMERGENCY_CONTROLS,
            ResourceType.SAFETY_SYSTEM,
            ResourceType.POWER_CONTROL,
            ResourceType.ADMIN_PANEL
        }
        
        # Emergency override roles
        self.emergency_roles: Set[AgentRole] = {
            AgentRole.SUPERVISOR,
            AgentRole.SAFETY_ROBOT
        }
    
    def analyze(self, request: AccessRequest) -> LayerDecision:
        """
        Evaluate request against policies.
        
        Args:
            request: AccessRequest object
            
        Returns:
            LayerDecision with policy compliance score
        """
        #start = time.time()
        start = time.perf_counter()
        
        violations = []
        compliance_score = 1.0
        
        # Rule 1: Role-based action permissions
        allowed_actions = self.role_actions.get(request.agent_role, set())
        if request.action not in allowed_actions:
            violations.append("action_not_allowed_for_role")
            compliance_score -= 0.4
        
        # Rule 2: Critical resources require human presence
        if request.resource_type in self.critical_resources:
            if not request.human_present:
                violations.append("critical_resource_requires_human")
                compliance_score -= 0.3
        
        # Rule 3: Emergency actions require authorized roles
        if request.emergency:
            if request.agent_role not in self.emergency_roles:
                violations.append("emergency_action_unauthorized_role")
                compliance_score -= 0.5
        
        # Rule 4: Maintenance windows
        # TODO: Extract hour from timestamp
        # if current_hour in self.maintenance_hours:
        #     if request.action not in [ActionType.MAINTENANCE, ActionType.READ]:
        #         violations.append("non_maintenance_action_during_window")
        #         compliance_score -= 0.2
        
        # Rule 5: Write operations require successful auth
        if request.action in [ActionType.WRITE, ActionType.DELETE, ActionType.EXECUTE]:
            if request.auth_status.value != "success":
                violations.append("write_requires_successful_auth")
                compliance_score -= 0.6
        
        # Clip score
        compliance_score = max(0.0, compliance_score)
        
        decision = "grant" if compliance_score >= 0.5 else "deny"
        
        #latency_ms = (time.time() - start) * 1000
        latency_ms = (time.perf_counter() - start) * 1000
        
        return LayerDecision(
            layer_name="policy",
            score=compliance_score,
            decision=decision,
            confidence=1.0,  # Policy decisions are deterministic
            latency_ms=latency_ms,
            explanation={
                "violations": violations,
                "compliance_score": compliance_score,
                "rules_checked": 5
            }
        )


