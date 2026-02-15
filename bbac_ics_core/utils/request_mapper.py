#!/usr/bin/env python3
"""
BBAC ICS Framework - Request Mapper
Responsible ONLY for converting raw dictionaries into AccessRequest objects.
Pure domain conversion (no pandas dependency).
"""

import logging
from typing import Dict, Any, List, Optional

from bbac_ics_core.utils.data_structures import (
    AccessRequest,
    AgentType,
    AgentRole,
    ActionType,
    AuthStatus,
    ResourceType,
)

logger = logging.getLogger(__name__)


class RequestMapper:
    """
    Converts raw dictionary records into AccessRequest domain objects.
    Completely independent from pandas.
    """

    # ==========================================================
    # Public API
    # ==========================================================

    def map_record(self, record: Dict[str, Any]) -> AccessRequest:
        """Convert single dictionary record into AccessRequest."""

        request_id = str(record.get("log_id", ""))

        timestamp_raw = record.get("timestamp")
        timestamp = self._parse_timestamp(timestamp_raw)

        agent_id = str(record.get("agent_id", ""))

        agent_type = self._parse_agent_type(record.get("agent_type"))
        agent_role = self._parse_agent_role(record, agent_type)

        action = self._parse_action(record.get("action"))

        resource = str(record.get("resource", ""))
        resource_type = self._parse_resource_type(record.get("resource_type"))

        location = str(record.get("location", "unknown"))

        human_present = self._parse_bool(record.get("human_present"))
        emergency = self._parse_bool(record.get("emergency_flag"))

        session_id = self._safe_str(record.get("session_id"))
        previous_action = self._parse_optional_action(record.get("previous_action"))

        auth_status = self._parse_auth_status(record.get("auth_status"))
        attempt_count = int(record.get("attempt_count", 0) or 0)

        policy_id = self._safe_str(record.get("policy_id"))

        return AccessRequest(
            request_id=request_id,
            timestamp=timestamp,
            agent_id=agent_id,
            agent_type=agent_type,
            agent_role=agent_role,
            action=action,
            resource=resource,
            resource_type=resource_type,
            location=location,
            human_present=human_present,
            emergency=emergency,
            session_id=session_id,
            previous_action=previous_action,
            auth_status=auth_status,
            attempt_count=attempt_count,
            policy_id=policy_id,
        )

    def dataframe_to_requests(
        self,
        df,
        max_requests: Optional[int] = None,
    ) -> List[AccessRequest]:
        """
        Converts DataFrame to AccessRequest list.
        Only pandas dependency boundary.
        """

        if max_requests:
            df = df.head(max_requests)

        records = df.to_dict(orient="records")

        return [self.map_record(r) for r in records]

    # ==========================================================
    # Parsing helpers
    # ==========================================================

    def _parse_timestamp(self, value) -> float:
        if value is None:
            return 0.0
        try:
            return float(value.timestamp()) if hasattr(value, "timestamp") else float(value)
        except Exception:
            return 0.0

    def _parse_bool(self, value) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ("true", "1", "yes")
        return bool(value)

    def _safe_str(self, value):
        return str(value) if value is not None else None

    def _parse_agent_type(self, value):
        try:
            return AgentType.from_string(str(value or "robot"))
        except Exception:
            return AgentType.ROBOT

    def _parse_agent_role(self, record, agent_type):
        role_str = record.get("robot_type") or record.get("human_role")
        try:
            return AgentRole.from_string(str(role_str or ""), agent_type)
        except Exception:
            return AgentRole.UNKNOWN

    def _parse_action(self, value):
        try:
            return ActionType.from_string(str(value or "read"))
        except Exception:
            return ActionType.READ

    def _parse_optional_action(self, value):
        if not value:
            return None
        try:
            return ActionType.from_string(str(value))
        except Exception:
            return None

    def _parse_auth_status(self, value):
        try:
            return AuthStatus.from_string(str(value or "success"))
        except Exception:
            return AuthStatus.SUCCESS

    def _parse_resource_type(self, value):
        try:
            return ResourceType.from_string(str(value or "other"))
        except Exception:
            return ResourceType.OTHER
