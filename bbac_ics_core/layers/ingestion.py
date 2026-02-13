#!/usr/bin/env python3
"""
BBAC ICS Framework - Ingestion Layer
Implements:
- Authentication filtering
- Data preprocessing and normalization
- Batch and stream processing
"""
import pandas as pd
from typing import Dict
from ..utils.data_structures import AccessRequest
from ..utils.data_utils import compute_temporal_features

def ingest_single(raw_event: Dict) -> AccessRequest:
    """
    Process single event for ROS node (real-time).
    
    Args:
        raw_event: Raw event dictionary from topic
        
    Returns:
        AccessRequest object
    """
    return AccessRequest.from_raw(raw_event)

"""
def ingest_batch(
    df: pd.DataFrame,
    max_attempts: int = 3,
    drop_failed_auth: bool = False
) -> pd.DataFrame:
    events = df.copy()
    
    # Authentication filter
    if drop_failed_auth:
        events = events[
            (events["auth_status"] == "success") &
            (events["attempt_count"] <= max_attempts)
        ]
    else:
        events = events[events["attempt_count"] <= max_attempts]
    
    # Temporal ordering
    events = events.sort_values("timestamp").reset_index(drop=True)
    
    # Normalization
    events = _normalize(events)
    
    return events
"""
        
def ingest_batch(
    df: pd.DataFrame,
    max_attempts: int = 3,
    drop_failed_auth: bool = False
) -> pd.DataFrame:
    """
    Process dataset batch for experiments.
    
    Args:
        df: DataFrame with raw events
        max_attempts: Maximum authentication attempts allowed
        drop_failed_auth: Whether to drop failed authentications
        
    Returns:
        Filtered and normalized DataFrame
    """
    events = df.copy()
    
    # Authentication filter
    if drop_failed_auth:
        events = events[
            (events["auth_status"] == "success") &
            (events["attempt_count"] <= max_attempts)
        ]
    else:
        events = events[events["attempt_count"] <= max_attempts]
    
    # Temporal ordering
    events = events.sort_values("timestamp").reset_index(drop=True)
    
    # Compute temporal features (time_gap)
    events = compute_temporal_features(events)
    
    # Normalization
    events = _normalize(events)
    
    return events
    

def _normalize(events: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize categorical fields.
    
    Args:
        events: DataFrame with events
        
    Returns:
        Normalized DataFrame
    """
    events = events.copy()
    
    categorical_cols = [
        "agent_type", "robot_type", "human_role", "action",
        "resource", "resource_type", "location", "auth_status"
    ]
    
    for col in categorical_cols:
        if col in events.columns:
            events[col] = events[col].astype(str).str.lower().str.strip()
    
    return events

