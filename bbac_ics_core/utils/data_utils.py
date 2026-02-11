#!/usr/bin/env python3
"""
BBAC ICS Framework - Data Utilities
Helper functions for data processing and manipulation.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json


def normalize_dataframe(
    df: pd.DataFrame,
    categorical_cols: List[str] = None
) -> pd.DataFrame:
    """
    Normalize DataFrame columns.
    
    Args:
        df: Input DataFrame
        categorical_cols: List of categorical columns to normalize
        
    Returns:
        Normalized DataFrame
    """
    df = df.copy()
    
    if categorical_cols is None:
        categorical_cols = [
            'agent_type', 'robot_type', 'human_role', 'action',
            'resource_type', 'location', 'auth_status'
        ]
    
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().str.strip()
    
    return df


def split_by_agent(
    df: pd.DataFrame,
    agent_col: str = 'agent_id'
) -> Dict[str, pd.DataFrame]:
    """
    Split DataFrame by agent ID.
    
    Args:
        df: Input DataFrame
        agent_col: Column name for agent identifier
        
    Returns:
        Dictionary mapping agent_id to DataFrame
    """
    agent_data = {}
    
    for agent_id in df[agent_col].unique():
        agent_data[agent_id] = df[df[agent_col] == agent_id].copy()
    
    return agent_data


def compute_temporal_features(
    df: pd.DataFrame,
    timestamp_col: str = 'timestamp'
) -> pd.DataFrame:
    """
    Compute temporal features from timestamps.
    
    Args:
        df: Input DataFrame
        timestamp_col: Timestamp column name
        
    Returns:
        DataFrame with temporal features
    """
    df = df.copy()
    
    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors='coerce')
    
    # Time gap (seconds since previous event)
    df['time_gap'] = df[timestamp_col].diff().dt.total_seconds().fillna(0.0)
    
    # Hour of day
    df['hour'] = df[timestamp_col].dt.hour
    
    # Day of week (0=Monday, 6=Sunday)
    df['day_of_week'] = df[timestamp_col].dt.dayofweek
    
    # Is weekend
    df['is_weekend'] = df['day_of_week'].isin([5, 6])
    
    return df


def encode_categorical(
    df: pd.DataFrame,
    columns: List[str],
    method: str = 'label'
) -> Tuple[pd.DataFrame, Dict]:
    """
    Encode categorical columns.
    
    Args:
        df: Input DataFrame
        columns: Columns to encode
        method: 'label' or 'onehot'
        
    Returns:
        Encoded DataFrame and encoding mapping
    """
    df = df.copy()
    encodings = {}
    
    if method == 'label':
        from sklearn.preprocessing import LabelEncoder
        
        for col in columns:
            if col not in df.columns:
                continue
            
            le = LabelEncoder()
            df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
            encodings[col] = {
                'encoder': le,
                'classes': le.classes_.tolist()
            }
    
    elif method == 'onehot':
        df = pd.get_dummies(df, columns=columns, prefix=columns)
        encodings['method'] = 'onehot'
        encodings['columns'] = columns
    
    return df, encodings


def balance_dataset(
    df: pd.DataFrame,
    label_col: str = 'ground_truth',
    method: str = 'undersample',
    random_state: int = 42
) -> pd.DataFrame:
    """
    Balance dataset classes.
    
    Args:
        df: Input DataFrame
        label_col: Label column name
        method: 'undersample' or 'oversample'
        random_state: Random seed
        
    Returns:
        Balanced DataFrame
    """
    if label_col not in df.columns:
        return df
    
    # Count classes
    class_counts = df[label_col].value_counts()
    
    if method == 'undersample':
        # Downsample to minimum class size
        min_count = class_counts.min()
        
        balanced_dfs = []
        for label in class_counts.index:
            label_df = df[df[label_col] == label]
            sampled = label_df.sample(n=min_count, random_state=random_state)
            balanced_dfs.append(sampled)
        
        return pd.concat(balanced_dfs, ignore_index=True).sample(
            frac=1, random_state=random_state
        ).reset_index(drop=True)
    
    elif method == 'oversample':
        # Upsample to maximum class size
        max_count = class_counts.max()
        
        balanced_dfs = []
        for label in class_counts.index:
            label_df = df[df[label_col] == label]
            sampled = label_df.sample(n=max_count, replace=True, random_state=random_state)
            balanced_dfs.append(sampled)
        
        return pd.concat(balanced_dfs, ignore_index=True).sample(
            frac=1, random_state=random_state
        ).reset_index(drop=True)
    
    return df


def extract_session_sequences(
    df: pd.DataFrame,
    session_col: str = 'session_id',
    action_col: str = 'action',
    min_length: int = 2
) -> Dict[str, List[str]]:
    """
    Extract action sequences per session.
    
    Args:
        df: Input DataFrame
        session_col: Session identifier column
        action_col: Action column
        min_length: Minimum sequence length
        
    Returns:
        Dictionary mapping session_id to action sequence
    """
    sequences = {}
    
    for session_id in df[session_col].unique():
        if pd.isna(session_id):
            continue
        
        session_df = df[df[session_col] == session_id].sort_values('timestamp')
        actions = session_df[action_col].tolist()
        
        if len(actions) >= min_length:
            sequences[session_id] = actions
    
    return sequences


def compute_feature_statistics(
    df: pd.DataFrame,
    feature_cols: List[str]
) -> Dict[str, Dict]:
    """
    Compute statistics for feature columns.
    
    Args:
        df: Input DataFrame
        feature_cols: Feature column names
        
    Returns:
        Dictionary of statistics per feature
    """
    stats = {}
    
    for col in feature_cols:
        if col not in df.columns:
            continue
        
        if pd.api.types.is_numeric_dtype(df[col]):
            stats[col] = {
                'mean': float(df[col].mean()),
                'std': float(df[col].std()),
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'median': float(df[col].median()),
                'q25': float(df[col].quantile(0.25)),
                'q75': float(df[col].quantile(0.75)),
            }
        else:
            value_counts = df[col].value_counts()
            stats[col] = {
                'unique_values': int(df[col].nunique()),
                'top_values': value_counts.head(10).to_dict(),
                'mode': str(df[col].mode()[0]) if not df[col].mode().empty else None
            }
    
    return stats


def save_results_json(
    results: Dict,
    output_path: Path,
    pretty: bool = True
):
    """
    Save results to JSON file.
    
    Args:
        results: Results dictionary
        output_path: Output file path
        pretty: Pretty print JSON
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        if pretty:
            json.dump(results, f, indent=2, default=str)
        else:
            json.dump(results, f, default=str)


def load_results_json(input_path: Path) -> Dict:
    """
    Load results from JSON file.
    
    Args:
        input_path: Input file path
        
    Returns:
        Results dictionary
    """
    input_path = Path(input_path)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Results file not found: {input_path}")
    
    with open(input_path, 'r') as f:
        return json.load(f)


def merge_experiment_results(
    result_files: List[Path],
    output_path: Optional[Path] = None
) -> Dict:
    """
    Merge multiple experiment results.
    
    Args:
        result_files: List of result file paths
        output_path: Optional output path for merged results
        
    Returns:
        Merged results dictionary
    """
    merged = {}
    
    for result_file in result_files:
        result = load_results_json(result_file)
        experiment_name = result_file.stem
        merged[experiment_name] = result
    
    if output_path:
        save_results_json(merged, output_path)
    
    return merged


__all__ = [
    'normalize_dataframe',
    'split_by_agent',
    'compute_temporal_features',
    'encode_categorical',
    'balance_dataset',
    'extract_session_sequences',
    'compute_feature_statistics',
    'save_results_json',
    'load_results_json',
    'merge_experiment_results',
]
