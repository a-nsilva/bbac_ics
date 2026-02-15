#!/usr/bin/env python3
"""
BBAC ICS Framework - Dataset Loader
Loads real dataset and converts to framework structures.
"""

import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from bbac_ics_core.utils.config_loader import ConfigLoader
from bbac_ics_core.utils.data_structures import (
    AccessRequest,
    AgentType,
    AgentRole,
    ActionType,
    AuthStatus,
    ResourceType,
)

logger = logging.getLogger(__name__)

class DataLoader:
    """Loads and manages the BBAC dataset with structure conversion."""
    
    def __init__(self, dataset_path: Optional[Path] = None):
        """Initialize dataset loader."""
        config = ConfigLoader.load()
        self.paths_config = config.get('paths', {})
        
        # Use provided path or default from config
        if dataset_path:
            self.dataset_path = Path(dataset_path)
        else:
            self.dataset_path = Path(
                self.paths_config.get('data_dir', 'data/raw')
            )
        
        # Initialize data containers
        self.train_data: Optional[pd.DataFrame] = None
        self.validation_data: Optional[pd.DataFrame] = None
        self.test_data: Optional[pd.DataFrame] = None
        
        # Validate dataset exists (only warns if missing to allow generation script to run)
        try:
            self._validate_dataset()
        except FileNotFoundError as e:
            logger.warning(f"Dataset check warning: {e}")

        logger.info(f"DatasetLoader initialized with path: {self.dataset_path}")
    
    def _validate_dataset(self):
        """Validate that required dataset files exist."""
        required_files = [
            self.paths_config.get('train_file', 'train.csv'),
            self.paths_config.get('validation_file', 'validation.csv'),
            self.paths_config.get('test_file', 'test.csv'),
        ]
        
        missing_files = []
        for filename in required_files:
            filepath = self.dataset_path / filename
            if not filepath.exists():
                missing_files.append(filename)
        
        if missing_files:
            error_msg = f"Missing required files in {self.dataset_path}: {missing_files}"
            raise FileNotFoundError(error_msg)
    
    def load_all(self) -> bool:
        """Load all dataset files."""
        try:
            logger.info(f"Loading dataset from {self.dataset_path}")
            
            # Load and normalize column names
            self.train_data = self._load_csv_file('train_file')
            self.validation_data = self._load_csv_file('validation_file')
            self.test_data = self._load_csv_file('test_file')
            
            logger.info("✓ All required dataset files loaded successfully")
            self._print_statistics()
            return True
            
        except Exception as e:
            logger.error(f"✗ Error loading dataset: {e}")
            return False
    
    def _load_csv_file(self, config_key: str) -> pd.DataFrame:
        """Load CSV file with proper parsing and renaming."""
        # Define default filenames if config is missing keys
        defaults = {
            'train_file': 'train.csv',
            'validation_file': 'validation.csv',
            'test_file': 'test.csv'
        }
        filename = self.paths_config.get(config_key, defaults.get(config_key))
        
        filepath = self.dataset_path / filename
        if not filepath.exists():
            logger.warning(f"File not found: {filepath}")
            return pd.DataFrame()
        
        df = pd.read_csv(filepath, low_memory=False)
        
        # Parse timestamps
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(
                df['timestamp'],
                format='mixed',
                errors='coerce'
            )
        
        # === CORREÇÃO 1: Mapeamento de Colunas ===
        # Garante que o código encontre as colunas esperadas
        column_mapping = {
            'user_id': 'agent_id',
            'resource_id': 'resource',
            'ground_truth': 'expected_decision',  # Mapeia CSV para padrão interno
            'label': 'expected_decision'
        }
        
        df = df.rename(columns={
            old: new for old, new in column_mapping.items()
            if old in df.columns
        })
        
        logger.debug(f"Loaded {len(df)} samples from {filename}")
        return df
    
    def to_access_request(self, row: pd.Series) -> AccessRequest:
        """Convert dataset row to AccessRequest structure."""
        
        # 1. Identificadores Básicos
        request_id = str(row.get('log_id', ''))
        
        # Timestamp
        timestamp_raw = row.get('timestamp')
        if pd.isna(timestamp_raw):
            timestamp = 0.0
        elif hasattr(timestamp_raw, 'timestamp'): # datetime object
            timestamp = timestamp_raw.timestamp()
        else:
            try:
                timestamp = float(timestamp_raw)
            except (ValueError, TypeError):
                timestamp = 0.0
        
        agent_id = str(row.get('agent_id', ''))
        
        # 2. Agent Type & Role
        agent_type_str = str(row.get('agent_type', 'robot')).strip()
        try:
            agent_type = AgentType.from_string(agent_type_str)
        except ValueError:
            agent_type = AgentType.ROBOT
        
        # Consolidate robot_type and human_role
        robot_type = row.get('robot_type', '')
        human_role = row.get('human_role', '')
        # Use o que não estiver vazio
        role_str = str(robot_type) if (robot_type and str(robot_type).strip()) else str(human_role)
        
        try:
            agent_role = AgentRole.from_string(role_str, agent_type)
        except (ValueError, AttributeError):
            agent_role = AgentRole.UNKNOWN
        
        # 3. Action & Resource
        action_str = str(row.get('action', 'read')).strip()
        try:
            action = ActionType.from_string(action_str)
        except ValueError:
            action = ActionType.READ
            
        resource = str(row.get('resource', ''))
        
        resource_type_str = str(row.get('resource_type', 'other')).strip()
        try:
            resource_type = ResourceType.from_string(resource_type_str)
        except ValueError:
            resource_type = ResourceType.OTHER
            
        # 4. Contexto Booleano (Tratamento robusto)
        def parse_bool(val):
            if pd.isna(val): return False
            if isinstance(val, bool): return val
            s = str(val).strip().lower()
            return s in ('true', '1', 'yes', 't')

        human_present = parse_bool(row.get('human_present'))
        emergency = parse_bool(row.get('emergency_flag'))
        location = str(row.get('location', 'unknown'))
        
        # 5. Session & Auth
        session_id_raw = row.get('session_id')
        session_id = str(session_id_raw) if not pd.isna(session_id_raw) else None
        
        auth_status_str = str(row.get('auth_status', 'success'))
        try:
            auth_status = AuthStatus.from_string(auth_status_str)
        except ValueError:
            auth_status = AuthStatus.SUCCESS

        # === CORREÇÃO 2: Previous Action ===
        # Trata NaN, 'nan', string vazia ou float nan explicitamente
        previous_action_raw = row.get('previous_action')
        previous_action = None
        
        if pd.isna(previous_action_raw):
            previous_action = None
        else:
            s_prev = str(previous_action_raw).strip()
            if s_prev.lower() not in ['', 'nan', 'none', 'null']:
                try:
                    previous_action = ActionType.from_string(s_prev)
                except ValueError:
                    previous_action = None

        # Meta info
        attempt_count = int(row.get('attempt_count', 0)) if not pd.isna(row.get('attempt_count')) else 0
        policy_id = str(row.get('policy_id')) if not pd.isna(row.get('policy_id')) else None

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

    def get_requests_from_dataframe(self, df: pd.DataFrame, max_requests: Optional[int] = None) -> List[AccessRequest]:
        """Convert DataFrame to list of AccessRequest structures."""
        if df is None or df.empty:
            return []
        
        if max_requests:
            df = df.head(max_requests)
        
        requests = []
        # iterrows é lento, mas seguro para estruturas complexas
        for _, row in df.iterrows():
            try:
                requests.append(self.to_access_request(row))
            except Exception as e:
                logger.error(f"Skipping row due to error: {e}")
                continue
                
        logger.info(f"Converted {len(requests)} rows to AccessRequest structures")
        return requests
    
    def _print_statistics(self):
        """Print dataset statistics."""
        if self.train_data is None: return
        
        stats = self.get_statistics()
        logger.info("=" * 50)
        logger.info(f"Training samples:    {stats['train_samples']:,}")
        logger.info(f"Validation samples:  {stats['validation_samples']:,}")
        logger.info(f"Test samples:        {stats['test_samples']:,}")
        logger.info("=" * 50)

    def get_statistics(self) -> Dict:
        """Get dataset statistics."""
        return {
            'train_samples': len(self.train_data) if self.train_data is not None else 0,
            'validation_samples': len(self.validation_data) if self.validation_data is not None else 0,
            'test_samples': len(self.test_data) if self.test_data is not None else 0,
        }

    def load_split(self, split: str) -> pd.DataFrame:
        """
        Load specific split (usado por scripts externos como publish_dataset).
        """
        # Garante que os dados foram carregados antes de pedir um split
        if self.train_data is None:
            self.load_all()

        if split == 'train':
            return self.train_data if self.train_data is not None else pd.DataFrame()
        elif split == 'validation':
            return self.validation_data if self.validation_data is not None else pd.DataFrame()
        elif split == 'test':
            return self.test_data if self.test_data is not None else pd.DataFrame()
        else:
            raise ValueError(f"Invalid split: {split}")
            
    def get_data_split(self, split: str = 'train') -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """Get features and labels for a specific split."""
        if split == 'train': data = self.train_data
        elif split == 'validation': data = self.validation_data
        elif split == 'test': data = self.test_data
        else: raise ValueError(f"Invalid split: {split}")
        
        if data is None:
            raise ValueError(f"{split} data not loaded")
        
        # Colunas de Rótulo (Labels) conhecidas
        # 'expected_decision' é o nome padrão após renomear 'ground_truth'
        label_columns = ['expected_decision', 'is_anomaly', 'anomaly_type']
        
        # Separa Features e Labels
        feature_cols = [c for c in data.columns if c not in label_columns]
        features = data[feature_cols].copy()
        
        present_labels = [c for c in label_columns if c in data.columns]
        labels = data[present_labels].copy() if present_labels else None
        
        return features, labels
