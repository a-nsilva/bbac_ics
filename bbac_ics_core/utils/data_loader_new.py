#!/usr/bin/env python3
"""
BBAC ICS Framework - Dataset Loader
Responsible ONLY for loading raw dataset splits.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import pandas as pd

from ..utils.data_structures import FullConfig, AccessRequest
from .request_mapper import RequestMapper

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Responsible ONLY for:
    - Validating dataset structure
    - Loading CSV splits
    - Returning raw DataFrames
    - Delegating domain conversion to RequestMapper
    """

    def __init__(
        self,
        config: FullConfig,
        dataset_path: Optional[Path] = None,
        mapper: Optional[RequestMapper] = None,
    ):
        self.config = config
        self.paths_config = config.paths

        self.dataset_path = (
            Path(dataset_path)
            if dataset_path is not None
            else Path(self.paths_config.data_dir)
        )

        self.mapper = mapper or RequestMapper()

        self.trainer_data: Optional[pd.DataFrame] = None
        self.validation_data: Optional[pd.DataFrame] = None
        self.test_data: Optional[pd.DataFrame] = None

        logger.info(f"DataLoader initialized with path: {self.dataset_path}")

    # ==========================================================
    # Dataset validation
    # ==========================================================

    def _validate_dataset(self):
        required_files = [
            self.paths_config.trainer_file,
            self.paths_config.validation_file,
            self.paths_config.test_file,
        ]

        missing_files = [
            f for f in required_files
            if not (self.dataset_path / f).exists()
        ]

        if missing_files:
            raise FileNotFoundError(
                f"Dataset incomplete at {self.dataset_path}. "
                f"Missing: {missing_files}"
            )

    # ==========================================================
    # Loading
    # ==========================================================

    def load_all(self) -> bool:
        try:
            logger.info(f"Loading dataset from {self.dataset_path}")

            self._validate_dataset()

            self.trainer_data = self._load_csv(self.paths_config.trainer_file)
            self.validation_data = self._load_csv(self.paths_config.validation_file)
            self.test_data = self._load_csv(self.paths_config.test_file)

            self._print_statistics()
            return True

        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            return False

    def _load_csv(self, filename: str) -> pd.DataFrame:
        filepath = self.dataset_path / filename

        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        df = pd.read_csv(filepath, low_memory=False)

        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(
                df["timestamp"],
                format="mixed",
                errors="coerce",
            )

        column_mapping = {
            "user_id": "agent_id",
            "resource_id": "resource",
        }

        df = df.rename(columns={
            old: new for old, new in column_mapping.items()
            if old in df.columns
        })

        logger.debug(f"Loaded {len(df)} rows from {filename}")
        return df

    # ==========================================================
    # Domain conversion (delegated)
    # ==========================================================

    def get_requests(
        self,
        split: str = "trainer",
        max_requests: Optional[int] = None,
    ) -> List[AccessRequest]:

        df = self.load_split(split)

        if df.empty:
            return []

        return self.mapper.dataframe_to_requests(df, max_requests)

    # ==========================================================
    # Accessors
    # ==========================================================

    def load_split(self, split: str) -> pd.DataFrame:
        if split == "trainer":
            return self.trainer_data if self.trainer_data is not None else pd.DataFrame()
        if split == "validation":
            return self.validation_data if self.validation_data is not None else pd.DataFrame()
        if split == "test":
            return self.test_data if self.test_data is not None else pd.DataFrame()

        raise ValueError(f"Invalid split: {split}")

    def get_data_split(
        self,
        split: str = "trainer"
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:

        data = self.load_split(split)

        if data.empty:
            raise ValueError(f"{split} data not loaded")

        label_columns = [
            "is_anomaly",
            "anomaly_type",
            "anomaly_severity",
            "expected_decision",
        ]

        feature_columns = [
            col for col in data.columns
            if col not in label_columns
        ]

        features = data[feature_columns].copy()

        label_cols_present = [
            col for col in label_columns if col in data.columns
        ]

        labels = data[label_cols_present].copy() if label_cols_present else None

        return features, labels

    # ==========================================================
    # Statistics
    # ==========================================================

    def _print_statistics(self):
        stats = self.get_statistics()

        logger.info("=" * 50)
        logger.info("DATASET STATISTICS")
        logger.info("=" * 50)
        logger.info(f"Training samples:   {stats['trainer_samples']:,}")
        logger.info(f"Validation samples: {stats['validation_samples']:,}")
        logger.info(f"Test samples:       {stats['test_samples']:,}")
        logger.info("=" * 50)

    def get_statistics(self) -> Dict:
        return {
            "trainer_samples": len(self.trainer_data) if self.trainer_data is not None else 0,
            "validation_samples": len(self.validation_data) if self.validation_data is not None else 0,
            "test_samples": len(self.test_data) if self.test_data is not None else 0,
            "dataset_path": str(self.dataset_path),
        }
