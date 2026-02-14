#!/usr/bin/env python3
"""
BBAC_ICS - Configuration Loader

Centralized configuration management using YAML files.
Loads and provides access to all framework configuration.
"""

import yaml
from pathlib import Path
from typing import Any, Dict

from ..utils.data_structures import *  # Importa todas as dataclasses


class ConfigLoader:
    """Singleton loader for YAML configuration."""

    _config: Dict[str, Any] = None

    @classmethod
    def load(cls, path: str = None) -> Dict[str, Any]:
        if cls._config:
            return cls._config 

        config_path = Path(path or "config/params.yaml")
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r") as f:
            cls._config = yaml.safe_load(f)

        return cls._config

    @classmethod
    def get(cls, key: str, default=None):
        if not cls._config:
            cls.load()
        return cls._config.get(key, default)
