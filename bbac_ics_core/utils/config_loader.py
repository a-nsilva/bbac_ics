#!/usr/bin/env python3
"""
BBAC_ICS - Configuration Loader

Centralized configuration management using YAML files.
Loads and provides access to all framework configuration.
"""

import yaml
from pathlib import Path
from typing import Any, Dict

from ament_index_python.packages import get_package_share_directory


class ConfigLoader:
    """Singleton loader for YAML configuration."""

    _config: Dict[str, Any] = None

    @classmethod
    def load(cls, path: str = None) -> Dict[str, Any]:
        if cls._config:
            return cls._config 

        # Tenta achar o arquivo na pasta de instalação do ROS (share directory)
        # Isso garante que funcione rodando via 'ros2 launch'
        if path is None:
            try:
                pkg_path = get_package_share_directory('bbac_ics')
                config_path = Path(pkg_path) / "config" / "params.yaml"
            except Exception:
                # Fallback para desenvolvimento local (se rodar o script direto)
                config_path = Path("config/params.yaml")
        else:
            config_path = Path(path)

        if not config_path.exists():
            # Tenta procurar relativo ao workspace se falhar
            config_path = Path("src/bbac_ics/config/params.yaml")
            if not config_path.exists():
                raise FileNotFoundError(f"Config file not found at: {config_path}")

        with open(config_path, "r") as f:
            raw_config = yaml.safe_load(f)

        # --- Lógica de Compatibilidade ROS 2 ---
        # Se o YAML tiver a estrutura 'ros__parameters', extraímos o miolo
        # para que o resto do código continue funcionando como antes.
        if 'bbac_main_node' in raw_config and 'ros__parameters' in raw_config['bbac_main_node']:
            cls._config = raw_config['bbac_main_node']['ros__parameters']
            # Adiciona parâmetros de outros nós se necessário (merge)
            if 'experiment_evaluator_node' in raw_config:
                 eval_params = raw_config['experiment_evaluator_node'].get('ros__parameters', {})
                 cls._config.update(eval_params)
        else:
            cls._config = raw_config

        return cls._config

    @classmethod
    def get(cls, key: str, default=None):
        if cls._config is None:
            cls.load()
        return cls._config.get(key, default)
