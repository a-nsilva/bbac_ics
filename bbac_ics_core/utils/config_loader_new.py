

class ConfigLoader:

    _config: FullConfig = None

    @classmethod
    def load(cls, path: str = None) -> FullConfig:
        if cls._config:
            return cls._config

        config_path = Path(path or "config/params.yaml")

        with open(config_path, "r") as f:
            raw = yaml.safe_load(f)

        cls._config = cls._build_config(raw)
        return cls._config
