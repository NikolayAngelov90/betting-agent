"""Configuration management for Football Betting Agent."""

import yaml
from pathlib import Path
from typing import Any, Dict


class Config:
    """Configuration loader and accessor."""

    def __init__(self, config_path: str = "config/config.yaml"):
        """Load configuration from YAML file.

        Args:
            config_path: Path to configuration YAML file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file.

        Returns:
            Configuration dictionary
        """
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by dot-notation key.

        Args:
            key: Configuration key (e.g., 'database.type')
            default: Default value if key not found

        Returns:
            Configuration value

        Example:
            >>> config = Config()
            >>> config.get('betting.min_odds')
            1.3
        """
        keys = key.split('.')
        value = self.config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default

        return value

    @property
    def database(self) -> Dict[str, Any]:
        """Get database configuration."""
        return self.config.get('database', {})

    @property
    def scraping(self) -> Dict[str, Any]:
        """Get scraping configuration."""
        return self.config.get('scraping', {})

    @property
    def betting(self) -> Dict[str, Any]:
        """Get betting configuration."""
        return self.config.get('betting', {})

    @property
    def models(self) -> Dict[str, Any]:
        """Get models configuration."""
        return self.config.get('models', {})

    @property
    def notifications(self) -> Dict[str, Any]:
        """Get notifications configuration."""
        return self.config.get('notifications', {})

    @property
    def logging(self) -> Dict[str, Any]:
        """Get logging configuration."""
        return self.config.get('logging', {})


# Global config instance
_config = None


def get_config() -> Config:
    """Get global configuration instance.

    Returns:
        Config instance
    """
    global _config
    if _config is None:
        _config = Config()
    return _config
