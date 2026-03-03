"""Configuration management for Football Betting Agent."""

import os
import yaml
from pathlib import Path
from typing import Any, Dict

from src.utils.logger import get_logger

logger = get_logger()

# Environment variable overrides for sensitive / deployment-specific settings.
# Keys are dot-notation config paths, values are env var names.
_ENV_OVERRIDES = {
    "data_sources.apifootball_key": "API_FOOTBALL_KEY",
    "data_sources.footballdataorg_key": "FOOTBALL_DATA_ORG_KEY",
    "notifications.telegram_bot_token": "TELEGRAM_BOT_TOKEN",
    "notifications.telegram_chat_id": "TELEGRAM_CHAT_ID",
    "database.sqlite_path": "DB_PATH",
    "database.url": "DATABASE_URL",
    "logging.level": "LOG_LEVEL",
}


class Config:
    """Configuration loader and accessor."""

    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._apply_env_overrides()
        self._validate()

    def _load_config(self) -> Dict[str, Any]:
        if not self.config_path.exists():
            # Fall back to example config (CI environment or fresh checkout)
            example = self.config_path.parent / "config.example.yaml"
            if example.exists():
                logger.warning(
                    f"Config file not found at {self.config_path}, "
                    f"using {example} (secrets will be injected from env)"
                )
                with open(example, 'r') as f:
                    return yaml.safe_load(f)
            raise FileNotFoundError(
                f"Config file not found: {self.config_path}. "
                f"Copy config/config.example.yaml to config/config.yaml and fill in your secrets."
            )

        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)

    def _apply_env_overrides(self):
        """Override config values from environment variables when set."""
        for config_key, env_var in _ENV_OVERRIDES.items():
            env_val = os.environ.get(env_var)
            if env_val is not None:
                self._set(config_key, env_val)
                logger.debug(f"Config override from env: {config_key} = {env_var}")

    def _set(self, key: str, value: Any):
        """Set a config value by dot-notation key."""
        keys = key.split(".")
        d = self.config
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value

    def _validate(self):
        """Warn about missing critical configuration."""
        warnings = []
        if not self.get("data_sources.apifootball_key"):
            warnings.append("data_sources.apifootball_key not set (API-Football disabled)")
        if not self.get("scraping.flashscore_leagues"):
            warnings.append("scraping.flashscore_leagues is empty (no leagues configured)")
        for w in warnings:
            logger.warning(f"Config: {w}")

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by dot-notation key.

        Args:
            key: Configuration key (e.g., 'database.type')
            default: Default value if key not found

        Returns:
            Configuration value
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
        return self.config.get('database', {})

    @property
    def scraping(self) -> Dict[str, Any]:
        return self.config.get('scraping', {})

    @property
    def betting(self) -> Dict[str, Any]:
        return self.config.get('betting', {})

    @property
    def models(self) -> Dict[str, Any]:
        return self.config.get('models', {})

    @property
    def notifications(self) -> Dict[str, Any]:
        return self.config.get('notifications', {})

    @property
    def logging(self) -> Dict[str, Any]:
        return self.config.get('logging', {})


# Global config instance
_config = None


def get_config() -> Config:
    """Get global configuration instance."""
    global _config
    if _config is None:
        _config = Config()
    return _config
