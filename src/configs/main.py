from pydantic import DirectoryPath, Field
from pydantic_settings import SettingsConfigDict

from .database_conf import DatabaseConfig
from .toimage_conf import ToImageConfig


class InjectionConfig(ToImageConfig, DatabaseConfig):
    """Injected configuration for the application."""

    OUTPUT_DIRECTORY: DirectoryPath = Field(description="Path to the output directory")

    model_config = SettingsConfigDict(env_file=".config.cfg", env_file_encoding="utf-8", extra="ignore", frozen=True)
