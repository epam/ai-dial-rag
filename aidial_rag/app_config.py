from pathlib import Path

from pydantic import Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)
from pydantic_settings.sources import PathType, YamlConfigSettingsSource

from aidial_rag.configuration_endpoint import RequestConfig
from aidial_rag.index_storage import IndexStorageConfig
from aidial_rag.resources.cpu_pools import CpuPoolsConfig


class AppConfig(BaseSettings):
    """Application configuration."""

    config_path: PathType = Field(
        default=Path(""),
        description="Path to the yaml configuration file."
        "See config directory for examples.",
    )

    dial_url: str = Field(
        default="http://dial-proxy.dial-proxy",
        validation_alias="dial_url",
        description="Url to the dial core.",
    )

    enable_debug_commands: bool = Field(
        default=False,
        validation_alias="enable_debug_commands",
        description="Enables support of debug commands in the messages. "
        "Should be `false` for prod envs. It is set to `true` only for staging. "
        "See [Debug commands](README.md#debug-commands) for more details.",
    )

    cpu_pools: CpuPoolsConfig = Field(default=CpuPoolsConfig())
    index_storage: IndexStorageConfig = Field(default=IndexStorageConfig())
    request: RequestConfig = Field(default=RequestConfig())

    model_config = SettingsConfigDict(
        env_prefix="DIAL_RAG__",
        env_nested_delimiter="__",
        nested_model_default_partial_update=True,
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        config_path = (
            init_settings().get("config_path")
            or env_settings().get("config_path")
            or ""
        )

        return (
            init_settings,
            env_settings,
            YamlConfigSettingsSource(settings_cls, yaml_file=config_path),
        )
