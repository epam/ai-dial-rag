from pathlib import Path

from pydantic import Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)
from pydantic_settings.sources import PathType, YamlConfigSettingsSource

from aidial_rag.base_config import (
    BaseConfig,
    IndexRebuildTrigger,
    collect_fields_with_trigger,
)
from aidial_rag.document_loaders import (
    HttpClientConfig,
    ParserConfig,
)
from aidial_rag.document_record import IndexSettings
from aidial_rag.index_storage import IndexStorageConfig
from aidial_rag.llm import LlmConfig
from aidial_rag.qa_chain import ChatChainConfig, QAChainConfig
from aidial_rag.query_chain import QueryChainConfig
from aidial_rag.resources.cpu_pools import CpuPoolsConfig
from aidial_rag.retrievers.colpali_retriever.colpali_index_config import (
    ColpaliIndexConfig,
)
from aidial_rag.retrievers.description_retriever.description_retriever import (
    DescriptionIndexConfig,
)
from aidial_rag.retrievers.multimodal_retriever import MultimodalIndexConfig


class IndexingConfig(BaseConfig):
    """Configuration for the document indexing."""

    # pyright does not understand default values for Annotated fields
    parser: ParserConfig = Field(default=ParserConfig())  # type: ignore

    multimodal_index: MultimodalIndexConfig | None = Field(
        default=None,
        description="Enables MultimodalRetriever which uses multimodal embedding models for pages "
        "images search.",
    )
    description_index: DescriptionIndexConfig | None = Field(
        default=DescriptionIndexConfig(),
        description="Enables DescriptionRetriever which uses vision model to generate page images "
        "descriptions and perform search on them.",
    )
    colpali_index: ColpaliIndexConfig | None = Field(
        default=None, description="Enables ColpaliRetriever"
    )

    def collect_fields_that_rebuild_index(self) -> IndexSettings:
        """Return the IndexingConfig fields that determine when the index needs to be rebuilt."""
        indexes = {}
        for name, _field_info in self.__class__.model_fields.items():
            index_config = getattr(self, name)
            if index_config is not None:
                indexes[name] = collect_fields_with_trigger(
                    index_config, IndexRebuildTrigger
                )

        return IndexSettings(indexes=indexes)


class RequestConfig(BaseConfig):
    """Configuration for the request process.
    These config fields could be changed for particular request.
    Changes in the indexing fields will require index rebuilding.
    """

    ignore_document_loading_errors: bool = Field(
        default=False,
        description="Ignore errors during document loading. "
        "Used for Web RAG for the request with multiple documents.",
    )

    use_profiler: bool = Field(
        default=False,
        description="Use profiler to collect performance metrics for the request.",
    )

    download: HttpClientConfig = Field(
        default=HttpClientConfig(),
        description="Configuration for downloading the attached documents.",
    )

    check_access: HttpClientConfig = Field(
        default=HttpClientConfig(),
        description="Configuration for checking access to the documents in the Dial.",
    )

    indexing: IndexingConfig = Field(default=IndexingConfig())

    qa_chain: QAChainConfig = Field(
        default=QAChainConfig(
            chat_chain=ChatChainConfig(
                llm=LlmConfig(
                    deployment_name="gpt-4o-2024-05-13",
                    max_prompt_tokens=16000,
                ),
            ),
            query_chain=QueryChainConfig(
                llm=LlmConfig(
                    deployment_name="gpt-4o-2024-05-13",
                    max_prompt_tokens=8000,
                ),
            ),
        ),
    )


# TODO: Add support for legacy env variables names


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
