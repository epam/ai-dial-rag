from pydantic import Field

from aidial_rag.base_config import (
    BaseConfig,
    IndexRebuildTrigger,
    collect_fields_with_trigger,
)
from aidial_rag.document_loaders import ParserConfig
from aidial_rag.document_record import IndexSettings
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
