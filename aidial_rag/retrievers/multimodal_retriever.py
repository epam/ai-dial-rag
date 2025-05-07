import asyncio
import logging
import sys
from typing import Annotated, List

import numpy as np
from langchain.schema import BaseRetriever, Document
from pydantic import Field

from aidial_rag.base_config import BaseConfig, IndexRebuildTrigger
from aidial_rag.content_stream import SupportsWriteStr
from aidial_rag.dial_config import DialConfig
from aidial_rag.document_record import DocumentRecord, MultiEmbeddings
from aidial_rag.embeddings.multimodal_embeddings import MultimodalEmbeddings
from aidial_rag.index_record import RetrievalType
from aidial_rag.resources.dial_limited_resources import (
    DialLimitedResources,
    map_with_resource_limits,
)
from aidial_rag.retrievers.embeddings_index import (
    EmbeddingsIndex,
    Metric,
    create_index_by_page,
    pack_simple_embeddings,
)
from aidial_rag.retrievers.page_image_retriever_utils import extract_page_images
from aidial_rag.utils import timed_block

# Error message in the openai library tells to use math.inf, but the type for the max_retries is int
MAX_RETRIES = 1_000_000_000  # One billion retries should be enough


logger = logging.getLogger(__name__)


class MultimodalIndexConfig(BaseConfig):
    """Configuration for the multimodal index."""

    embeddings_model: Annotated[
        str,
        IndexRebuildTrigger(),
        Field(
            default="multimodalembedding@001",
            description=(
                "The name of the multimodal embeddings model to use. "
                "The model should support both text and images. "
                "The change of the model will require index rebuilding. "
                "Example of supported models: "
                "`multimodalembedding@001`, "
                "`azure-ai-vision-embeddings`, "
                "`amazon.titan-embed-image-v1`."
            ),
        ),
    ]
    metric: Metric = Field(
        default=Metric.SQEUCLIDEAN_DIST,
        description=(
            "Metric to use for the embeddings search. "
            "Possible values: `sqeuclidean_dist` or `cosine_sim`. "
            "Use `sqeuclidean_dist` for `multimodalembedding@001` and `cosine_sim` "
            "for the `azure-ai-vision-embeddings` and `azure-ai-vision-embeddings`."
        ),
    )
    image_size: int = Field(
        default=1536,
        description="Sets the size of the page images that will be used for the multimodal embeddings model.",
    )
    estimated_task_tokens: int = Field(
        default=500,
        description=(
            "The number of the `embeddings_model` tokens to be consumed by calculating embeddings "
            "of a single page image. This value is used to calculate the number of model requests "
            "to run in parallel based on the user's minute token limit. "
            "Should be 500 for `multimodalembedding@001`, 1 for the `azure-ai-vision-embeddings` "
            "and 75 for `amazon.titan-embed-image-v1`."
        ),
    )
    time_limit_multiplier: float = Field(
        default=1.5,
        description=(
            "The multiplier allows to set the ratio between the estimated time for the image embeddings "
            "calculations and the timeout used for this operation."
        ),
    )
    min_time_limit_sec: float = Field(
        default=5 * 60,
        description=(
            "The minimal time limit for the timeout for the image embeddings calculations. "
            "Since the minute token limit is used in the Dial, the estimated time could have a relatively "
            "high margin of error for the small number of minutes. "
            "So, 5 minutes minimal time limit is recommended."
        ),
    )


class MultimodalRetriever(BaseRetriever):
    index: EmbeddingsIndex
    dial_config: DialConfig
    index_config: MultimodalIndexConfig

    @staticmethod
    def has_index(document_records: List[DocumentRecord]) -> bool:
        return any(
            doc.multimodal_embeddings_index is not None
            for doc in document_records
        )

    @classmethod
    def from_doc_records(
        cls,
        dial_config: DialConfig,
        index_config: MultimodalIndexConfig,
        document_records: List[DocumentRecord],
        k: int = 1,
    ) -> "MultimodalRetriever":
        # multimodal_embeddings_index is just a list of embeddings by the page number
        # we need to convert it to index for chunks
        indexes = [
            create_index_by_page(doc.chunks, doc.multimodal_embeddings_index)
            for doc in document_records
        ]

        return cls(
            index=EmbeddingsIndex(
                retrieval_type=RetrievalType.IMAGE,
                indexes=indexes,
                metric=index_config.metric,
                limit=k,
            ),
            dial_config=dial_config,
            index_config=index_config,
        )

    def _find_relevant_documents(self, query_emb: np.ndarray) -> List[Document]:
        return self.index.find(query=query_emb)

    def _get_relevant_documents(
        self, query: str, *args, **kwargs
    ) -> List[Document]:
        multimodal_embeddings = MultimodalEmbeddings(
            self.dial_config, self.index_config.embeddings_model
        )
        query_emb = np.array(multimodal_embeddings.embed_query(query))
        return self._find_relevant_documents(query_emb)

    async def _aget_relevant_documents(self, query: str, *args, **kwargs):
        multimodal_embeddings = MultimodalEmbeddings(
            self.dial_config, self.index_config.embeddings_model
        )
        query_emb = np.array(await multimodal_embeddings.aembed_query(query))
        return await asyncio.get_running_loop().run_in_executor(
            None, self._find_relevant_documents, query_emb
        )

    @staticmethod
    async def build_index(
        dial_config: DialConfig,
        dial_limited_resources: DialLimitedResources,
        index_config: MultimodalIndexConfig,
        mime_type: str,
        original_document: bytes,
        stageio: SupportsWriteStr = sys.stderr,
    ) -> MultiEmbeddings | None:
        async with timed_block("Building Multimodal indexes", stageio):
            logger.debug("Building Multimodal indexes.")

            multimodal_embeddings = MultimodalEmbeddings(
                dial_config,
                index_config.embeddings_model,
                max_retries=MAX_RETRIES,
            )

            extract_pages_kwargs = {"scaled_size": index_config.image_size}

            extracted_images = await extract_page_images(
                mime_type,
                original_document,
                extract_pages_kwargs,
                stageio,
            )
            if extracted_images is None:
                return

            stageio.write("Building image embeddings\n")
            embeddings = await map_with_resource_limits(
                dial_limited_resources,
                extracted_images,
                multimodal_embeddings.aembed_image,
                index_config.estimated_task_tokens,
                index_config.embeddings_model,
                stageio,
                index_config.time_limit_multiplier,
                index_config.min_time_limit_sec,
            )

            return pack_simple_embeddings(embeddings)
