import logging
import os
from functools import cache
from typing import Iterable, List

import numpy as np
from langchain.schema.embeddings import Embeddings

# HuggingFaceBgeEmbeddings is marked as deprecated in langchain_community.embeddings
# but langchain_huggingface does not provide BGE specific embeddings implementation
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

from aidial_rag.batched import batched_map_with_progress
from aidial_rag.content_stream import SupportsWriteStr
from aidial_rag.embeddings.detect_device import DeviceType, detect_device
from aidial_rag.resources.cpu_pools import (
    run_in_indexing_embeddings_pool,
    run_in_query_embeddings_pool,
)

logger = logging.getLogger(__name__)


# 32 is the default batch size for BGE model in SentenceTransformers
# but 128 works faster on CPU with openvino backend
EMBEDDINGS_BATCH_SIZE = 128

# Path to pre-downloaded epam/bge-small-en model for normal use in docker
# epam/bge-small-en model name is used for the local runs only
BGE_EMBEDDINGS_MODEL_NAME_OR_PATH = os.environ.get(
    "BGE_EMBEDDINGS_MODEL_PATH", "epam/bge-small-en"
)

BGE_EMBEDDINGS_DEVICE = detect_device(
    os.environ.get("BGE_EMBEDDINGS_DEVICE", DeviceType.AUTO)
)

MODEL_KWARGS_BY_DEVICE = {
    DeviceType.CPU: {
        "device": "cpu",
        "backend": "openvino",
    },
    DeviceType.CUDA: {
        "device": "cuda",
        "backend": "torch",
        "torch_dtype": "float16",
        "attn_implementation": "sdpa",
    },
}


@cache
def bge_embedding_impl() -> HuggingFaceBgeEmbeddings:
    device = BGE_EMBEDDINGS_DEVICE
    logger.info(f"BGE embeddings device: {device}")

    bge_embedding_impl = HuggingFaceBgeEmbeddings(
        model_name=BGE_EMBEDDINGS_MODEL_NAME_OR_PATH,
        model_kwargs=MODEL_KWARGS_BY_DEVICE[device],
        encode_kwargs={
            "normalize_embeddings": True,
        },
        show_progress=True,
    )
    bge_embedding_impl.client.compile()
    return bge_embedding_impl


EMBEDDING_LENGTH = len(bge_embedding_impl().embed_query(""))


class AsyncEmbeddings(Embeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError()

    def embed_query(self, text: str) -> List[float]:
        raise NotImplementedError()

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        return await run_in_indexing_embeddings_pool(
            bge_embedding_impl().embed_documents, texts
        )

    async def aembed_documents_numpy(
        self, texts: List[str]
    ) -> List[np.ndarray]:
        # TODO : Use sentence-transformers directly to avoid List[float] <-> np.ndarray conversions
        embeddings = await self.aembed_documents(texts)
        return [
            np.array(embedding, dtype=np.float32) for embedding in embeddings
        ]

    async def aembed_query(self, text: str) -> List[float]:
        return await run_in_query_embeddings_pool(
            bge_embedding_impl().embed_query, text
        )


bge_embedding = AsyncEmbeddings()


async def build_embeddings(texts: Iterable[str], stageio: SupportsWriteStr):
    return await batched_map_with_progress(
        texts,
        bge_embedding.aembed_documents_numpy,
        EMBEDDINGS_BATCH_SIZE,
        file=stageio,
    )
