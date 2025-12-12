import os
from functools import cache
from typing import Any, Dict, Iterable, List

import numpy as np
from sentence_transformers import SentenceTransformer

from aidial_rag.base_config import BaseConfig
from aidial_rag.batched import batched_map_with_progress
from aidial_rag.content_stream import SupportsWriteStr
from aidial_rag.embeddings.detect_device import DeviceType, detect_device
from aidial_rag.embeddings.embeddings import EmbeddingsModel
from aidial_rag.resources.cpu_pools import run_in_embeddings_pool

# 32 is the default batch size for BGE model in SentenceTransformers
# but 128 works faster on CPU with openvino backend
EMBEDDINGS_BATCH_SIZE = 128

# Path to pre-downloaded epam/bge-small-en model for normal use in docker
# epam/bge-small-en model name is used for the local runs only
BGE_EMBEDDINGS_MODEL_NAME_OR_PATH = os.environ.get(
    "BGE_EMBEDDINGS_MODEL_PATH", "epam/bge-small-en"
)

DEFAULT_QUERY_BGE_INSTRUCTION = (
    "Represent this question for searching relevant passages: "
)

BGE_EMBEDDINGS_DEVICE = detect_device(
    os.environ.get("BGE_EMBEDDINGS_DEVICE", DeviceType.AUTO)
)

MODEL_KWARGS_BY_DEVICE: Dict[str, Any] = {
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


class LocalEmbeddingsConfig(BaseConfig):
    model_name_or_path: str = BGE_EMBEDDINGS_MODEL_NAME_OR_PATH
    device: DeviceType = BGE_EMBEDDINGS_DEVICE
    query_instruction: str = DEFAULT_QUERY_BGE_INSTRUCTION
    document_instruction: str = ""
    batch_size: int = EMBEDDINGS_BATCH_SIZE


class LocalEmbeddingsModel(EmbeddingsModel):
    def __init__(self, config: LocalEmbeddingsConfig) -> None:
        model_kwargs = MODEL_KWARGS_BY_DEVICE[config.device]
        self._config = config
        self._model = SentenceTransformer(
            config.model_name_or_path,
            **model_kwargs,
        )

    def _embed_query(self, text: str) -> np.ndarray:
        if self._config.query_instruction:
            text = self._config.query_instruction + text
        return self._model.encode(
            [text],
            show_progress_bar=False,
        )[0]

    def _embed_documents_batch(self, texts: List[str]) -> np.ndarray:
        if self._config.document_instruction:
            texts = [self._config.document_instruction + text for text in texts]
        return self._model.encode(
            texts,
            batch_size=EMBEDDINGS_BATCH_SIZE,
            show_progress_bar=False,
        )

    async def aembed_query(self, text: str) -> np.ndarray:
        return await run_in_embeddings_pool(self._embed_query, text)

    async def _aembed_documents_batch(self, texts: List[str]) -> np.ndarray:
        return await run_in_embeddings_pool(self._embed_documents_batch, texts)

    async def aembed_documents(
        self, texts: Iterable[str], progress_io: SupportsWriteStr
    ) -> Iterable[np.ndarray]:
        return await batched_map_with_progress(
            texts,
            self._aembed_documents_batch,
            EMBEDDINGS_BATCH_SIZE,
            file=progress_io,
        )


@cache
def create_local_bge_embeddings_model() -> EmbeddingsModel:
    return LocalEmbeddingsModel(
        LocalEmbeddingsConfig(
            model_name_or_path=BGE_EMBEDDINGS_MODEL_NAME_OR_PATH,
            device=BGE_EMBEDDINGS_DEVICE,
        )
    )
