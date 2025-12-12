from typing import Iterable

import numpy as np

from aidial_rag.content_stream import SupportsWriteStr


class EmbeddingsModel:
    """Interface for embeddings models."""

    async def aembed_query(self, text: str) -> np.ndarray:
        raise NotImplementedError()

    async def aembed_documents(
        self, texts: Iterable[str], progress_io: SupportsWriteStr
    ) -> Iterable[np.ndarray]:
        raise NotImplementedError()
