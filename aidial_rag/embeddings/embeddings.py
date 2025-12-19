from typing import Iterable, Protocol, runtime_checkable

import numpy as np

from aidial_rag.content_stream import SupportsWriteStr


@runtime_checkable
class EmbeddingsModel(Protocol):
    """Interface for embeddings models."""

    async def aembed_query(self, text: str) -> np.ndarray: ...

    async def aembed_documents(
        self, texts: Iterable[str], progress_io: SupportsWriteStr
    ) -> Iterable[np.ndarray]: ...
