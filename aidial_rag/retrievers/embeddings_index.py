from typing import Iterable, List, Tuple

import numpy as np
import numpy.typing as npt
from docarray import DocList
from docarray.typing import NdArray
from langchain.schema import Document

from aidial_rag.document_record import Chunk, ItemEmbeddings, MultiEmbeddings
from aidial_rag.index_record import RetrievalType, to_metadata_doc
from aidial_rag.retrievers.embeddings_metrics import ENUM_TO_METRIC, Metric


class DocIndex:
    chunk_ids: npt.NDArray[np.int64]
    embeddings: np.ndarray

    def __init__(
        self,
        chunk_ids: npt.NDArray[np.int64] | None = None,
        embeddings: np.ndarray | None = None,
    ):
        self.chunk_ids = (
            chunk_ids if chunk_ids is not None else np.array([], dtype=np.int64)
        )
        self.embeddings = (
            embeddings
            if embeddings is not None
            else np.array([], dtype=np.float32)
        )


class EmbeddingsIndex:
    retrieval_type: RetrievalType
    doc_indexes: List[DocIndex]
    metric: str
    limit: int

    def __init__(
        self,
        retrieval_type: RetrievalType,
        indexes: List[DocIndex],
        metric: Metric = Metric.SQEUCLIDEAN_DIST,
        limit: int = 1,
    ):
        self.retrieval_type = retrieval_type
        self.metric = metric
        self.limit = limit
        self.doc_indexes = indexes

    def find_in_doc(
        self, query: np.ndarray, doc_index: DocIndex
    ) -> Tuple[npt.NDArray[np.int64], npt.NDArray[np.float32]]:
        metric_func = ENUM_TO_METRIC[Metric(self.metric)]
        distances = metric_func(query, doc_index.embeddings)

        # Do not use np.argpartition, because it is not stable
        top_indices = np.argsort(distances, kind="stable")[: self.limit]

        return doc_index.chunk_ids[top_indices], distances[top_indices]

    def find(self, query: np.ndarray) -> List[Document]:
        doc_ids = np.array([], dtype=np.int64)
        chunk_ids = np.array([], dtype=np.int64)
        distances = np.array([], dtype=np.float32)

        for i, doc_index in enumerate(self.doc_indexes):
            if len(doc_index.embeddings) == 0:
                continue
            top_chunk_ids, top_distances = self.find_in_doc(query, doc_index)

            doc_ids = np.concatenate(
                (
                    doc_ids,
                    np.full(len(top_chunk_ids), fill_value=i, dtype=np.int64),
                )
            )
            chunk_ids = np.concatenate((chunk_ids, top_chunk_ids))
            distances = np.concatenate((distances, top_distances))

        top_indices = np.argsort(distances, kind="stable")[: self.limit]
        return [
            to_metadata_doc(
                doc_id, chunk_id, retrieval_type=self.retrieval_type
            )
            for doc_id, chunk_id in zip(
                doc_ids[top_indices], chunk_ids[top_indices], strict=True
            )
        ]


def _get_page_index(chunk: Chunk) -> int:
    # Page numbers are 1-based
    return chunk.metadata["page_number"] - 1


def to_ndarray(arr: np.ndarray):
    return NdArray(shape=arr.shape, buffer=arr, dtype=arr.dtype)


def create_index_by_page(
    chunks: DocList[Chunk],
    pages_embeddings: MultiEmbeddings | None,
) -> DocIndex:
    if pages_embeddings is None:
        return DocIndex()

    chunk_ids = []
    embeddings = []
    for i, chunk in enumerate(chunks):
        page_embeddings = pages_embeddings[_get_page_index(chunk)].embeddings
        chunk_ids.extend([i] * len(page_embeddings))
        embeddings.extend(page_embeddings)

    return DocIndex(
        chunk_ids=np.array(chunk_ids, dtype=np.int64),
        embeddings=np.array(embeddings, dtype=np.float32),
    )


def create_index_by_chunk(
    chunks_embeddings: MultiEmbeddings | None,
) -> DocIndex:
    if chunks_embeddings is None:
        return DocIndex()

    chunk_ids = []
    embeddings = []
    for i, chunk_embeddings in enumerate(chunks_embeddings):
        chunk_ids.extend([i] * len(chunk_embeddings.embeddings))
        embeddings.extend(chunk_embeddings.embeddings)

    return DocIndex(
        chunk_ids=np.array(chunk_ids, dtype=np.int64),
        embeddings=np.array(embeddings),
    )


def pack_multi_embeddings(
    indexes: List[int], embeddings: Iterable[np.ndarray], number_of_pages: int
) -> MultiEmbeddings:
    page_embeddings = [[] for _ in range(number_of_pages)]
    for page_index, embedding in zip(indexes, embeddings, strict=True):
        page_embeddings[page_index].append(embedding)

    return MultiEmbeddings(
        [
            ItemEmbeddings(
                embeddings=to_ndarray(np.array(embeddings, dtype=np.float32))
            )
            for embeddings in page_embeddings
        ]
    )


def pack_simple_embeddings(embeddings: Iterable[np.ndarray]) -> MultiEmbeddings:
    return MultiEmbeddings(
        [
            ItemEmbeddings(
                embeddings=to_ndarray(np.array([embedding], dtype=np.float32))
            )
            for embedding in embeddings
        ]
    )
