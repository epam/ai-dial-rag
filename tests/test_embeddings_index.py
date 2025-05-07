import numpy as np
import pytest

from aidial_rag.index_record import RetrievalType, to_metadata_doc
from aidial_rag.retrievers.embeddings_index import (
    DocIndex,
    EmbeddingsIndex,
    Metric,
)

DOC1 = DocIndex(
    chunk_ids=np.array([0, 1], dtype=np.int64),
    embeddings=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
)
DOC2 = DocIndex(
    chunk_ids=np.array([0], dtype=np.int64),
    embeddings=np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
)
DOC3 = DocIndex(
    chunk_ids=np.array([], dtype=np.int64),
    embeddings=np.array([], dtype=np.float32),
)


@pytest.mark.parametrize("metric", list(Metric))
def test_search_stability(metric):
    docs = [DOC1, DOC2, DOC3]

    index = EmbeddingsIndex(
        retrieval_type=RetrievalType.TEXT,
        indexes=docs,
        metric=metric,
        limit=1,
    )
    result = index.find(np.array([1.0, 0.0, 0.0]))
    assert result == [
        to_metadata_doc(0, 0, RetrievalType.TEXT),
    ]

    index_reversed = EmbeddingsIndex(
        retrieval_type=RetrievalType.TEXT,
        indexes=docs[::-1],
        metric=metric,
        limit=1,
    )
    result_reversed = index_reversed.find(np.array([1.0, 0.0, 0.0]))
    assert result_reversed == [
        to_metadata_doc(1, 0, RetrievalType.TEXT),
    ]


@pytest.mark.parametrize("metric", list(Metric))
@pytest.mark.parametrize("limit", [1, 2, 3, 10])
def test_different_limits(metric, limit):
    docs = [DOC1, DOC2, DOC3]

    index = EmbeddingsIndex(
        retrieval_type=RetrievalType.TEXT,
        indexes=docs,
        metric=metric,
        limit=limit,
    )
    result = index.find(np.array([1.0, 0.0, 0.0]))
    expected = [
        to_metadata_doc(0, 0, RetrievalType.TEXT),
        to_metadata_doc(1, 0, RetrievalType.TEXT),
        to_metadata_doc(0, 1, RetrievalType.TEXT),
    ][:limit]
    assert result == expected


@pytest.mark.parametrize("metric", list(Metric))
def test_empty_index(metric):
    query = np.array([0.0, 0.0, 0.0])

    assert (
        EmbeddingsIndex(
            retrieval_type=RetrievalType.TEXT,
            indexes=[],
            metric=metric,
            limit=1,
        ).find(query)
        == []
    )

    assert (
        EmbeddingsIndex(
            retrieval_type=RetrievalType.TEXT,
            indexes=[DOC3],
            metric=metric,
            limit=1,
        ).find(query)
        == []
    )
