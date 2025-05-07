from enum import StrEnum

import numpy as np
import torch


class Metric(StrEnum):
    COSINE_SIM = "cosine_sim"
    EUCLIDEAN_DIST = "euclidean_dist"
    SQEUCLIDEAN_DIST = "sqeuclidean_dist"
    INNER_PRODUCT = "inner_product"


def _metric_for_inner_product(
    query: np.ndarray, docs: np.ndarray
) -> np.ndarray:
    """Compute inner product between a query and a set of documents.
    This is used for ranking, so we want to return the negative value.
    """
    return -np.inner(query, docs)


def _metric_for_cosine_sim(query: np.ndarray, docs: np.ndarray) -> np.ndarray:
    """Compute negative cosine similarity between a query and a set of documents.
    This is used for ranking, so we want to return the negative value.
    """
    # Use torch to avoid handling special cases like zero vectors
    return -torch.nn.functional.cosine_similarity(
        torch.from_numpy(docs),
        torch.from_numpy(query),
    ).numpy()


def _metric_for_sqeuclidean_dist(
    query: np.ndarray, docs: np.ndarray
) -> np.ndarray:
    """Compute squared Euclidean distance between a query and a set of documents."""

    # We do not use np.sum((docs - query) ** 2, axis=1), because it is less precise
    doc_sq = np.sum(docs**2, axis=1)
    query_sq = np.sum(query**2)
    query_dot = np.dot(docs, query)
    return doc_sq - 2 * query_dot + query_sq


def _metric_for_euclidean_dist(
    query: np.ndarray, docs: np.ndarray
) -> np.ndarray:
    """Compute Euclidean distance between a query and a set of documents."""
    return np.sqrt(_metric_for_sqeuclidean_dist(query, docs))


ENUM_TO_METRIC = {
    Metric.COSINE_SIM: _metric_for_cosine_sim,
    Metric.EUCLIDEAN_DIST: _metric_for_euclidean_dist,
    Metric.SQEUCLIDEAN_DIST: _metric_for_sqeuclidean_dist,
    Metric.INNER_PRODUCT: _metric_for_inner_product,
}

assert len(ENUM_TO_METRIC) == len(Metric), (
    "All metrics should be defined in ENUM_TO_METRIC"
)
