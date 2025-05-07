import numpy as np

from aidial_rag.retrievers.embeddings_metrics import ENUM_TO_METRIC, Metric


def test_cosine_similarity():
    cosine_sim_func = ENUM_TO_METRIC[Metric.COSINE_SIM]
    np.testing.assert_allclose(
        cosine_sim_func(
            np.array([1.0, 0.0, 0.0, 0.0]),
            np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]),
        ),
        np.array([-1.0, 0.0]),
    )
    np.testing.assert_allclose(
        cosine_sim_func(
            np.array([-1.0, 0.0, 0.0, 0.0]),
            np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]),
        ),
        np.array([1.0, 0.0]),
    )

    # non-normalized query
    np.testing.assert_allclose(
        cosine_sim_func(
            np.array([2.0, 0.0, 0.0, 0.0]),
            np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]),
        ),
        np.array([-1.0, 0.0]),
    )

    # Zero vector is considered orthogonal to all vectors, do not fail with division by zero here
    np.testing.assert_allclose(
        cosine_sim_func(
            np.array([0.0, 0.0, 0.0, 0.0]),
            np.array([[1.0, 0.0, 0.0, 0.0], [0, 1, 0, 0], [0, 0, 0, 0]]),
        ),
        np.array([0.0, 0.0, 0.0]),
    )

    # non-normalized docs
    np.testing.assert_allclose(
        cosine_sim_func(
            np.array([1.0, 0.0, 0.0, 0.0]),
            np.array(
                [
                    [2.0, 0.0, 0.0, 0.0],
                    [0.0, 2.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ]
            ),
        ),
        np.array([-1.0, 0.0, 0.0]),
    )


def test_inner_product():
    inner_product_func = ENUM_TO_METRIC[Metric.INNER_PRODUCT]
    np.testing.assert_allclose(
        inner_product_func(
            np.array([1, 0, 0, 0]), np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        ),
        np.array([-1.0, 0.0]),
    )

    np.testing.assert_allclose(
        inner_product_func(
            np.array([-1, 0, 0, 0]), np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        ),
        np.array([1.0, 0.0]),
    )

    # inner product makes sense only for normalized vectors, but still should not fail with division by zero
    np.testing.assert_allclose(
        inner_product_func(
            np.array([2, 0, 0, 0]), np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        ),
        np.array([-2.0, 0.0]),
    )
    np.testing.assert_allclose(
        inner_product_func(
            np.array([0, 0, 0, 0]),
            np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0]]),
        ),
        np.array([0.0, 0.0, 0.0]),
    )
    np.testing.assert_allclose(
        inner_product_func(
            np.array([1, 0, 0, 0]),
            np.array([[2, 0, 0, 0], [0, 2, 0, 0], [0, 0, 0, 0]]),
        ),
        np.array([-2.0, 0.0, 0.0]),
    )


def normalize(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / norm


def test_cosine_to_inner_equivalence():
    # Check that cosine similarity is equivalent to inner product for normalized vectors
    cosine_sim_func = ENUM_TO_METRIC[Metric.COSINE_SIM]
    inner_product_func = ENUM_TO_METRIC[Metric.INNER_PRODUCT]

    query = normalize(np.array([1, 2, 3, 4]))
    docs = normalize(
        np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [2, 0, 0, 0],
                [3, 3, 3, 0],
                [0, 0, 0, 0],
            ]
        )
    )

    np.testing.assert_allclose(
        cosine_sim_func(query, docs), inner_product_func(query, docs)
    )


def test_euclidean_distance():
    euclidean_dist_func = ENUM_TO_METRIC[Metric.EUCLIDEAN_DIST]
    np.testing.assert_allclose(
        euclidean_dist_func(
            np.array([1, 0, 0, 0]), np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        ),
        np.array([0.0, np.sqrt(2)]),
    )
    np.testing.assert_allclose(
        euclidean_dist_func(
            np.array([-1, 0, 0, 0]), np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        ),
        np.array([2.0, np.sqrt(2)]),
    )
    np.testing.assert_allclose(
        euclidean_dist_func(
            np.array([2, 0, 0, 0]), np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        ),
        np.array([1.0, np.sqrt(4 + 1)]),
    )
    np.testing.assert_allclose(
        euclidean_dist_func(
            np.array([1, 0, 0, 0]),
            np.array([[2, 0, 0, 0], [3, 3, 3, 0], [0, 0, 0, 0]]),
        ),
        np.array([1.0, np.sqrt(4 + 9 + 9), 1.0]),
    )


def test_squared_euclidean_distance():
    sqeuclidean_dist_func = ENUM_TO_METRIC[Metric.SQEUCLIDEAN_DIST]
    np.testing.assert_allclose(
        sqeuclidean_dist_func(
            np.array([1, 0, 0, 0]), np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        ),
        np.array([0.0, 2.0]),
    )
    np.testing.assert_allclose(
        sqeuclidean_dist_func(
            np.array([-1, 0, 0, 0]), np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        ),
        np.array([4.0, 2.0]),
    )
    np.testing.assert_allclose(
        sqeuclidean_dist_func(
            np.array([2, 0, 0, 0]), np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        ),
        np.array([1.0, (4 + 1)]),
    )
    np.testing.assert_allclose(
        sqeuclidean_dist_func(
            np.array([1, 0, 0, 0]),
            np.array([[2, 0, 0, 0], [3, 3, 3, 0], [0, 0, 0, 0]]),
        ),
        np.array([1.0, (4 + 9 + 9), 1]),
    )
    np.testing.assert_allclose(
        sqeuclidean_dist_func(
            np.array([0, 0, 0, 0]), np.array([[1, 1, 1, 1], [2, 2, 2, 2]])
        ),
        np.array([4.0, 16.0]),
    )


def test_euclidian_equivalence():
    # Check that euclidean distance is equivalent to squared euclidean distance
    euclidean_dist_func = ENUM_TO_METRIC[Metric.EUCLIDEAN_DIST]
    sqeuclidean_dist_func = ENUM_TO_METRIC[Metric.SQEUCLIDEAN_DIST]

    query = np.array([1, 2, 3, 4])
    docs = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [2, 0, 0, 0], [3, 3, 3, 0], [0, 0, 0, 0]]
    )

    np.testing.assert_allclose(
        euclidean_dist_func(query, docs) ** 2,
        sqeuclidean_dist_func(query, docs),
    )
