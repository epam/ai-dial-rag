import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor

from pydantic import Field

from aidial_rag.base_config import BaseConfig

CPU_COUNT = os.cpu_count() or 1
DEFAULT_CPU_POOL_WORKERS: int = max(1, CPU_COUNT - 2)


logger = logging.getLogger(__name__)


class CpuPoolsConfig(BaseConfig):
    """Configuration for CPU pools."""

    indexing_cpu_pool: int = Field(
        default=DEFAULT_CPU_POOL_WORKERS,
        description="Process pool for document parsing, image extraction and similar CPU-bound tasks. "
        "Is set to `max(1, CPU_COUNT - 2)` to leave some CPU cores for other tasks.",
    )
    indexing_embeddings_pool: int = Field(
        default=1,
        description="Embedding process itself uses multiple cores. "
        "Should be `1`, unless you have a lot of cores and can explicitly see "
        "the underutilisation (i.e. you only have a very small documents in the requests).",
    )
    query_embeddings_pool: int = Field(
        default=1,
        description="Embedding process for the query. Should be `1`, unless you have a lot of cores.",
    )


class CpuPools:
    indexing_cpu_pool: ThreadPoolExecutor
    indexing_embeddings_pool: ThreadPoolExecutor
    query_embeddings_pool: ThreadPoolExecutor

    def __init__(self, config: CpuPoolsConfig) -> None:
        # Using ThreadPoolExecutor instead of ProcessPoolExecutor, because
        # ProcessPoolExecutor can stuck with zombie processes
        self.indexing_cpu_pool = ThreadPoolExecutor(
            max_workers=config.indexing_cpu_pool,
            thread_name_prefix="indexing_cpu",
        )

        self.indexing_embeddings_pool = ThreadPoolExecutor(
            max_workers=config.indexing_embeddings_pool,
            thread_name_prefix="indexing_embeddings",
        )

        # TODO: Do we need a separate pool for query embeddings?
        self.query_embeddings_pool = ThreadPoolExecutor(
            max_workers=config.query_embeddings_pool,
            thread_name_prefix="query_embeddings",
        )

    def _run_in_pool(self, pool, func, *args, **kwargs):
        return asyncio.get_running_loop().run_in_executor(
            pool, func, *args, **kwargs
        )

    def run_in_indexing_cpu_pool(self, func, *args, **kwargs):
        return self._run_in_pool(self.indexing_cpu_pool, func, *args, **kwargs)

    def run_in_indexing_embeddings_pool(self, func, *args, **kwargs):
        return self._run_in_pool(
            self.indexing_embeddings_pool, func, *args, **kwargs
        )

    def run_in_query_embeddings_pool(self, func, *args, **kwargs):
        return self._run_in_pool(
            self.query_embeddings_pool, func, *args, **kwargs
        )

    _instance = None

    @classmethod
    def instance(cls) -> "CpuPools":
        if cls._instance is None:
            logger.warning(
                "CpuPools instance is not initialized. Initializing with default config."
            )
            cls.init_cpu_pools(CpuPoolsConfig())
        assert isinstance(cls._instance, cls)
        return cls._instance

    @classmethod
    def init_cpu_pools(cls, config: CpuPoolsConfig):
        if cls._instance is not None:
            raise RuntimeError("CpuPools instance already initialized.")
        cls._instance = cls(config)
        return cls._instance


async def init_cpu_pools(config: CpuPoolsConfig):
    """Init and warm up the pools to avoid the first call overhead"""

    cpu_pools = CpuPools.init_cpu_pools(config)
    await cpu_pools.run_in_indexing_cpu_pool(sum, range(10))
    await cpu_pools.run_in_indexing_embeddings_pool(sum, range(10))
    await cpu_pools.run_in_query_embeddings_pool(sum, range(10))


def run_in_indexing_cpu_pool(func, *args, **kwargs):
    return CpuPools.instance().run_in_indexing_cpu_pool(func, *args, **kwargs)


def run_in_indexing_embeddings_pool(func, *args, **kwargs):
    return CpuPools.instance().run_in_indexing_embeddings_pool(
        func, *args, **kwargs
    )


def run_in_query_embeddings_pool(func, *args, **kwargs):
    return CpuPools.instance().run_in_query_embeddings_pool(
        func, *args, **kwargs
    )
