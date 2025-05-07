import asyncio
import logging
import os
import pickle
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing import get_context

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


class UnpicklableExceptionError(RuntimeError):
    pass


def _run_in_process_wrapper(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception as e:
        try:
            # python has an issue if unpicklable exception will be passed between processes
            # https://github.com/python/cpython/issues/120810
            # The exception created with kwargs could cause the issue
            pickle.loads(pickle.dumps(e))  # noqa: S301
        except Exception as pe:
            logger.exception(pe)
            # Unpicklable exception could break the process pool and cause the following error:
            # `concurrent.futures.process.BrokenProcessPool: A child process terminated abruptly, the process pool is not usable anymore`
            # To avoid this, we raise a custom exception with the original traceback in the __cause__ attribute
            raise UnpicklableExceptionError(
                "Unpicklable exception raised in subprocess"
            ) from e
        raise


class CpuPools:
    indexing_cpu_pool: ProcessPoolExecutor
    indexing_embeddings_pool: ThreadPoolExecutor
    query_embeddings_pool: ThreadPoolExecutor

    def __init__(self, config: CpuPoolsConfig) -> None:
        # Using process pool for indexing to avoid GIL limitations
        self.indexing_cpu_pool = ProcessPoolExecutor(
            max_workers=config.indexing_cpu_pool,
            # Spawn is used to avoid inheriting the file descriptors from the parent process
            mp_context=get_context("spawn"),
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
        return self._run_in_pool(
            self.indexing_cpu_pool,
            _run_in_process_wrapper,
            func,
            *args,
            **kwargs,
        )

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
