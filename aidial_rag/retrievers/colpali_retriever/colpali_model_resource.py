import asyncio
import os
import threading
from typing import Annotated, DefaultDict, List, Optional, Tuple

import torch
from pydantic import BaseModel, Field, model_validator

from aidial_rag.embeddings.detect_device import autodetect_device
from aidial_rag.retrievers.colpali_retriever.colpali_index_config import (
    ColpaliIndexConfig,
)
from aidial_rag.retrievers.colpali_retriever.colpali_models import (
    KNOWN_MODELS,
    get_model_cache_path,
    get_model_local_path,
    get_model_processor_classes,
)

# Path to pre-downloaded ColPali models for normal use in docker
# if None model will be downloaded from Hugging Face
COLPALI_MODELS_BASE_PATH = os.environ.get("COLPALI_MODELS_BASE_PATH", None)


class ColpaliModelResourceConfig(BaseModel):
    model_name: Annotated[
        str,
        Field(
            default="vidore/colSmol-256M",
            description="Model name, should be one of KNOWN_MODELS keys",
        ),
    ]

    def validate_model_name(self):
        """Validate that model name is known"""
        if self.model_name not in KNOWN_MODELS:
            raise ValueError(
                f"Model name '{self.model_name}' is not known. Please use one of the following: {list(KNOWN_MODELS.keys())}"
            )

    @model_validator(mode="after")
    def validate_model_consistency(self):
        """Validate that model name is known."""
        self.validate_model_name()
        return self


class ColpaliBatchProcessor:
    """Handles batching of image processing requests across multiple concurrent tasks."""

    def __init__(
        self,
        process_batch_func,
        pool_func,
        batch_size: int = 8,
        batch_wait_time: float = 0.05,
    ):
        self.pending_items: DefaultDict[
            int, List[Tuple[str, asyncio.Future]]
        ] = DefaultDict(list)

        self.process_batch_func = process_batch_func
        self.pool_func = pool_func
        self.batch_size = batch_size
        self.batch_wait_time = batch_wait_time
        self._lock = asyncio.Lock()

        self.processing_task: Optional[asyncio.Task] = (
            None  # current processing task
        )
        self.processing_queue = []  # queue of ids to process
        self.ids_to_reuse = []  # ids to reuse for new tasks
        self.items_count = 0  # total items in queue

        # Validate pool function
        if pool_func is not None and not callable(pool_func):
            raise ValueError("pool_func must be callable")

    async def register_processing_id(self) -> int:
        """Return id to process requests to create fair queue."""
        async with self._lock:
            id = None
            if self.ids_to_reuse:
                id = self.ids_to_reuse.pop()
            else:
                id = len(self.processing_queue)
                self.processing_queue.append(id)

            return id

    async def unregister_processing_id(self, id: int):
        """Unregister processing id."""
        async with self._lock:
            self.ids_to_reuse.append(id)

    async def add_item(self, item: str, id: int) -> asyncio.Future:
        """Add item to batch, return future for the result."""
        future = asyncio.Future()

        async with self._lock:
            self.pending_items[id].append((item, future))
            self.items_count += 1

            # Start processing task if not already running
            if self.processing_task is None or self.processing_task.done():
                self.processing_task = asyncio.create_task(
                    self._process_batches()
                )

        return future

    async def _process_batches(self):
        """Process batches. Iterates over queue and get 1 item if possible from id then put id in the end of the queue to make fair queue."""
        while True:
            batch_items = []
            should_wait = False

            async with self._lock:
                if self.items_count < self.batch_size:
                    should_wait = True
                elif self.items_count == 0:
                    break  # No more items, exit processing

            # Wait to collect more items
            if should_wait:
                await asyncio.sleep(self.batch_wait_time)

            async with self._lock:
                if self.items_count > 0:
                    batch_size = min(self.items_count, self.batch_size)
                    while len(batch_items) < batch_size:
                        # taking from the front of the queue
                        id = self.processing_queue.pop(0)
                        if len(self.pending_items[id]) > 0:
                            batch_items.append(self.pending_items[id].pop(0))
                            self.items_count -= 1

                        # add back to queue and move to the next
                        self.processing_queue.append(id)

            # Only process if we have items
            if batch_items:
                await self._process_batch(batch_items)
            else:
                break

    async def _process_batch(
        self, batch_items: List[Tuple[str, asyncio.Future]]
    ):
        """Process a batch of images."""
        try:
            # Extract images from batch
            images = [item[0] for item in batch_items]
            futures = [item[1] for item in batch_items]

            batch_results = await self.pool_func(
                self.process_batch_func, images
            )

            # Distribute results back to futures
            for future, result in zip(futures, batch_results, strict=False):
                if not future.done():
                    future.set_result(result)

        except Exception as e:
            # Set exception for all futures in batch
            for _, future in batch_items:
                if not future.done():
                    future.set_exception(e)


class ColpaliModelResource:
    def __init__(
        self,
        config: ColpaliModelResourceConfig | None,
        colpali_index_config: ColpaliIndexConfig | None,
    ):
        self.lock = threading.Lock()
        self.model_resource_config: ColpaliModelResourceConfig | None = None
        self.colpali_index_config: ColpaliIndexConfig | None = None
        self.index_config: ColpaliIndexConfig | None = None
        self.model = None
        self.device: torch.device | None = None
        self.processor = None
        self.batch_processor: Optional[ColpaliBatchProcessor] = None
        self.query_batch_processor: Optional[ColpaliBatchProcessor] = None
        if colpali_index_config is not None and config is not None:
            self.__set_config(config)

    def __set_config(self, config: ColpaliModelResourceConfig):
        config.validate_model_name()

        with self.lock:
            if self.model_resource_config == config:
                return
            self.model_resource_config = config
            device = autodetect_device().value
            self.device = torch.device(device)

            model_class, processor_class = get_model_processor_classes(
                config.model_name
            )

            # Check if local model path exists otherwise use Hugging Face
            model_name = config.model_name
            cache_path = None
            if COLPALI_MODELS_BASE_PATH:
                local_model_path = get_model_local_path(
                    COLPALI_MODELS_BASE_PATH, model_name
                )
                if os.path.exists(local_model_path):
                    model_name = local_model_path
                    cache_path = get_model_cache_path(local_model_path)
            self.model = model_class.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map=self.device,
                cache_dir=cache_path if cache_path else None,
            ).eval()
            self.processor = processor_class.from_pretrained(model_name)

            assert self.model is not None
            assert self.processor is not None
            assert self.device is not None

    def get_model_processor_device(self):
        with self.lock:
            if (
                self.model_resource_config is None
                or self.device is None
                or self.model is None
                or self.processor is None
            ):
                raise ValueError("ColpaliModelResourceConfig is required")
            return self.model, self.processor, self.device

    def get_batch_processor(
        self,
        process_batch_func,
        pool_func,
        batch_size: int = 8,
        batch_wait_time: float = 0.05,
    ) -> ColpaliBatchProcessor:
        """Get or create the indexing batch processor instance with the given processing function.

        Args:
            process_batch_func: Function to process batches
            pool_func: Pool function to use
            batch_size: Number of items to process in each batch
            batch_wait_time: Time to wait for more items before processing batch
        """
        if self.batch_processor is None:
            self.batch_processor = ColpaliBatchProcessor(
                process_batch_func, pool_func, batch_size, batch_wait_time
            )
        return self.batch_processor

    def get_query_batch_processor(
        self,
        process_batch_func,
        pool_func,
        batch_size: int = 8,
        batch_wait_time: float = 0.05,
    ) -> ColpaliBatchProcessor:
        """Get or create the query batch processor instance with the given processing function.

        Args:
            process_batch_func: Function to process batches
            pool_func: pool function to use
            batch_size: Number of items to process in each batch
            batch_wait_time: Time to wait for more items before processing batch
        """
        if self.query_batch_processor is None:
            self.query_batch_processor = ColpaliBatchProcessor(
                process_batch_func, pool_func, batch_size, batch_wait_time
            )
        return self.query_batch_processor
