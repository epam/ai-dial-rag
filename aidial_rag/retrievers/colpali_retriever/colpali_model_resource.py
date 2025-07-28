import asyncio
import os
import threading
from enum import StrEnum
from typing import Annotated, Any, List, Optional, Tuple

import torch
from pydantic import BaseModel, Field, model_validator

from aidial_rag.embeddings.detect_device import autodetect_device
from aidial_rag.retrievers.colpali_retriever.colpali_index_config import (
    ColpaliIndexConfig,
)

# Path to pre-downloaded ColPali models for normal use in docker
# Model names are used for local runs only
COLPALI_MODELS_BASE_PATH = os.environ.get("COLPALI_MODELS_BASE_PATH", None)


class ColpaliModelType(StrEnum):
    COLPALI = "ColPali"
    COLQWEN = "ColQwen"
    COLIDEFICS = "ColIdefics"


# Mapping of known model names to their expected model types
# can be extended with more models if needed
KNOWN_MODELS = {
    # ColIdefics models
    "vidore/colSmol-256M": ColpaliModelType.COLIDEFICS,
    "vidore/colpali-v1.3": ColpaliModelType.COLPALI,
    "vidore/colqwen2-v1.0": ColpaliModelType.COLQWEN,
}


def get_model_processor_classes(
    model_type: ColpaliModelType,
) -> tuple[Any, Any]:
    """Get model and processor classes for a given model type"""
    from colpali_engine.models import (
        ColIdefics3,
        ColIdefics3Processor,
        ColPali,
        ColPaliProcessor,
        ColQwen2,
        ColQwen2Processor,
    )

    match model_type:
        case ColpaliModelType.COLPALI:
            return ColPali, ColPaliProcessor
        case ColpaliModelType.COLIDEFICS:
            return ColIdefics3, ColIdefics3Processor
        case ColpaliModelType.COLQWEN:
            return ColQwen2, ColQwen2Processor
        case _:
            raise ValueError("Invalid ColPali model type")


def get_safe_model_name(model_name: str) -> str:
    """Convert model name to safe directory name"""
    return model_name.replace("/", "_")


def get_model_local_path(base_path: str, model_name: str) -> str:
    """Get the local path for a model given base path and model name"""
    safe_name = get_safe_model_name(model_name)
    return f"{base_path}/{safe_name}"


class ColpaliModelResourceConfig(BaseModel):
    model_name: Annotated[
        str,
        Field(
            default="vidore/colSmol-256M",
            description="Model name, should be one of KNOWN_MODELS keys",
        ),
    ]
    model_type: Annotated[
        ColpaliModelType,
        Field(
            default=ColpaliModelType.COLIDEFICS,
            description="Type of ColPali model",
        ),
    ]

    def validate_consistency(self):
        """validation of model name and type consistency"""
        if self.model_name in KNOWN_MODELS:
            expected_type = KNOWN_MODELS[self.model_name]
            if self.model_type != expected_type:
                raise ValueError(
                    f"Model name '{self.model_name}' is known to be of type '{expected_type}', "
                    f"but '{self.model_type}' was specified. Please use the correct model type."
                )
        else:
            raise ValueError(
                f"Model name '{self.model_name}' is not known. Please use one of the following: {list(KNOWN_MODELS.keys())}"
            )

    @model_validator(mode="after")
    def validate_model_consistency(self):
        """Validate that model name and type are consistent."""
        self.validate_consistency()
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
        self.pending_items: List[
            Tuple[str, asyncio.Future]
        ] = []  # (image, future)
        self.processing_task: Optional[asyncio.Task] = None
        self.process_batch_func = process_batch_func
        self.pool_func = pool_func
        self.batch_size = batch_size
        self.batch_wait_time = batch_wait_time
        self._lock = asyncio.Lock()

        # Validate pool function
        if pool_func is not None and not callable(pool_func):
            raise ValueError("pool_func must be callable")

    async def add_item(self, item: str) -> asyncio.Future:
        """Add item to batch, return future for the result."""
        future = asyncio.Future()

        async with self._lock:
            self.pending_items.append((item, future))

            # Start processing task if not already running
            if self.processing_task is None or self.processing_task.done():
                self.processing_task = asyncio.create_task(
                    self._process_batches()
                )

        return future

    async def _process_batches(self):
        """Process batches"""
        while True:
            batch_items = []
            should_wait = False

            async with self._lock:
                if len(self.pending_items) < self.batch_size:
                    should_wait = True
                elif len(self.pending_items) == 0:
                    break  # No more items, exit processing

            # Wait to collect more items
            if should_wait:
                await asyncio.sleep(self.batch_wait_time)
            async with self._lock:
                if self.pending_items:
                    batch_size = min(len(self.pending_items), self.batch_size)
                    batch_items = self.pending_items[:batch_size]
                    self.pending_items = self.pending_items[batch_size:]

            # Only process if we have items
            if batch_items:
                await self._process_batch(batch_items)

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
        config.validate_consistency()

        with self.lock:
            if self.model_resource_config == config:
                return
            self.model_resource_config = config
            device = autodetect_device()
            self.device = torch.device(device)

            model_class, processor_class = get_model_processor_classes(
                config.model_type
            )

            # Check if local model path exists otherwise use Hugging Face
            model_name = config.model_name
            if COLPALI_MODELS_BASE_PATH:
                local_model_path = get_model_local_path(
                    COLPALI_MODELS_BASE_PATH, model_name
                )
                if os.path.exists(local_model_path):
                    model_name = local_model_path
            self.model = model_class.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map=self.device,
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
