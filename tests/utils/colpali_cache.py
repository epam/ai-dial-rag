import hashlib
import pickle
from pathlib import Path
from typing import Any, List, Optional

import torch
from torch import Tensor

from aidial_rag.retrievers.colpali_retriever.colpali_index_config import (
    ColpaliIndexConfig,
)
from aidial_rag.retrievers.colpali_retriever.colpali_model_resource import (
    ColpaliModelResource,
)


class CachedModel:
    """Combined recording/replay model."""

    def __init__(
        self,
        real_model: Optional[Any] = None,
        cache_key: str = "",
        cache_dir: Optional[Path] = None,
        use_cache: bool = False,
    ):
        self.real_model = real_model
        self.cache_key = cache_key
        self.cache_dir = cache_dir
        self.use_cache = use_cache

        # Recording state
        self.recorded_query_embeddings: List[Tensor] = []
        self.recorded_image_embeddings: List[Tensor] = []
        self.query_call_count = 0
        self.image_call_count = 0

        # Load cached data if replaying
        if use_cache and cache_dir:
            self._load_cached_data()

    def _load_cached_data(self) -> None:
        if not self.cache_dir:
            return
        query_path = self.cache_dir / f"{self.cache_key}_query_embeddings.pkl"
        image_path = self.cache_dir / f"{self.cache_key}_image_embeddings.pkl"
        if query_path.exists():
            with open(query_path, "rb") as f:
                query_data = pickle.load(f)  # noqa: S301
                # Handle both old dict format and new list format
                if isinstance(query_data, dict):
                    self.recorded_query_embeddings = query_data.get(
                        "query_embeddings", []
                    )
                else:
                    self.recorded_query_embeddings = query_data
        if image_path.exists():
            with open(image_path, "rb") as f:
                image_data = pickle.load(f)  # noqa: S301
                # Handle both old dict format and new list format
                if isinstance(image_data, dict):
                    self.recorded_image_embeddings = image_data.get(
                        "image_embeddings", []
                    )
                else:
                    self.recorded_image_embeddings = image_data

    def _save_embeddings(self, embeddings: List[Tensor], filename: str) -> None:
        """Save embeddings to cache."""
        if self.cache_dir:
            cache_path = self.cache_dir / f"{self.cache_key}_{filename}.pkl"
            with open(cache_path, "wb") as f:
                pickle.dump(embeddings, f)

    def __call__(self, **kwargs) -> Tensor:
        is_image_call = "pixel_values" in kwargs

        if self.use_cache:
            # Replay mode
            embeddings = (
                self.recorded_image_embeddings
                if is_image_call
                else self.recorded_query_embeddings
            )
            call_count = (
                self.image_call_count
                if is_image_call
                else self.query_call_count
            )

            if call_count >= len(embeddings):
                call_count = 0
                if is_image_call:
                    self.image_call_count = 0
                else:
                    self.query_call_count = 0

            output = embeddings[call_count]

            # Update counter
            if is_image_call:
                self.image_call_count += 1
            else:
                self.query_call_count += 1

            return output
        else:
            # Recording mode
            if self.real_model is None:
                raise RuntimeError("Real model is required for recording mode")

            output = self.real_model(**kwargs)

            if is_image_call:
                self.recorded_image_embeddings.append(output.detach().cpu())
                self._save_embeddings(
                    self.recorded_image_embeddings, "image_embeddings"
                )
            else:
                self.recorded_query_embeddings.append(output.detach().cpu())
                self._save_embeddings(
                    self.recorded_query_embeddings, "query_embeddings"
                )

            return output


class CachedColpaliModelResource(ColpaliModelResource):
    """Simplified cached version of ColpaliModelResource."""

    def __init__(
        self,
        colpali_config: Optional[ColpaliIndexConfig],
        use_cache: bool = True,
        cache_dir: str = "tests/cache/test_colpali_retriever",
    ):
        super().__init__(colpali_config)
        self.use_cache = use_cache
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_key = self._get_cache_key() if colpali_config else ""

    def _get_cache_key(self) -> str:
        """Generate cache key for model configuration."""
        if self.config is None:
            raise ValueError("Config is required")
        content = f"{self.config.model_name}_{self.config.model_type}"
        return hashlib.sha256(content.encode()).hexdigest()

    def get_model_processor_device(self) -> tuple[Any, Any, torch.device]:
        """Get model, processor, and device with caching support."""
        if self.config is None:
            raise ValueError("ColpaliIndexConfig is required")

        if self.use_cache:
            # Replay mode
            model = CachedModel(
                cache_key=self.cache_key,
                cache_dir=self.cache_dir,
                use_cache=True,
            )
            # Use the real processor directly
            _, real_processor, _ = super().get_model_processor_device()
            processor = real_processor
            device = torch.device("cpu")
        else:
            # Recording mode
            real_model, real_processor, device = (
                super().get_model_processor_device()
            )
            model = CachedModel(
                real_model, self.cache_key, self.cache_dir, use_cache=False
            )
            processor = real_processor

        return model, processor, device
