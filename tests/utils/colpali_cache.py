import hashlib
import pickle
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch import Tensor

from aidial_rag.retrievers.colpali_retriever.colpali_index_config import (
    ColpaliIndexConfig,
)
from aidial_rag.retrievers.colpali_retriever.colpali_model_resource import (
    ColpaliModelResource,
    ColpaliModelResourceConfig,
)


class CachedModel:
    """Combined recording/replay model based on hash of input"""

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

        # Hash-based caching - store individual tensors for each input
        self.query_embeddings_cache: Dict[str, Tensor] = {}
        self.image_embeddings_cache: Dict[str, Tensor] = {}
        # Load cached data if replaying
        if use_cache and cache_dir:
            self._load_cached_data()

    def _get_input_hash(self, kwargs: Dict[str, Any]) -> str:
        """Generate hash from input kwargs."""
        import hashlib

        # For images, hash the pixel_values
        if "pixel_values" in kwargs:
            pixel_values = kwargs["pixel_values"]
            hash_input = pixel_values.numpy().tobytes()
        # For queries, hash the input_ids
        elif "input_ids" in kwargs:
            # Hash the input_ids tensor
            input_ids = kwargs["input_ids"]
            hash_input = input_ids.numpy().tobytes()
        else:
            # Fallback: hash the entire kwargs
            raise ValueError("No input found in kwargs")

        hash_result = hashlib.md5(hash_input).hexdigest()  # noqa: S324
        return hash_result

    def _load_cached_data(self) -> None:
        if not self.cache_dir:
            return
        query_path = (
            self.cache_dir / f"{self.cache_key}_query_embeddings_cache.pkl"
        )
        image_path = (
            self.cache_dir / f"{self.cache_key}_image_embeddings_cache.pkl"
        )
        if query_path.exists():
            with open(query_path, "rb") as f:
                self.query_embeddings_cache = pickle.load(f)  # noqa: S301
        if image_path.exists():
            with open(image_path, "rb") as f:
                self.image_embeddings_cache = pickle.load(f)  # noqa: S301

    def _save_embeddings(
        self, embeddings_cache: Dict[str, Tensor], filename: str
    ) -> None:
        """Save embeddings cache to disk."""
        if self.cache_dir:
            cache_path = self.cache_dir / f"{self.cache_key}_{filename}.pkl"
            with open(cache_path, "wb") as f:
                pickle.dump(embeddings_cache, f)

    def __call__(self, **kwargs) -> Tensor:
        is_image_call = "pixel_values" in kwargs
        cache = (
            self.image_embeddings_cache
            if is_image_call
            else self.query_embeddings_cache
        )

        if self.use_cache:
            # Replay mode: split input, get individual embeddings, assemble batch
            batch_size = kwargs["input_ids"].shape[0]
            individual_embeddings = []

            for i in range(batch_size):
                # Split input into individual item
                individual_kwargs = {}
                for key, value in kwargs.items():
                    individual_kwargs[key] = value[i : i + 1]

                # Get individual embedding from cache
                input_hash = self._get_input_hash(individual_kwargs)
                if input_hash in cache:
                    cached_embedding = cache[input_hash]
                    individual_embeddings.append(cached_embedding)
                else:
                    raise RuntimeError(
                        f"No cached embedding found for {input_hash}"
                    )

            # Assemble batch from individual embeddings
            result = torch.cat(individual_embeddings, dim=0)
            return result
        else:
            # Recording mode: process whole batch, split result, cache individual items
            if self.real_model is None:
                raise RuntimeError("Real model is required for recording mode")
            output = self.real_model(**kwargs)  # Process whole batch

            # Split batch result and input, then cache individual mappings
            batch_size = output.shape[0]
            for i in range(batch_size):
                # Split input into individual item
                individual_kwargs = {}
                for key, value in kwargs.items():
                    individual_kwargs[key] = value[i : i + 1]

                # Cache individual mapping
                input_hash = self._get_input_hash(individual_kwargs)
                individual_tensor = output[i].unsqueeze(0).detach().cpu()
                cache[input_hash] = individual_tensor

            # Save cache
            if is_image_call:
                self._save_embeddings(
                    self.image_embeddings_cache, "image_embeddings_cache"
                )
            else:
                self._save_embeddings(
                    self.query_embeddings_cache, "query_embeddings_cache"
                )

            return output


class CachedColpaliModelResource(ColpaliModelResource):
    """Simplified cached version of ColpaliModelResource."""

    def __init__(
        self,
        colpali_model_resource_config: Optional[ColpaliModelResourceConfig],
        colpali_index_config: Optional[ColpaliIndexConfig],
        use_cache: bool = True,
        cache_dir: str = "tests/cache/test_colpali_retriever",
    ):
        super().__init__(colpali_model_resource_config, colpali_index_config)
        self.use_cache = use_cache
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_key = (
            self._get_cache_key() if colpali_model_resource_config else ""
        )

    def _get_cache_key(self) -> str:
        """Generate cache key for model configuration."""
        if self.model_resource_config is None:
            raise ValueError("Model resource config is required")
        content = f"{self.model_resource_config.model_name}_{self.model_resource_config.model_type}"
        return hashlib.sha256(content.encode()).hexdigest()

    def get_model_processor_device(self) -> tuple[Any, Any, torch.device]:
        """Get model, processor, and device with caching support."""
        if self.model_resource_config is None:
            raise ValueError("ColpaliModelResourceConfig is required")

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
