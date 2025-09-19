"""
ColPali model definitions and utilities.
This module contains the model mappings and utilities that can be imported
without requiring the full aidial_rag package.
"""

from enum import StrEnum
from pathlib import Path
from typing import Any, Optional

import torch
from colpali_engine.models import (
    ColIdefics3,
    ColIdefics3Processor,
    ColPali,
    ColPaliProcessor,
    ColQwen2,
    ColQwen2Processor,
)


class ColpaliModelType(StrEnum):
    COLPALI = "ColPali"
    COLQWEN = "ColQwen"
    COLIDEFICS = "ColIdefics"


class KnownModels(StrEnum):
    """Known ColPali model names"""

    COLSMOL_256M = "vidore/colSmol-256M"
    COLPALI_V1_3 = "vidore/colpali-v1.3"
    COLQWEN2_V1_0 = "vidore/colqwen2-v1.0"


# Mapping of known model names to their expected model types
# can be extended with more models if needed
MODEL_NAME_TO_TYPE = {
    # ColIdefics models
    KnownModels.COLSMOL_256M: ColpaliModelType.COLIDEFICS,
    KnownModels.COLPALI_V1_3: ColpaliModelType.COLPALI,
    KnownModels.COLQWEN2_V1_0: ColpaliModelType.COLQWEN,
}


def get_model_processor_classes(
    model_name: str,
) -> tuple[Any, Any]:
    """Get model and processor classes by a given model name"""

    if model_name not in MODEL_NAME_TO_TYPE:
        raise ValueError(f"Unknown model name: {model_name}")

    model_type = MODEL_NAME_TO_TYPE[KnownModels(model_name)]

    match model_type:
        case ColpaliModelType.COLPALI:
            return ColPali, ColPaliProcessor
        case ColpaliModelType.COLIDEFICS:
            return ColIdefics3, ColIdefics3Processor
        case ColpaliModelType.COLQWEN:
            return ColQwen2, ColQwen2Processor
        case _:
            raise ValueError("Invalid ColPali model type")


def get_model_local_path(base_path: str, model_name: str) -> Path:
    """Get the local path for a model given base path and model name"""
    return Path(base_path) / Path(model_name)


def get_model_cache_path(model_path: Path) -> Path:
    """Get the cache path for a model given model path"""
    return model_path / "cache"


def load_model_and_processor(
    model_name: str, model_path: Optional[str], device: torch.device
) -> tuple[Any, Any]:
    """Load model and processor for a given model name"""
    model_class, processor_class = get_model_processor_classes(model_name)

    cache_path = None
    print(f"model_path: {model_path}")
    # if model_path is set, load model from local path
    if model_path:
        local_model_path = get_model_local_path(model_path, model_name)
        if local_model_path.exists():
            model_name = str(local_model_path)
            cache_path = get_model_cache_path(local_model_path)
            print(f"loading model from local path: {local_model_path}")
        else:
            raise ValueError(
                f"Model {model_name} not found in local path {local_model_path}"
            )
    model = model_class.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device,
        cache_dir=cache_path
        if cache_path
        else None,  # cache containt base models weights
        local_files_only=model_path
        is not None,  # if set use only local files from folder
    ).eval()
    processor = processor_class.from_pretrained(model_name)

    return model, processor
