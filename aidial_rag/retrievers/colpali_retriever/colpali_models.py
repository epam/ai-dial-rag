"""
ColPali model definitions and utilities.
This module contains the model mappings and utilities that can be imported
without requiring the full aidial_rag package.
"""

import os
from enum import StrEnum
from pathlib import Path
from typing import Any


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


def get_model_processor_classes(model_type: ColpaliModelType) -> tuple[Any, Any]:
    """Get model and processor classes for a given model type"""
    from colpali_engine.models import (
        ColIdefics3, ColIdefics3Processor,
        ColPali, ColPaliProcessor,
        ColQwen2, ColQwen2Processor,
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


def get_model_local_path(base_path: str, model_name: str) -> Path:
    """Get the local path for a model given base path and model name"""
    safe_name = get_safe_model_name(model_name)
    return Path(base_path) / safe_name

def get_model_cache_path(model_path: Path) -> Path:
    """Get the cache path for a model given model path"""
    return model_path / "cache"

# Path to pre-downloaded ColPali models for normal use in docker
# Model names are used for local runs only
COLPALI_MODELS_BASE_PATH = os.environ.get("COLPALI_MODELS_BASE_PATH", None) 