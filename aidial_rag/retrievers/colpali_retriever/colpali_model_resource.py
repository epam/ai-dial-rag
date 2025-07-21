import threading
from enum import StrEnum
from typing import Annotated

import torch
from pydantic import BaseModel, Field, model_validator

from aidial_rag.embeddings.detect_device import autodetect_device
from aidial_rag.retrievers.colpali_retriever.colpali_index_config import (
    ColpaliIndexConfig,
)


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


class ColpaliModelResource:
    def __init__(
        self,
        config: ColpaliModelResourceConfig | None,
        colpali_index_config: ColpaliIndexConfig | None,
    ):
        self.lock = threading.Lock()
        self.gpu_lock = threading.Lock()
        self.model_resource_config: ColpaliModelResourceConfig | None = None
        self.colpali_index_config: ColpaliIndexConfig | None = None
        self.index_config: ColpaliIndexConfig | None = None
        self.model = None
        self.device: torch.device | None = None
        self.processor = None
        if colpali_index_config is not None and config is not None:
            self.__set_config(config)

    def get_gpu_lock(self):
        """Get the thread lock specifically for GPU operations."""
        return self.gpu_lock

    def __set_config(self, config: ColpaliModelResourceConfig):
        config.validate_consistency()

        with self.lock:
            if self.model_resource_config == config:
                return
            self.model_resource_config = config
            device = autodetect_device()
            self.device = torch.device(device)

            from colpali_engine.models import (
                ColIdefics3,
                ColIdefics3Processor,
                ColPali,
                ColPaliProcessor,
                ColQwen2,
                ColQwen2Processor,
            )

            match config.model_type:
                case ColpaliModelType.COLPALI:
                    model_class = ColPali
                    processor_class = ColPaliProcessor
                case ColpaliModelType.COLIDEFICS:
                    model_class = ColIdefics3
                    processor_class = ColIdefics3Processor
                case ColpaliModelType.COLQWEN:
                    model_class = ColQwen2
                    processor_class = ColQwen2Processor
                case _:
                    raise ValueError("Invalid ColPali model type")

            self.model = model_class.from_pretrained(
                config.model_name, torch_dtype=torch.float16, device_map=device
            ).eval()
            self.processor = processor_class.from_pretrained(config.model_name)
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
