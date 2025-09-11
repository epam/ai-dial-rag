from typing import Annotated

import torch
from pydantic import BaseModel, Field, model_validator

from aidial_rag.embeddings.detect_device import autodetect_device
from aidial_rag.retrievers.colpali_retriever.colpali_index_config import (
    ColpaliIndexConfig,
)
from aidial_rag.retrievers.colpali_retriever.colpali_models import (
    MODEL_NAME_TO_TYPE,
    KnownModels,
    load_model_and_processor,
)

DEFAULT_BATCH_SIZE = 8


class ColpaliModelResourceConfig(BaseModel):
    model_name: Annotated[
        str,
        Field(
            default=KnownModels.COLSMOL_256M,
            description="Model name, should be one of MODEL_NAME_TO_TYPE keys",
        ),
    ]
    batch_size: Annotated[
        int,
        Field(
            default=DEFAULT_BATCH_SIZE,
            description="Batch size for processing queries and images",
        ),
    ]

    def validate_model_name(self):
        """Validate that model name is known"""
        if self.model_name not in MODEL_NAME_TO_TYPE:
            raise ValueError(
                f"Model name '{self.model_name}' is not known. Please use one of the following: {list(MODEL_NAME_TO_TYPE.keys())}"
            )

    @model_validator(mode="after")
    def validate_model_consistency(self):
        """Validate that model name is known."""
        self.validate_model_name()
        return self


class ColpaliModelResource:
    def __init__(
        self,
        model_resource_config: ColpaliModelResourceConfig | None,
        colpali_index_config: ColpaliIndexConfig | None,
    ):
        self.model_resource_config: ColpaliModelResourceConfig | None = (
            model_resource_config
        )
        self.model = None
        self.device: torch.device | None = None
        self.processor = None
        # if both are set then we can load model
        if (
            colpali_index_config is not None
            and colpali_index_config.enabled
            and model_resource_config is not None
        ):
            self.device = torch.device(autodetect_device().value)
            self.model, self.processor = load_model_and_processor(
                model_resource_config.model_name, self.device
            )

    def get_model_processor_device(self):
        if (
            self.model_resource_config is None
            or self.device is None
            or self.model is None
            or self.processor is None
        ):
            raise ValueError(
                "ColpaliModelResourceConfig andColpaliIndexConfig are required"
            )
        return self.model, self.processor, self.device

    def get_batch_size(self):
        return self.model_resource_config.batch_size if self.model_resource_config else DEFAULT_BATCH_SIZE