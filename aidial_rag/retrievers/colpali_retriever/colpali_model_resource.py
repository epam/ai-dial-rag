from typing import Annotated, Any, List, Optional

import torch
from PIL import Image as pil_image
from pydantic import BaseModel, Field, model_validator
from torch import Tensor

from aidial_rag.embeddings.detect_device import autodetect_device
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
    model_path: Annotated[
        Optional[str],
        Field(
            default=None,
            description="Path to pre-downloaded ColPali models, if None then model will be downloaded from Hugging Face",
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
    """ColPali model resource, that stores model and processor"""

    def __init__(self, model_resource_config: ColpaliModelResourceConfig):
        self.model_resource_config: ColpaliModelResourceConfig = (
            model_resource_config
        )

        self.device = torch.device(autodetect_device().value)
        self.model, self.processor = load_model_and_processor(
            model_resource_config.model_name,
            model_resource_config.model_path,
            self.device,
        )

    def _run_model(self, inputs: Any) -> List[Tensor]:
        """Method to run the model with inputs."""
        if self.model is None or self.processor is None or self.device is None:
            raise ValueError(
                "ColPali model, processor, or device is not initialized."
            )

        with torch.no_grad():
            embeddings = self.model(**inputs)

        # Split batch tensor into individual tensors and move to CPU
        return [tensor.cpu().unsqueeze(0) for tensor in embeddings]

    def calculate_queries_embeddings(self, queries: List[str]) -> List[Tensor]:
        """Embed queries using the ColPali model."""
        assert self.processor is not None
        # Process queries with ColPali
        inputs = self.processor.process_queries(queries).to(self.device)

        return self._run_model(inputs)

    def calculate_images_embeddings(
        self, images: List[pil_image.Image]
    ) -> List[Tensor]:
        """Embed images using the ColPali model."""
        # Process images with ColPali
        inputs = self.processor.process_images(images).to(self.device)
        return self._run_model(inputs)

    def calculate_scores(
        self, query_embeddings: Tensor, image_embeddings: List[Tensor]
    ) -> Tensor:
        """Calculate scores between query and image embeddings."""
        return self.processor.score_multi_vector(
            query_embeddings, image_embeddings
        )

    def get_batch_size(self):
        return (
            self.model_resource_config.batch_size
            if self.model_resource_config
            else DEFAULT_BATCH_SIZE
        )
