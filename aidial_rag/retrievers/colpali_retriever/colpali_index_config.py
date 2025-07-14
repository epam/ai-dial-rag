from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, Field, model_validator

from aidial_rag.base_config import IndexRebuildTrigger


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


class ColpaliIndexConfig(BaseModel):
    model_name: Annotated[
        str,
        IndexRebuildTrigger(),
        Field(
            default="vidore/colSmol-256M",
            description="Model name, should be one of KNOWN_MODELS keys",
        ),
    ]
    model_type: Annotated[
        ColpaliModelType,
        IndexRebuildTrigger(),
        Field(
            default=ColpaliModelType.COLIDEFICS,
            description="Type of ColPali model",
        ),
    ]
    image_size: Annotated[
        int,
        IndexRebuildTrigger(),
        Field(
            default=512,
            description=(
                "Specifies the size to which page images are initially resized before embedding calculation."
                "Note: Each model's processor may further resize images to the "
                "dimensions required by that model(ColPali: 448, ColQwen: not fixed, ColIdefics: 512)"
            ),
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
            raise ValueError(f"Model name '{self.model_name}' is not known. Please use one of the following: {list(KNOWN_MODELS.keys())}")

    @model_validator(mode='after')
    def validate_model_consistency(self):
        """Validate that model name and type are consistent."""
        self.validate_consistency()
        return self
