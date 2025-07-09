import enum
from typing import Annotated

from pydantic import BaseModel, Field

from aidial_rag.base_config import IndexRebuildTrigger


class ColpaliModelType(str, enum.Enum):
    COLPALI = "ColPali"
    COLQWEN = "ColQwen"
    COLIDEFICS = "ColIdefics"


class ColpaliIndexConfig(BaseModel):
    model_name: Annotated[
        str,
        IndexRebuildTrigger(),
        Field(
            default="vidore/colSmol-256M",
        ),
    ]
    model_type: Annotated[
        ColpaliModelType,
        IndexRebuildTrigger(),
        Field(
            default=ColpaliModelType.COLIDEFICS,
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
                "dimensions required by that model."
            ),
        ),
    ]
