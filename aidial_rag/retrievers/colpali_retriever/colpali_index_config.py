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
