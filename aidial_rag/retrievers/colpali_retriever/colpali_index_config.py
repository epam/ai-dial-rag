from typing import Annotated

from pydantic import BaseModel, Field

from aidial_rag.base_config import IndexRebuildTrigger


class ColpaliIndexConfig(BaseModel):
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
