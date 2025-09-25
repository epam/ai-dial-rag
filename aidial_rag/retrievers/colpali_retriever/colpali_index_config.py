from typing import Annotated

from pydantic import BaseModel, Field


class ColpaliIndexConfig(BaseModel):
    enabled: Annotated[
        bool,
        Field(default=True, description="Enable ColPali index building"),
    ]
