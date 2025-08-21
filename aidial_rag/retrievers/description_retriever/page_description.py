import logging
from typing import List

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


PAGE_DESCRIPTION_PROMPT_TEMPLATE = """
Please create detailed description of provided image.
Ignore page header, footer, basic logo and background.
Describe all images (illustration), tables.
Text with bullet points is NOT a table or image.

Use only provided information.
DO NOT make up answer.

Provide answer as a PageDescription function call.
Make sure to properly escape special characters, like double quotes, in string fields.
"""


class ImageDescription(BaseModel):
    """Image description"""

    image_summary: str = Field(
        description="the summary of the image description"
    )
    keyfact: str = Field(description="the most important fact from the image")


class TableDescription(BaseModel):
    """Table description"""

    table_summary: str = Field(
        description="the summary of the table description"
    )
    keyfact: str = Field(description="the most important fact from the table")


class PageDescription(BaseModel):
    """Page description"""

    page_summary: str = Field(description="the summary of the page description")
    keyfact: str = Field(description="the most important fact from the page")
    images: List[ImageDescription] = Field(
        description="the array of the descriptions for the images on the page",
        default_factory=list,
    )
    tables: List[TableDescription] = Field(
        description="the array of the descriptions for the tables on the page",
        default_factory=list,
    )

    # We do not want to log user data outside of DEBUG log level
    model_config = ConfigDict(hide_input_in_errors=True)

    def to_chunks(self) -> List[str]:
        page_chunk_list: List[str] = []

        def add_into_page_chunk_list(chunk: str):
            chunk = chunk.replace("\n", " ").replace("\r", " ")
            page_chunk_list.append(chunk)

        add_into_page_chunk_list(self.page_summary)

        if self.keyfact:
            add_into_page_chunk_list(self.keyfact)

        for image in self.images:
            add_into_page_chunk_list(image.image_summary)
            add_into_page_chunk_list(image.keyfact)

        for table in self.tables:
            add_into_page_chunk_list(table.table_summary)
            add_into_page_chunk_list(table.keyfact)

        return page_chunk_list
