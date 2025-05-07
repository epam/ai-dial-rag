"""
Page description class
"""

# pylint: disable=C0301,C0103,C0303,C0411,W1203

import enum
import json
import logging
from dataclasses import dataclass

from dataclasses_json import dataclass_json

logger = logging.getLogger(__name__)


@enum.unique
class ImageDetails(enum.Enum):
    LOW = "low"
    HIGH = "high"
    AUTO = "auto"


@dataclass_json
@dataclass(frozen=True)
class PageChart:
    """
    PageChart class
    """

    description: str
    type: str
    keyfact: str


@dataclass_json
@dataclass(frozen=True)
class PageImage:
    """
    PageImage class
    """

    description: str
    keyfact: str


@dataclass_json
@dataclass(frozen=True)
class PageTable:
    """
    PageTable class
    """

    description: str
    keyfact: str


@dataclass_json
@dataclass(frozen=True)
class PageDescription:
    """
    Page description class
    """

    page_summary: str
    key_fact: str
    image_quality: ImageDetails
    image_quality_explanation: str
    images: list[PageImage]
    tables: list[PageTable]

    @classmethod
    def from_json_str(cls, json_str: str) -> "PageDescription":
        """
        Convert from json string
        """
        json_page = json.loads(json_str)

        page_summary = json_page["page_summary"]
        keyfact = json_page["keyfact"]

        image_quality_str = json_page["image_quality"]["level"]
        image_quality_str = image_quality_str.lower()
        if image_quality_str == "detailed":
            image_quality = ImageDetails.HIGH
        elif image_quality_str == "normal":
            image_quality = ImageDetails.LOW
        else:
            image_quality = ImageDetails.AUTO
        image_quality_explanation = json_page["image_quality"]["explanation"]

        # all image (illustration) descriptions
        images: list[PageImage] = []
        for image_json in json_page["images"]:
            if "image" in image_json:
                image_description = image_json["image"]["description"]
                image_keyfact = image_json["image"]["keyfact"]
            else:
                image_description = image_json["description"]
                image_keyfact = image_json["keyfact"]

            if "no images are present" in image_description.lower():
                continue

            images.append(PageImage(image_description, image_keyfact))

        # embedding for all table descriptions
        tables: list[PageTable] = []
        for table_json in json_page["tables"]:
            if "table" in table_json:
                table_description = table_json["table"]["description"]
                table_keyfact = table_json["table"]["keyfact"]
            else:
                table_description = table_json["description"]
                table_keyfact = table_json["keyfact"]

            if "no tables are present" in table_description.lower():
                continue

            tables.append(PageTable(table_description, table_keyfact))
        result = cls(
            page_summary,
            keyfact,
            image_quality,
            image_quality_explanation,
            images,
            tables,
        )
        return result

    def to_chunks(self) -> list[str]:
        page_chunk_list: list[str] = []

        def add_into_page_chunk_list(chunk: str):
            chunk = chunk.replace("\n", " ").replace("\r", " ")
            page_chunk_list.append(chunk)

        add_into_page_chunk_list(self.page_summary)
        add_into_page_chunk_list(self.key_fact)

        for image in self.images:
            add_into_page_chunk_list(image.description)
            add_into_page_chunk_list(image.keyfact)

        for table in self.tables:
            add_into_page_chunk_list(table.description)
            add_into_page_chunk_list(table.keyfact)

        return page_chunk_list
