import logging
from typing import AsyncGenerator, Iterable, List, Optional

from PIL.Image import Image

from aidial_rag.errors import InvalidDocumentError
from aidial_rag.image_processor.document_image_extractor import (
    DocumentPageImageExtractor,
)
from aidial_rag.image_processor.image_page_image_extractor import (
    ImagePageImageExtractor,
)
from aidial_rag.image_processor.pdf_page_image_extractor import (
    PdfPageImageExtractor,
)
from aidial_rag.utils import check_mime_type

logger = logging.getLogger(__name__)


extractors = [ImagePageImageExtractor(), PdfPageImageExtractor()]

supported_mime_types = {
    mime
    for extractor in extractors
    for mime in extractor.get_supported_mime_types()
}


def get_extractor(mime_type: str) -> DocumentPageImageExtractor:
    for extractor in extractors:
        if extractor.is_mime_supported(mime_type):
            return extractor
    raise InvalidDocumentError(f"Unsupported file type: {mime_type}")


async def extract_pages_gen(
    mime_type: str,
    file_bytes: bytes,
    page_numbers: Iterable[int],
    scaled_size: Optional[int] = None,
) -> AsyncGenerator[Image, None]:
    extractor = get_extractor(mime_type)
    async for image in extractor.extract_pages_gen(
        file_bytes, page_numbers, scaled_size
    ):
        yield image


async def extract_pages(
    mime_type: str,
    file_bytes: bytes,
    page_numbers: Iterable[int],
    scaled_size: Optional[int] = None,
) -> List[Image]:
    return [
        i
        async for i in extract_pages_gen(
            mime_type, file_bytes, page_numbers, scaled_size
        )
    ]


def extract_number_of_pages(mime_type: str, document_bytes: bytes) -> int:
    return get_extractor(mime_type).get_number_of_pages(document_bytes)


def are_image_pages_supported(mime: str) -> bool:
    return check_mime_type(mime, supported_mime_types)


def is_image(mime_type):
    return ImagePageImageExtractor().is_mime_supported(mime_type)
