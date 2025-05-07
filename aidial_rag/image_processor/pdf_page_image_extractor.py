import asyncio
import io
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import AsyncGenerator, Iterable, List, Optional

import pdfplumber
from pdfplumber.page import Page
from PIL.Image import Image

from aidial_rag.image_processor.document_image_extractor import (
    DocumentPageImageExtractor,
)

logger = logging.getLogger(__name__)


class PdfPageImageExtractor(DocumentPageImageExtractor):
    supported_mime_types: List[str] = ["application/pdf"]

    # Use a thread pool with a single worker for non-thread safe methods
    _thread_pool = ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="pdf_page_image_extractor",
    )

    def get_number_of_pages(self, file_bytes: bytes) -> int:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            return len(pdf.pages)

    def __get_page_image(
        self, page: Page, scaled_size: Optional[int] = None
    ) -> Image:
        width = None
        height = None
        if page.width > page.height:
            width = scaled_size
        else:
            height = scaled_size

        # __get_page_image is not thread safe, because to_image is not thread safe
        return page.to_image(width=width, height=height).original

    async def extract_pages_gen(
        self,
        file_bytes: bytes,
        page_numbers: Iterable[int],
        scaled_size: Optional[int] = None,
    ) -> AsyncGenerator[Image, None]:
        loop = asyncio.get_running_loop()
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            total_pages = len(pdf.pages)
            for page_number in page_numbers:
                if not (1 <= page_number <= total_pages):
                    raise RuntimeError(
                        f"Invalid page number: {page_number}. Page number is ordinal number of the page. The document has {total_pages} pages."
                    )

                logger.debug(f"Extracting page {page_number}...")
                page = pdf.pages[page_number - 1]

                image = await loop.run_in_executor(
                    self._thread_pool, self.__get_page_image, page, scaled_size
                )
                logger.debug(f"Extracted page {page_number} as image")
                yield image
