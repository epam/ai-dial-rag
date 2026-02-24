import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from typing import AsyncGenerator, Iterable, List, Optional

import pypdfium2 as pdfium
from PIL.Image import Image

from aidial_rag.image_processor.document_image_extractor import (
    DocumentPageImageExtractor,
)

logger = logging.getLogger(__name__)


def _calculate_scale(
    width: float, height: float, scaled_size: Optional[int]
) -> float:
    """Calculate scale factor to scale the larger dimension to scaled_size."""
    if not scaled_size:
        return 1.0

    if width > height:
        return scaled_size / width
    else:
        return scaled_size / height


def _get_number_of_pages(file_bytes: bytes) -> int:
    # Not thread safe because of pypdfium2
    with closing(pdfium.PdfDocument(file_bytes)) as pdf:
        return len(pdf)


def _render_page(
    file_bytes: bytes,
    page_number: int,
    scaled_size: Optional[int] = None,
) -> Image:
    # Not thread safe because of pypdfium2
    with closing(pdfium.PdfDocument(file_bytes)) as pdf:
        page = pdf[page_number - 1]  # pypdfium2 uses 0-based indexing

        scale = _calculate_scale(
            page.get_width(), page.get_height(), scaled_size
        )

        bitmap = page.render(
            # scale is float, but default value make pyright think it's int
            scale=scale,  # pyright: ignore [reportArgumentType]
            no_smoothtext=True,
            no_smoothpath=True,
            no_smoothimage=True,
            prefer_bgrx=True,
        )

        return bitmap.to_pil().convert("RGB")


class PdfPageImageExtractor(DocumentPageImageExtractor):
    supported_mime_types: List[str] = ["application/pdf"]

    # Use a thread pool with a single worker for non-thread safe methods
    _thread_pool = ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="pdf_page_image_extractor",
    )

    async def get_number_of_pages(self, file_bytes: bytes) -> int:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._thread_pool, _get_number_of_pages, file_bytes
        )

    async def extract_pages_gen(
        self,
        file_bytes: bytes,
        page_numbers: Iterable[int],
        scaled_size: Optional[int] = None,
    ) -> AsyncGenerator[Image, None]:
        loop = asyncio.get_running_loop()

        total_pages = await self.get_number_of_pages(file_bytes)
        for page_number in page_numbers:
            if not (1 <= page_number <= total_pages):
                raise RuntimeError(
                    f"Invalid page number: {page_number}. Page number is ordinal number of the page. The document has {total_pages} pages."
                )

            logger.debug(f"Extracting page {page_number}...")

            # Render in thread pool, because pypdfium2 is not thread safe
            image = await loop.run_in_executor(
                self._thread_pool,
                _render_page,
                file_bytes,
                page_number,
                scaled_size,
            )
            logger.debug(f"Extracted page {page_number} as image")
            yield image
