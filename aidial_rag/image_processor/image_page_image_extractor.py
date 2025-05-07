import io
from typing import AsyncGenerator, Iterable, List, Optional

from PIL.Image import Image
from PIL.Image import open as pil_image_open

from aidial_rag.image_processor.document_image_extractor import (
    DocumentPageImageExtractor,
)
from aidial_rag.image_processor.resize import resize_image


class ImagePageImageExtractor(DocumentPageImageExtractor):
    supported_mime_types: List[str] = ["image/*"]

    def get_number_of_pages(self, file_bytes: bytes) -> int:
        return 1

    async def extract_pages_gen(
        self,
        file_bytes: bytes,
        page_numbers: Iterable[int],  # Not used
        scaled_size: Optional[int] = None,
    ) -> AsyncGenerator[Image, None]:
        page_numbers_list = list(
            page_numbers
        )  # Convert to list for easy length checking
        if not (
            len(page_numbers_list) == 1
            and all(page == 1 for page in page_numbers_list)
        ):
            raise RuntimeError(
                f"Invalid page numbers: {page_numbers_list}. Page list should contain only 1 element. Image has only 1 page."
            )
        with pil_image_open(io.BytesIO(file_bytes)) as img:
            if scaled_size is None or scaled_size > max(img.width, img.height):
                # No need to resize
                # Need copy here since the original image is closed when the context manager exits
                yield img.copy()
            else:
                yield resize_image(img, (scaled_size, scaled_size))
