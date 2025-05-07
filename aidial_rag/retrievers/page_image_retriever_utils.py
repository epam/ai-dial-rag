import logging
import sys
from typing import AsyncGenerator

from aidial_rag.content_stream import SupportsWriteStr
from aidial_rag.image_processor.base64 import pil_image_as_base64
from aidial_rag.image_processor.extract_pages import (
    are_image_pages_supported,
    extract_number_of_pages,
    extract_pages_gen,
)
from aidial_rag.resources.cpu_pools import CpuPools
from aidial_rag.resources.dial_limited_resources import AsyncGeneratorWithTotal

logger = logging.getLogger(__name__)


async def extract_page_images(
    mime_type: str,
    original_document: bytes,
    extract_pages_kwargs: dict,
    stageio: SupportsWriteStr = sys.stderr,
) -> AsyncGeneratorWithTotal | None:
    if not are_image_pages_supported(mime_type):
        stageio.write(
            f"Skipping page images: not supported file type: {mime_type}\n"
        )
        return None

    number_of_pages = extract_number_of_pages(mime_type, original_document)

    stageio.write("Extracting page images\n")
    stageio.write(f"Number of pages: {number_of_pages}\n")

    # Page numbers are 1-based
    page_nums = list(range(1, number_of_pages + 1))

    page_images_gen = extract_pages_gen(
        mime_type, original_document, page_nums, **extract_pages_kwargs
    )

    cpu_pools = CpuPools.instance()
    images_base64_agen: AsyncGenerator[str, None] = (
        await cpu_pools.run_in_indexing_cpu_pool(
            pil_image_as_base64, page_image, "PNG"
        )
        async for page_image in page_images_gen
    )

    return AsyncGeneratorWithTotal(images_base64_agen, number_of_pages)
