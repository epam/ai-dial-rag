import pytest

from aidial_rag.errors import InvalidDocumentError
from aidial_rag.image_processor.extract_pages import (
    extract_number_of_pages,
    extract_pages,
)


def test_number_of_pages():
    with open("tests/data/test_pdf_with_image_and_text.pdf", "rb") as pdf_bytes:
        num_pages = extract_number_of_pages("application/pdf", pdf_bytes.read())

    assert num_pages == 1


@pytest.mark.asyncio
async def test_extract_pages_pdf():
    with open("tests/data/test_pdf_with_image_and_text.pdf", "rb") as pdf_bytes:
        images = await extract_pages("application/pdf", pdf_bytes.read(), [1])

    assert len(images) == 1
    assert images[0].height == 792
    assert images[0].width == 612


@pytest.mark.asyncio
async def test_extract_pages_invalid_document():
    with open("tests/data/alps_wiki.html", "rb") as pdf_bytes:
        with pytest.raises(InvalidDocumentError):
            await extract_pages("text/html", pdf_bytes.read(), [1])
