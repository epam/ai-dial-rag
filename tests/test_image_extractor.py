import pytest

from aidial_rag.image_processor.extract_pages import extract_pages


@pytest.mark.asyncio
async def test_attachment_test():
    with open("tests/data/test_pdf_with_image_and_text.pdf", "rb") as pdf_bytes:
        images = await extract_pages(
            mime_type="application/pdf",
            file_bytes=pdf_bytes.read(),
            page_numbers=[1],
        )

    assert len(images) == 1
    assert images[0].height == 792
    assert images[0].width == 612


@pytest.mark.asyncio
async def test_attachment_test_scale():
    with open("tests/data/test_pdf_with_image_and_text.pdf", "rb") as pdf_bytes:
        images = await extract_pages(
            mime_type="application/pdf",
            file_bytes=pdf_bytes.read(),
            page_numbers=[1],
            scaled_size=500,
        )

    assert len(images) == 1
    assert images[0].height == 500
    assert images[0].width == 387


@pytest.mark.asyncio
async def test_attachment_image():
    with open("tests/data/test_image.jpg", "rb") as pdf_bytes:
        images = await extract_pages(
            mime_type="image/jpeg",
            file_bytes=pdf_bytes.read(),
            page_numbers=[1],
        )
    assert len(images) == 1
    assert images[0].height == 289
    assert images[0].width == 386


@pytest.mark.asyncio
async def test_attachment_image_invalid_page1():
    with open("tests/data/test_image.jpg", "rb") as pdf_bytes:
        with pytest.raises(RuntimeError):
            await extract_pages(
                mime_type="image/jpeg",
                file_bytes=pdf_bytes.read(),
                page_numbers=[2],
            )


@pytest.mark.asyncio
async def test_attachment_image_invalid_page2():
    with open("tests/data/test_image.jpg", "rb") as pdf_bytes:
        with pytest.raises(RuntimeError):
            await extract_pages(
                mime_type="image/jpeg",
                file_bytes=pdf_bytes.read(),
                page_numbers=[1, 1],
            )
