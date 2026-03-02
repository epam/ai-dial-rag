import sys

import pytest

from aidial_rag.attachment_link import AttachmentLink
from aidial_rag.converter import convert_document_if_needed
from aidial_rag.document_loaders import load_attachment, parse_document
from tests.utils.local_http_server import start_local_server

DATA_DIR = "tests/data"
PORT = 5007


async def load_document(name):
    document_link = f"http://localhost:{PORT}/{name}"

    attachment_link = AttachmentLink(
        dial_link=document_link,
        absolute_url=document_link,
        display_name=name,
    )

    file_metadata, original_file_bytes = await load_attachment(
        attachment_link, {}
    )

    mime_type, buffer = await convert_document_if_needed(
        mime_type=file_metadata.mime_type,
        doc_bytes=original_file_bytes,
        io_stream=sys.stderr,
    )

    chunks = await parse_document(
        sys.stderr, buffer, mime_type, attachment_link, mime_type
    )
    assert chunks
    return chunks


@pytest.fixture
def local_server():
    with start_local_server(data_dir=DATA_DIR, port=PORT) as server:
        yield server


@pytest.mark.asyncio
async def test_load_pdf_with_image_and_text(local_server):
    await load_document("test_pdf_with_image_and_text.pdf")


@pytest.mark.asyncio
async def test_load_pdf_with_image_and_no_text(local_server):
    chunks = await load_document("test_pdf_with_image.pdf")
    assert len(chunks) == 1
    assert chunks[0].page_content == ""
    assert chunks[0].metadata["filetype"] == "application/pdf"
    assert chunks[0].metadata["page_number"] == 1


@pytest.mark.asyncio
async def test_load_pdf_with_broken_xref(local_server):
    """Test loading a PDF with broken xref table

    Some pdf libraries will unable to load this pdf, because they unable to find pages catalog using broken xref table and return empty document. But the PDF can be loaded successfully via sequential file scanning when the xref table is broken.
    """
    chunks = await load_document("test_pdf_with_image_broken_xref.pdf")
    # Should successfully load at least 1 chunk (1 page with image)
    assert len(chunks) >= 1
    assert chunks[0].metadata.get("page_number") == 1
    assert chunks[0].metadata["filetype"] == "application/pdf"


@pytest.mark.asyncio
async def test_load_single_line_text(local_server):
    chunks = await load_document("hello.txt")
    assert len(chunks) == 1
    assert chunks[0].page_content == "Hello, world!"
    assert chunks[0].metadata["filetype"] == "text/plain"
    assert "page_number" not in chunks[0].metadata
    assert "orig_elements" not in chunks[0].metadata


@pytest.mark.asyncio
async def test_load_docx_with_long_table(local_server):
    """Test that docx with long table can be loaded successfully.

    Older versions of libreoffice may hang when converting this docx to pdf.
    See https://ask.libreoffice.org/t/file-can-be-converted-to-pdf-using-gui-but-not-with-cmd/125926/3
    """

    chunks = await load_document("test_long_table.docx")
    assert len(chunks) == 22
    assert chunks[0].page_content.startswith("X")
    assert chunks[0].metadata["filetype"] == "application/pdf"
    assert chunks[0].metadata["page_number"] == 1
