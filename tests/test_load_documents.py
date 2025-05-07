import sys

import pytest

from aidial_rag.attachment_link import AttachmentLink
from aidial_rag.document_loaders import load_attachment, parse_document
from aidial_rag.documents import parse_content_type
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

    _file_name, content_type, buffer = await load_attachment(
        attachment_link, {}
    )
    mime_type, _ = parse_content_type(content_type)
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
