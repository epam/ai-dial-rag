import importlib.metadata
import sys
from pathlib import Path

import pytest

from aidial_rag.attachment_link import AttachmentLink
from aidial_rag.document_loaders import load_attachment, parse_document
from aidial_rag.document_record import (
    FORMAT_VERSION,
    DocumentRecord,
    IndexSettings,
    build_chunks_list,
)
from aidial_rag.documents import parse_content_type
from aidial_rag.index_storage import SERIALIZATION_CONFIG
from aidial_rag.retrievers.bm25_retriever import BM25Retriever
from aidial_rag.retrievers.semantic_retriever import SemanticRetriever
from tests.utils.local_http_server import start_local_server

DATA_DIR = "tests/data"
PORT = 5007

INDEX_DIR = Path("tests/data/index")

AIDIAL_RAG_VERSION = importlib.metadata.version("aidial-rag")


@pytest.fixture
def local_server():
    with start_local_server(data_dir=DATA_DIR, port=PORT) as server:
        yield server


@pytest.mark.skip(reason="Run manually to create index file")
@pytest.mark.asyncio
async def test_prepare_index(local_server):
    name = "alps_wiki.pdf"
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
    text_chunks = await parse_document(
        sys.stderr, buffer, mime_type, attachment_link, mime_type
    )

    assert len(text_chunks) == 177

    chunks = await build_chunks_list(text_chunks)
    text_index = await BM25Retriever.build_index(chunks)
    embeddings_index = await SemanticRetriever.build_index(chunks)

    doc_record = DocumentRecord(
        format_version=FORMAT_VERSION,
        index_settings=IndexSettings(),
        chunks=chunks,
        text_index=text_index,
        embeddings_index=embeddings_index,
        multimodal_embeddings_index=None,
        description_embeddings_index=None,
        document_bytes=buffer,
        mime_type=mime_type,
    )

    index_bytes = doc_record.to_bytes(**SERIALIZATION_CONFIG)
    index_file = INDEX_DIR / f"doc_record_{AIDIAL_RAG_VERSION}.bin"
    with open(index_file, "wb") as f:
        f.write(index_bytes)


@pytest.mark.parametrize(
    "index_file",
    [
        "doc_record_0.34.0rc0.bin",
    ],
)
@pytest.mark.asyncio
async def test_load_old_indexes(index_file):
    with open(INDEX_DIR / index_file, "rb") as f:
        doc_record = DocumentRecord.from_bytes(f.read(), **SERIALIZATION_CONFIG)

    assert isinstance(doc_record, DocumentRecord)
