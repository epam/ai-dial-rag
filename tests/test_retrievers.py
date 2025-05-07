import sys
from operator import itemgetter

import pytest
from langchain.schema.runnable import RunnablePassthrough

from aidial_rag.attachment_link import AttachmentLink
from aidial_rag.document_loaders import load_attachment, parse_document
from aidial_rag.document_record import (
    FORMAT_VERSION,
    DocumentRecord,
    IndexSettings,
    build_chunks_list,
)
from aidial_rag.documents import parse_content_type
from aidial_rag.retrievers.bm25_retriever import BM25Retriever
from aidial_rag.retrievers.semantic_retriever import SemanticRetriever
from aidial_rag.retrievers_postprocess import get_text_chunks
from tests.utils.local_http_server import start_local_server

DATA_DIR = "tests/data"
PORT = 5007


@pytest.fixture
def local_server():
    with start_local_server(data_dir=DATA_DIR, port=PORT) as server:
        yield server


async def run_retrevier(retriever, doc_records, query):
    retriever_chain = (
        RunnablePassthrough().assign(
            found_items=(itemgetter("query") | retriever)
        )
        | get_text_chunks
    )

    return await retriever_chain.ainvoke(
        {"query": query, "doc_records": doc_records}
    )


@pytest.mark.asyncio
async def test_retrievers(local_server):
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
    doc_records = [doc_record]

    bm25_retriever = BM25Retriever.from_doc_records(doc_records, 7)

    res = await run_retrevier(bm25_retriever, doc_records, "Colle di Cadibona")
    assert len(res)
    assert res[0].metadata["page_number"] == 3
    assert res[0].metadata["chunk_id"] == 31
    assert "Colle di Cadibona" in res[0].page_content

    semantic_retriever = SemanticRetriever.from_doc_records(doc_records, 7)
    res = await run_retrevier(
        semantic_retriever, doc_records, "what is the climate in the alps?"
    )

    print(res)
    assert len(res)
    assert res[0].metadata["page_number"] == 10
    assert res[0].metadata["chunk_id"] == 103
    assert (
        "Climate\n\n"
        "The Alps are a classic example of what happens when a temperate area at "
        "lower altitude gives way to higher-elevation terrain."
        in res[0].page_content
    )


@pytest.mark.asyncio
async def test_pdf_with_no_text(local_server):
    name = "test_pdf_with_image.pdf"
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

    assert len(text_chunks) == 1

    chunks = await build_chunks_list(text_chunks)
    text_index = await BM25Retriever.build_index(chunks)

    doc_record = DocumentRecord(
        format_version=FORMAT_VERSION,
        index_settings=IndexSettings(),
        chunks=chunks,
        text_index=text_index,
        embeddings_index=None,
        multimodal_embeddings_index=None,
        description_embeddings_index=None,
        document_bytes=buffer,
        mime_type=mime_type,
    )
    doc_records = [doc_record]

    assert not BM25Retriever.has_index(doc_records)

    with pytest.raises(ValueError, match="Text index is empty."):
        BM25Retriever.from_doc_records(doc_records)
