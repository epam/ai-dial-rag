import os
import sys
from operator import itemgetter

import pytest
from langchain.schema.runnable import RunnablePassthrough
from pydantic import SecretStr

from aidial_rag.app_config import IndexingConfig
from aidial_rag.attachment_link import AttachmentLink
from aidial_rag.dial_config import DialConfig
from aidial_rag.dial_user_limits import get_user_limits_for_model
from aidial_rag.document_loaders import load_attachment, parse_document
from aidial_rag.document_record import (
    FORMAT_VERSION,
    DocumentRecord,
    build_chunks_list,
)
from aidial_rag.documents import parse_content_type
from aidial_rag.resources.dial_limited_resources import DialLimitedResources
from aidial_rag.retrievers.multimodal_retriever import (
    Metric,
    MultimodalIndexConfig,
    MultimodalRetriever,
)
from aidial_rag.retrievers_postprocess import get_text_chunks
from tests.utils.local_http_server import start_local_server

DATA_DIR = "tests/data"
PORT = 5008


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
    document = await parse_document(
        sys.stderr, buffer, mime_type, attachment_link, mime_type
    )
    assert document
    return document


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


def has_dial_access():
    dial_url = os.environ.get("DIAL_URL")
    api_key = os.environ.get("DIAL_RAG_API_KEY")
    return dial_url and api_key


async def run_test_retrievers(
    local_server, multimodal_index_config: MultimodalIndexConfig
):
    dial_config = DialConfig(
        dial_url=os.environ.get("DIAL_URL", "http://localhost:8080"),
        api_key=SecretStr(os.environ.get("DIAL_RAG_API_KEY", "dial_api_key")),
    )

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

    index_config = IndexingConfig(
        multimodal_index=multimodal_index_config,
        description_index=None,
    )

    chunks = await build_chunks_list(text_chunks)
    multimodal_index = await MultimodalRetriever.build_index(
        dial_config=dial_config,
        dial_limited_resources=DialLimitedResources(
            lambda model_name: get_user_limits_for_model(
                dial_config, model_name
            )
        ),
        index_config=multimodal_index_config,
        mime_type=mime_type,
        original_document=buffer,
        stageio=sys.stderr,
    )

    doc_record = DocumentRecord(
        format_version=FORMAT_VERSION,
        index_settings=index_config.collect_fields_that_rebuild_index(),
        chunks=chunks,
        text_index=None,
        embeddings_index=None,
        multimodal_embeddings_index=multimodal_index,
        description_embeddings_index=None,
        document_bytes=buffer,
        mime_type=mime_type,
    )
    doc_records = [doc_record]

    multimodal_retriever = MultimodalRetriever.from_doc_records(
        dial_config=dial_config,
        index_config=multimodal_index_config,
        document_records=doc_records,
        k=7,
    )

    res = await run_retrevier(
        multimodal_retriever, doc_records, "image of butterfly"
    )
    assert len(res)
    assert res[0].metadata["page_number"] == 13


@pytest.mark.skipif(
    not has_dial_access(), reason="DIAL_URL and DIAL_RAG_API_KEY are not set"
)
@pytest.mark.asyncio
async def test_multimodalembedding_001(local_server):
    await run_test_retrievers(
        local_server,
        multimodal_index_config=MultimodalIndexConfig(
            embeddings_model="multimodalembedding@001",
            metric=Metric.SQEUCLIDEAN_DIST,
        ),
    )


@pytest.mark.skipif(
    not has_dial_access(), reason="DIAL_URL and DIAL_RAG_API_KEY are not set"
)
@pytest.mark.asyncio
async def test_azure_vision_embeddings(local_server):
    await run_test_retrievers(
        local_server,
        multimodal_index_config=MultimodalIndexConfig(
            embeddings_model="azure-ai-vision-embeddings",
            metric=Metric.COSINE_SIM,
        ),
    )
