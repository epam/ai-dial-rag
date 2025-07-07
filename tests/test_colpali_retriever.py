import sys
import pytest

from aidial_rag.attachment_link import AttachmentLink
from aidial_rag.document_loaders import load_attachment, parse_document
from aidial_rag.document_record import FORMAT_VERSION, DocumentRecord, build_chunks_list, IndexSettings
from aidial_rag.documents import parse_content_type
from aidial_rag.resources.colpali_model_resource import ColpaliModelResource
from aidial_rag.retrievers.colpali_retriever.colpali_index_config import ColpaliIndexConfig, ColpaliModelType
from aidial_rag.retrievers.colpali_retriever.colpali_retriever import ColpaliRetriever
from aidial_rag.documents import get_default_image_chunk
from tests.utils.local_http_server import start_local_server

DATA_DIR = "tests/data"
PORT = 5010

@pytest.fixture
def local_server():
    with start_local_server(data_dir=DATA_DIR, port=PORT) as server:
        yield server

@pytest.mark.asyncio
async def test_colpali_retriever_simple(local_server):
    name = "alps_wiki_small.pdf"
    name = "alps_wiki.pdf"
    document_link = f"http://localhost:{PORT}/{name}"

    attachment_link = AttachmentLink(
        dial_link=document_link,
        absolute_url=document_link,
        display_name=name,
    )

    _file_name, content_type, buffer = await load_attachment(attachment_link, {})
    mime_type, _ = parse_content_type(content_type)
    text_chunks = await parse_document(
        sys.stderr, buffer, mime_type, attachment_link, mime_type
    )
    chunks_list = await build_chunks_list(text_chunks)

    # Build Colpali index
    colpali_model_resource = ColpaliModelResource()
    model_type = ColpaliModelType.COLPALI
    model_name = "vidore/colpali-v1.3"

    # model_name = "vidore/colqwen2-v1.0"
    # model_type = ColpaliModelType.COLQWEN
    colpali_index_config = ColpaliIndexConfig(model_name=model_name, model_type=model_type)
    colpali_index = await ColpaliRetriever.build_index(
        colpali_model_resource, colpali_index_config, sys.stderr, mime_type, buffer
    )

    doc_record = DocumentRecord(
        format_version=FORMAT_VERSION,
        index_settings=IndexSettings(indexes={}),
        chunks=chunks_list,
        text_index=None,
        embeddings_index=None,
        multimodal_embeddings_index=None,
        description_embeddings_index=None,
        colpali_embeddings_index=colpali_index,
        document_bytes=buffer,
        mime_type=mime_type,
    )
    doc_records = [doc_record]

    retriever = ColpaliRetriever.from_doc_records(
        colpali_model_resource, colpali_index_config, doc_records, k=2
    )

    results = retriever._get_relevant_documents("image of butterfly")
    assert results, "No results returned"
    # The expected page number for 'image of butterfly' is 13
    assert results[0].metadata.get("page_number") == 13
