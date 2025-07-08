import sys
import os
import pytest

from aidial_rag.attachment_link import AttachmentLink
from aidial_rag.document_loaders import load_attachment, parse_document
from aidial_rag.document_record import FORMAT_VERSION, DocumentRecord, build_chunks_list, IndexSettings
from aidial_rag.documents import parse_content_type
from aidial_rag.resources.colpali_model_resource import ColpaliModelResource
from tests.utils.colpali_cache import CachedColpaliModelResource
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
    """
    Test ColPali retriever with simple query using cached model.
    If data should be updated, set use_cache to False to record new cache.
    """
    use_cache = True

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

    # Use cached ColPali model resource with recording/replay system
    colpali_model_resource = CachedColpaliModelResource(use_cache=use_cache)
    colpali_index_config = ColpaliIndexConfig(
        model_name="vidore/colpali-v1.3",
        model_type=ColpaliModelType.COLPALI
    )

    # Build index using cached model
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

    # Test retrieval
    results = retriever._get_relevant_documents("what is the second image description where there is an image of butterfly")
    assert results, "No results returned"
    
    if use_cache:
        # Verify we're using recorded outputs
        recorded_outputs = colpali_model_resource.get_recorded_outputs()
        recorded_scores = colpali_model_resource.get_recorded_scores()
        assert len(recorded_outputs) > 0 or len(recorded_scores) > 0, "Should have recorded data"
        
    # The expected page number for 'image of butterfly' is 13
    # Since we're using mock scores that prioritize page 13, this should work
    chunk_id = results[0].metadata.get("chunk_id")
    if chunk_id is not None and chunk_id < len(text_chunks):
        page_number = text_chunks[chunk_id].metadata.get("page_number")
        # The mock processor should return page 13 as the top result
        assert page_number == 13, f"Expected page 13, got page {page_number}"



