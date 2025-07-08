import sys
import os
import pytest
from unittest.mock import patch

from aidial_rag.attachment_link import AttachmentLink
from aidial_rag.document_loaders import load_attachment, parse_document
from aidial_rag.document_record import FORMAT_VERSION, DocumentRecord, build_chunks_list, IndexSettings
from aidial_rag.documents import parse_content_type
from aidial_rag.resources.colpali_model_resource import ColpaliModelResource
from tests.utils.colpali_cache import CachedColpaliModelResource
from aidial_rag.retrievers.colpali_retriever.colpali_index_config import ColpaliIndexConfig, ColpaliModelType
from aidial_rag.retrievers.colpali_retriever.colpali_retriever import ColpaliRetriever
from tests.utils.local_http_server import start_local_server
from tests.utils.e2e_decorator import e2e_test

# Test configuration
COLPALI_TEST_CONFIG = {
    "query": "what is the caption of the image of butterfly?",
    "expected_answer": "The alpine Apollo butterfly has adapted to|alpine conditions.",
    "expected_page": 13,
}

DATA_DIR = "tests/data"
PORT = 5010


@pytest.fixture
def local_server():
    """Start local HTTP server for serving test documents."""
    with start_local_server(data_dir=DATA_DIR, port=PORT) as server:
        yield server


async def load_document(name, port=PORT):
    """Load document from local server and parse it into chunks."""
    document_link = f"http://localhost:{port}/{name}"
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
    
    return text_chunks, buffer, mime_type


def check_expected_text(expected_text, actual_text):
    """
    Check if expected text (or any of its alternatives separated by '|') 
    is present in the actual text.
    """
    actual_text_lower = actual_text.lower()
    if "|" in expected_text:
        alternatives = expected_text.split("|")
        return any(alt.lower() in actual_text_lower for alt in alternatives)
    else:
        return expected_text.lower() in actual_text_lower


def create_colpali_only_config():
    """Create app configuration that uses only ColPali index."""
    from aidial_rag.app_config import AppConfig, RequestConfig, IndexingConfig
    
    return AppConfig(
        dial_url="http://localhost:8081",
        enable_debug_commands=True,
        request=RequestConfig(
            indexing=IndexingConfig(
                multimodal_index=None,  # Disable multimodal index
                description_index=None,  # Disable description index
                colpali_index=ColpaliIndexConfig(
                    model_name="vidore/colpali-v1.3",
                    model_type=ColpaliModelType.COLPALI
                )
            )
        )
    )


def mock_create_retriever(
    response_choice,
    dial_config,
    document_records,
    multimodal_index_config,
    colpali_model_resource,
    colpali_index_config,
):
    """Mock create_retriever to return only ColPali retriever."""
    return ColpaliRetriever.from_doc_records(
        colpali_model_resource,
        colpali_index_config,
        document_records,
        2,
    )


def run_e2e_test(attachments, question, expected_text):
    """Run end-to-end test using the app's chat completion endpoint."""
    from aidial_rag.app import create_app
    from fastapi.testclient import TestClient
    
    app = create_app(create_colpali_only_config())
    client = TestClient(app)
    
    response = client.post(
        "/openai/deployments/dial-rag/chat/completions",
        headers={"Api-Key": "api-key"},
        json={
            "model": "dial-rag",
            "messages": [
                {
                    "role": "user",
                    "content": question,
                    "custom_content": {"attachments": attachments},
                }
            ],
            "max_tokens": 1000,
            "temperature": 0.0,
        },
        timeout=100.0,
    )

    assert response.status_code == 200
    json_response = response.json()
    content = json_response["choices"][0]["message"]["content"]
    
    # Check if expected text is in the response
    found_expected = check_expected_text(expected_text, content)
    
    if not found_expected:
        print(f"Expected one of: {expected_text}")
        print(f"Got: {content}")
        assert False, "Expected text not found in response"
    
    return json_response


@pytest.mark.asyncio
async def test_colpali_retriever(local_server):
    """
    Unit test for ColPali retriever with simple query using cached model.
    Tests the retriever directly without going through the full app.
    """
    use_cache = not os.environ.get('REFRESH', '').lower() == 'true'

    # Load and process document
    text_chunks, buffer, mime_type = await load_document("alps_wiki.pdf")
    chunks_list = await build_chunks_list(text_chunks)

    # Setup ColPali model and index
    colpali_model_resource = CachedColpaliModelResource(use_cache=use_cache)
    colpali_index_config = ColpaliIndexConfig(
        model_name="vidore/colpali-v1.3",
        model_type=ColpaliModelType.COLPALI
    )

    # Build index
    colpali_index = await ColpaliRetriever.build_index(
        colpali_model_resource, colpali_index_config, sys.stderr, mime_type, buffer
    )

    # Create document record
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

    # Create retriever and test
    retriever = ColpaliRetriever.from_doc_records(
        colpali_model_resource, colpali_index_config, doc_records, k=2
    )

    # Test retrieval
    results = retriever._get_relevant_documents(COLPALI_TEST_CONFIG["query"])
    assert results, "No results returned"
    
    # Verify cache usage if applicable
    if use_cache:
        recorded_outputs = colpali_model_resource.get_recorded_outputs()
        recorded_scores = colpali_model_resource.get_recorded_scores()
        assert len(recorded_outputs) > 0 or len(recorded_scores) > 0, "Should have recorded data"
        
    # Verify expected page number
    chunk_id = results[0].metadata.get("chunk_id")
    if chunk_id is not None and chunk_id < len(text_chunks):
        page_number = text_chunks[chunk_id].metadata.get("page_number")
        expected_page = COLPALI_TEST_CONFIG["expected_page"]
        assert page_number == expected_page, f"Expected page {expected_page}, got page {page_number}"


@pytest.mark.asyncio
@e2e_test(filenames=("alps_wiki.pdf",))
async def test_colpali_retrieval_e2e(attachments):
    """
    End-to-end test for ColPali retrieval through the full app.
    Tests the complete flow from document upload to answer generation.
    """
    # Patch create_retriever to use only ColPali retriever
    with patch('aidial_rag.app.create_retriever', new=mock_create_retriever):
        run_e2e_test(
            attachments=attachments,
            question=COLPALI_TEST_CONFIG["query"],
            expected_text=COLPALI_TEST_CONFIG["expected_answer"]
        )
