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
from aidial_rag.documents import get_default_image_chunk
from tests.utils.local_http_server import start_local_server
from tests.utils.e2e_decorator import e2e_test

# Shared test configuration
COLPALI_TEST_CONFIGS = [
    {
        "name": "test_colpali_retrieval",
        "query": "what is the caption of the image of butterfly?",
        "expected_answer": "The alpine Apollo butterfly has adapted to|alpine conditions.",
        "expected_page": 13,
        "description": "Test ColPali retriever"
    }
]

DATA_DIR = "tests/data"
PORT = 5010



@pytest.fixture
def local_server():
    with start_local_server(data_dir=DATA_DIR, port=PORT) as server:
        yield server

def check_expected_text(expected_text, actual_text):
    """
    Due to LLM response entropy, the response may contain different synonyms of same term
    The function able to check if a string contains any of synonyms from expected_text in actual text, divided by "|"
    character. For example, for the question "What's this document about?", LLM might give different right answers:
    "This document is about..." or "The document describes..." or "It covers..."
    """
    expected_terms = expected_text.split("|")
    return any(term.lower() in actual_text.lower() for term in expected_terms)

@pytest.mark.asyncio
async def test_colpali_retriever(local_server):
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

    # Test retrieval using shared config
    test_config = COLPALI_TEST_CONFIGS[0]  # Use butterfly test
    results = retriever._get_relevant_documents(test_config["query"])
    assert results, "No results returned"
    
    if use_cache:
        # Verify we're using recorded outputs
        recorded_outputs = colpali_model_resource.get_recorded_outputs()
        recorded_scores = colpali_model_resource.get_recorded_scores()
        assert len(recorded_outputs) > 0 or len(recorded_scores) > 0, "Should have recorded data"
        
    # Check expected page number
    chunk_id = results[0].metadata.get("chunk_id")
    if chunk_id is not None and chunk_id < len(text_chunks):
        page_number = text_chunks[chunk_id].metadata.get("page_number")
        assert page_number == test_config["expected_page"], f"Expected page {test_config['expected_page']}, got page {page_number}"

# E2E test functions
def create_colpali_only_config():
    """Create a configuration that uses only ColPali index."""
    from aidial_rag.app_config import AppConfig, RequestConfig, IndexingConfig
    from tests.utils.e2e_decorator import e2e_test
    
    middleware_host = "http://localhost:8081"
    
    return AppConfig(
        dial_url=middleware_host,
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


def run_colpali_e2e_test(attachments, test_config):
    """Run a ColPali-specific e2e test with the given test configuration."""
    from aidial_rag.app import create_app
    from fastapi.testclient import TestClient
    
    # Use ColPali-only configuration
    app_config = create_colpali_only_config()
    
    # Check if REFRESH is set to determine cache usage
    use_cache = not os.environ.get('REFRESH', '').lower() == 'true'
    
    # Patch the ColpaliModelResource at the module level where it's imported
    with patch('aidial_rag.app.ColpaliModelResource', lambda: CachedColpaliModelResource(use_cache=use_cache)):
        app = create_app(app_config)
        client = TestClient(app)
        
        # Use ColPali model command to ensure we're using ColPali retrieval
        response = client.post(
            "/openai/deployments/dial-rag/chat/completions",
            headers={"Api-Key": "api-key"},
            json={
                "model": "dial-rag",
                "messages": [
                    {
                        "role": "user",
                        "content": test_config["query"],
                        "custom_content": {"attachments": attachments},
                    }
                ],
                "max_tokens": 1000,
                "temperature": 0.0,
            },
        )
        
        assert response.status_code == 200, f"Request failed: {response.text}"
        
        response_data = response.json()
        assert "choices" in response_data, "No choices in response"
        assert len(response_data["choices"]) > 0, "Empty choices"
        
        content = response_data["choices"][0]["message"]["content"]
        assert content, "Empty content in response"
        
        # Check if any expected text is in the response
        found_expected = check_expected_text(test_config["expected_answer"], content)
        
        if not found_expected:
            print(f"Expected one of: {test_config['expected_answer']}")
            print(f"Got: {content}")
            assert False, "Expected text not found in response"


@pytest.mark.asyncio
async def test_colpali_retrieval_e2e():
    """Test ColPali retrieval with butterfly query using shared configuration."""
    # Custom e2e test that uses colpali_cache directory
    from tests.utils.e2e_decorator import start_server, wait_for_server_ready, stop_server, fail_on_warnings
    from tests.utils.cache_middleware import CacheMiddlewareConfig, CacheMiddlewareApp
    import uvicorn
    import asyncio
    
    # Create custom cache configuration that uses test_colpali_retriever directory
    cache_config = CacheMiddlewareConfig(
        dial_core_host=os.getenv("DIAL_CORE_HOST", "localhost:8080"),
        dial_core_api_key=os.getenv("DIAL_CORE_API_KEY", "dial_api_key"),
        base_path="tests",  # Use tests as base path so data is found at tests/data/
        test_name="test_colpali_retrieval_e2e",
        module_name="test_colpali_retriever",
        refresh=os.getenv('REFRESH', '').lower() == 'true',
    )
    
    # Create cache middleware app with custom config
    cache_middleware_app = CacheMiddlewareApp(cache_config)
    
    # Start server with custom cache middleware
    config = uvicorn.Config(cache_middleware_app, host="127.0.0.1", port=8081, log_level="debug")
    server = uvicorn.Server(config)
    
    # Start the server in a separate task
    loop = asyncio.get_event_loop()
    server_future = loop.run_in_executor(None, server.run)
    
    try:
        await wait_for_server_ready(port=8081)
        
        # Create attachments for the test
        attachments = [
            {
                "type": "application/pdf",
                "title": "alps_wiki.pdf",
                "url": "files/6iTkeGUs2CvUehhYLmMYXB/alps_wiki.pdf",
            }
        ]
        
        # Run the actual test
        with patch('aidial_rag.app.create_retriever', new=mock_create_retriever):
            run_colpali_e2e_test(attachments, COLPALI_TEST_CONFIGS[0])
            
    finally:
        # Shutdown the server
        server.should_exit = True
        await asyncio.wait_for(server_future, timeout=30)



