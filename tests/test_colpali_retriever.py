import os
import sys
from unittest.mock import patch

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
from aidial_rag.retrievers.colpali_retriever.colpali_index_config import (
    ColpaliIndexConfig,
)
from aidial_rag.retrievers.colpali_retriever.colpali_model_resource import (
    ColpaliModelResourceConfig,
    ColpaliModelType,
)
from aidial_rag.retrievers.colpali_retriever.colpali_retriever import (
    ColpaliRetriever,
)
from tests.utils.colpali_cache import CachedColpaliModelResource
from tests.utils.e2e_decorator import e2e_test
from tests.utils.local_http_server import start_local_server

# Test configuration
COLPALI_TEST_CONFIG = {
    "query": "what is the caption of the image of butterfly?",
    "expected_answer": "The alpine Apollo butterfly has adapted to alpine conditions",
    "expected_page": 13,
}

DATA_DIR = "tests/data"
PORT = 5010
MIDDLEWARE_HOST = "http://localhost:8081"


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

    _file_name, content_type, buffer = await load_attachment(
        attachment_link, {}
    )
    mime_type, _ = parse_content_type(content_type)
    text_chunks = await parse_document(
        sys.stderr, buffer, mime_type, attachment_link, mime_type
    )

    return text_chunks, buffer, mime_type


def create_colpali_only_config():
    """Create app configuration that uses Azure ColPali config."""
    from aidial_rag.app_config import AppConfig
    from aidial_rag.retrievers.colpali_retriever.colpali_model_resource import (
        ColpaliModelResourceConfig,
        ColpaliModelType,
    )

    return AppConfig(
        dial_url=MIDDLEWARE_HOST,
        enable_debug_commands=True,
        config_path="config/azure_colpali.yaml",
        colpali_model_resource_config=ColpaliModelResourceConfig(
            model_name="vidore/colSmol-256M",
            model_type=ColpaliModelType.COLIDEFICS,
        ),
    )


def mock_create_retriever(
    response_choice,
    dial_config,
    document_records,
    multimodal_index_config,
    colpali_model_resource,
    colpali_index_config,
):
    """Mock create_retriever to return only ColPali retriever with cached model."""
    use_cache = not os.environ.get("REFRESH", "").lower() == "true"
    # Create model resource config from the index config for backward compatibility
    colpali_model_resource_config = ColpaliModelResourceConfig(
        model_name="vidore/colSmol-256M",
        model_type=ColpaliModelType.COLIDEFICS,
    )
    cached_model_resource = CachedColpaliModelResource(
        colpali_model_resource_config, colpali_index_config, use_cache=use_cache
    )

    return ColpaliRetriever.from_doc_records(
        cached_model_resource,
        colpali_index_config,
        document_records,
        7,
    )


def create_cached_app_config():
    """Create app configuration that uses cached ColPali model resource."""
    from aidial_rag.app import DialRAGApplication

    class CachedDialRAGApplication(DialRAGApplication):
        def __init__(self, app_config):
            super().__init__(app_config)
            # Replace the real model resource with cached one
            use_cache = not os.environ.get("REFRESH", "").lower() == "true"
            self.colpali_model_resource = CachedColpaliModelResource(
                app_config.colpali_model_resource_config,
                app_config.request.indexing.colpali_index,
                use_cache=use_cache,
            )

    return CachedDialRAGApplication


def run_e2e_test(attachments, question, expected_text):
    """Run end-to-end test using the app's chat completion endpoint."""
    from fastapi.testclient import TestClient

    from aidial_rag.app import create_app

    # Create app with cached model resource
    app_config = create_colpali_only_config()
    cached_app_class = create_cached_app_config()

    # Patch the app creation to use cached model resource
    with patch("aidial_rag.app.DialRAGApplication", new=cached_app_class):
        app = create_app(app_config)
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
    found_expected = expected_text.lower() in content.lower()
    assert found_expected, f"Expected one of: {expected_text}\nGot: {content}"

    return json_response


def test_model_name_type_validation():
    """Test that model name and type validation works correctly."""

    # Test valid configuration
    valid_config = ColpaliModelResourceConfig(
        model_name="vidore/colSmol-256M",
        model_type=ColpaliModelType.COLIDEFICS,
    )
    assert valid_config.model_name == "vidore/colSmol-256M"
    assert valid_config.model_type == ColpaliModelType.COLIDEFICS

    # Test invalid configuration - should raise ValueError
    with pytest.raises(
        ValueError,
        match="Model name 'vidore/colSmol-256M' is known to be of type 'ColIdefics'",
    ):
        ColpaliModelResourceConfig(
            model_name="vidore/colSmol-256M",
            model_type=ColpaliModelType.COLPALI,  # Wrong type
        )

    # Test unknown model name - should raise error (current behavior)
    with pytest.raises(
        ValueError, match="Model name 'unknown/model' is not known"
    ):
        ColpaliModelResourceConfig(
            model_name="unknown/model",
            model_type=ColpaliModelType.COLPALI,
        )


@pytest.mark.asyncio
async def test_colpali_retriever(local_server):
    """
    Unit test for ColPali retriever that checks if retrieved page number is correct.
    """
    use_cache = not os.environ.get("REFRESH", "").lower() == "true"
    # Load and process document
    text_chunks, buffer, mime_type = await load_document("alps_wiki.pdf")
    chunks_list = await build_chunks_list(text_chunks)

    # Setup ColPali model and index using Azure config
    colpali_model_resource_config = ColpaliModelResourceConfig(
        model_name="vidore/colSmol-256M",
        model_type=ColpaliModelType.COLIDEFICS,
    )
    colpali_index_config = ColpaliIndexConfig(
        image_size=512,
    )

    colpali_model_resource = CachedColpaliModelResource(
        colpali_model_resource_config, colpali_index_config, use_cache=use_cache
    )

    # Build index
    colpali_index = await ColpaliRetriever.build_index(
        colpali_model_resource,
        colpali_index_config,
        sys.stderr,
        mime_type,
        buffer,
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
        colpali_model_resource, colpali_index_config, doc_records, k=7
    )

    # Test retrieval
    results = retriever._get_relevant_documents(COLPALI_TEST_CONFIG["query"])
    assert results, "No results returned"

    # Verify expected page number
    chunk_id = results[0].metadata.get("chunk_id")
    assert chunk_id is not None and chunk_id < len(text_chunks)
    page_number = text_chunks[chunk_id].metadata.get("page_number")
    expected_page = COLPALI_TEST_CONFIG["expected_page"]
    assert page_number == expected_page, (
        f"Expected page {expected_page}, got page {page_number}"
    )


@pytest.mark.asyncio
@e2e_test(filenames=("alps_wiki.pdf",))
async def test_colpali_retrieval_e2e(attachments):
    """
    End-to-end test for ColPali retrieval through the full app.
    Tests the complete flow from document upload to answer generation.
    """
    # Patch create_retriever to use only ColPali retriever
    with patch("aidial_rag.app.create_retriever", new=mock_create_retriever):
        run_e2e_test(
            attachments=attachments,
            question=COLPALI_TEST_CONFIG["query"],
            expected_text=COLPALI_TEST_CONFIG["expected_answer"],
        )
