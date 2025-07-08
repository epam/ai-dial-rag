import asyncio
import json
import logging
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from aidial_rag.app import create_app, DialRAGApplication, APP_NAME, process_load_errors, doc_to_attach
from aidial_rag.app_config import AppConfig, RequestConfig, IndexingConfig
from aidial_rag.attachment_link import get_attachment_links, format_document_loading_errors
from aidial_rag.commands import process_commands
from aidial_rag.documents import load_documents
from aidial_rag.qa_chain import generate_answer
from aidial_rag.request_context import create_request_context
from aidial_rag.retrievers.colpali_retriever.colpali_retriever import ColpaliRetriever
from aidial_rag.retrievers.colpali_retriever.colpali_index_config import ColpaliIndexConfig, ColpaliModelType
from aidial_rag.utils import timed_stage, profiler_if_enabled
from tests.utils.e2e_decorator import e2e_test
from tests.utils.colpali_cache import CachedColpaliModelResource

middleware_host = "http://localhost:8081"

def check_expected_text(expected_text, actual_text):
    """
    Due to LLM response entropy, the response may contain different synonyms of same term
    The function able to check if a string contains any of synonyms from expected_text in actual text, divided by "|"
    character. For example, for the question "What's this document about?", LLM might give different right answers:
    "This document is about..." or "The document describes..." or "It covers..."
    """
    expected_terms = expected_text.split("|")
    return any(term.lower() in actual_text.lower() for term in expected_terms)

def create_colpali_only_config():
    """Create a configuration that uses only ColPali index."""
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
    # Create a mock ColPali retriever
    if (
        colpali_index_config is not None
        and colpali_model_resource is not None
        and ColpaliRetriever.has_index(document_records)
    ):
        return ColpaliRetriever.from_doc_records(
            colpali_model_resource,
            colpali_index_config,
            document_records,
            2,
        )
    else:
        # Fallback to a simple mock retriever if ColPali is not available
        from aidial_rag.retrievers.all_documents_retriever import AllDocumentsRetriever
        return AllDocumentsRetriever.from_doc_records(document_records)

def run_colpali_test(attachments, question, expected_text_list):
    """Run a ColPali-specific test with the given question and expected responses."""
    # Use ColPali-only configuration
    app_config = create_colpali_only_config()
    
    # Patch the ColpaliModelResource at the module level where it's imported
    with patch('aidial_rag.app.ColpaliModelResource', CachedColpaliModelResource):
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
                        "content": question,
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
        found_expected = any(
            check_expected_text(expected_text, content)
            for expected_text in expected_text_list
        )
        
        if not found_expected:
            print(f"Expected one of: {expected_text_list}")
            print(f"Got: {content}")
            assert False, "Expected text not found in response"

@pytest.mark.asyncio
@e2e_test(filenames=["alps_wiki.pdf"])
#@e2e_test(filenames=["test_image.png"])
async def test_colpali_retrieval_butterfly(attachments):
    """Test ColPali retrieval with a question about an image in the document."""
    # Patch the create_retriever function at the module level where it's imported
    with patch('aidial_rag.app.create_retriever', new=mock_create_retriever):
        run_colpali_test(
            attachments,
            "What image is shown on page 13?",
            ["butterfly|butterflies"],  # Expected to find butterfly image on page 13
        )