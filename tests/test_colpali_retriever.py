import asyncio
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
from aidial_rag.resources.dial_limited_resources import AsyncGeneratorWithTotal
from aidial_rag.retrieval_chain import _make_retrieval_stage_default
from aidial_rag.retrievers.colpali_retriever.colpali_index_config import (
    ColpaliIndexConfig,
)
from aidial_rag.retrievers.colpali_retriever.colpali_model_resource import (
    ColpaliModelResourceConfig,
)
from aidial_rag.retrievers.colpali_retriever.colpali_retriever import (
    ColpaliRetriever,
)
from aidial_rag.retrievers.page_image_retriever_utils import extract_page_images
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
    )

    return AppConfig(
        dial_url=MIDDLEWARE_HOST,
        enable_debug_commands=True,
        config_path="config/azure_colpali.yaml",
        colpali_model_resource_config=ColpaliModelResourceConfig(
            model_name="vidore/colSmol-256M",
        ),
    )


def mock_create_retriever(
    dial_config,
    document_records,
    indexing_config,
    colpali_model_resource,
    make_retrieval_stage=_make_retrieval_stage_default,
):
    """Mock create_retriever to return only ColPali retriever with cached model."""
    use_cache = not os.environ.get("REFRESH", "").lower() == "true"
    # Create model resource config from the index config for backward compatibility
    colpali_model_resource_config = ColpaliModelResourceConfig(
        model_name="vidore/colSmol-256M",
    )
    cached_model_resource = CachedColpaliModelResource(
        colpali_model_resource_config,
        indexing_config.colpali_index,
        use_cache=use_cache,
    )

    colpali_retriever = make_retrieval_stage(
        ColpaliRetriever.from_doc_records(
            cached_model_resource,
            indexing_config.colpali_index,
            document_records,
            7,
        ),
        "Colpali search",
    )
    return colpali_retriever


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
            "custom_fields": {
                "configuration": {"indexing": {"description_index": None}}
            },
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


def test_model_name_validation():
    """Test that model name validation works correctly."""

    # Test valid configuration
    valid_config = ColpaliModelResourceConfig(
        model_name="vidore/colSmol-256M",
    )
    assert valid_config.model_name == "vidore/colSmol-256M"

    # Test unknown model name - should raise error
    with pytest.raises(
        ValueError, match="Model name 'unknown/model' is not known"
    ):
        ColpaliModelResourceConfig(
            model_name="unknown/model",
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
    with patch(
        "aidial_rag.retrieval_chain.create_retriever", new=mock_create_retriever
    ):
        run_e2e_test(
            attachments=attachments,
            question=COLPALI_TEST_CONFIG["query"],
            expected_text=COLPALI_TEST_CONFIG["expected_answer"],
        )


@pytest.fixture
def test_queries():
    """Fixture providing test queries for tests."""
    return ["what is the caption of the image of butterfly?" for _ in range(16)]


@pytest.fixture
def colpali_model_resource():
    """Fixture providing ColPali model resource."""
    use_cache = (
        True  # To update the cache  test_colpali_retriever with REFRESH=true
    )

    colpali_model_resource_config = ColpaliModelResourceConfig(
        model_name="vidore/colSmol-256M",
    )
    colpali_index_config = ColpaliIndexConfig(
        image_size=512,
    )

    model_resource = CachedColpaliModelResource(
        colpali_model_resource_config, colpali_index_config, use_cache=use_cache
    )

    # Ensure colpali_index_config is set
    model_resource.colpali_index_config = colpali_index_config

    return model_resource


@pytest.fixture
def colpali_retriever(local_server, colpali_model_resource):
    """Fixture providing ColPali retriever without embeddings"""
    model, processor, device = (
        colpali_model_resource.get_model_processor_device()
    )

    retriever = ColpaliRetriever(
        document_embeddings=[],
        model=model,
        processor=processor,
        device=device,
        k=7,
        model_resource=colpali_model_resource,
    )

    return retriever


@pytest.fixture
async def test_images(local_server, colpali_model_resource):
    """Fixture providing test images extracted from PDF."""
    # Load PDF and extract images
    _, buffer, mime_type = await load_document("alps_wiki.pdf")

    # Extract images from PDF
    extracted_images = await extract_page_images(
        mime_type,
        buffer,
        {"scaled_size": colpali_model_resource.colpali_index_config.image_size},
        sys.stderr,
    )

    assert extracted_images is not None, "No images extracted from PDF"

    # Collect all images
    all_images = []
    async for image_data in extracted_images.agen:
        all_images.append(image_data)

    return all_images


# Shared functions for embedding tests
async def run_query_embedding(colpali_retriever, query, query_id):
    """Run a single query embedding."""
    embeddings = await colpali_retriever.aembed_queries([query])
    embedding_count = len(embeddings)
    return f"query_{query_id}", embedding_count


async def embed_image_batch(colpali_retriever, image_batch, task_id):
    """Run a batch of image embeddings."""

    async def image_generator():
        for image_data in image_batch:
            yield image_data

    images_gen = AsyncGeneratorWithTotal(image_generator(), len(image_batch))
    embeddings = await ColpaliRetriever.embed_images(
        colpali_retriever.model_resource,
        colpali_retriever.model_resource.colpali_index_config,
        images_gen,
        sys.stderr,
    )
    return f"image_batch_{task_id}", len(embeddings)


@pytest.mark.asyncio
async def test_colpali_parallel_queries(test_queries, colpali_retriever):
    """Test all queries in parallel"""

    # Create parallel tasks for all queries
    parallel_tasks = [
        run_query_embedding(colpali_retriever, query, i)
        for i, query in enumerate(test_queries)
    ]

    parallel_results = await asyncio.gather(*parallel_tasks)

    # Verify results
    for task_id, embedding_count in parallel_results:
        assert embedding_count > 0, f"No embeddings for query {task_id}"


@pytest.mark.asyncio
async def test_colpali_mixed_query_and_image(
    test_queries, colpali_retriever, test_images
):
    """Test mixed query and image embeddings in parallel."""

    test_images = await test_images

    assert len(test_images) >= 2, (
        f"Need at least 2 images, got {len(test_images)}"
    )

    # Split images into two halves for parallel processing
    mid_point = len(test_images) // 2
    first_half = test_images[:mid_point]
    second_half = test_images[mid_point:]

    # Create parallel tasks for both queries and images
    parallel_tasks = []

    # Add image  tasks
    parallel_tasks.append(embed_image_batch(colpali_retriever, first_half, 0))
    parallel_tasks.append(embed_image_batch(colpali_retriever, second_half, 1))
    # Add query tasks
    for i, query in enumerate(test_queries):
        parallel_tasks.append(run_query_embedding(colpali_retriever, query, i))

    parallel_results = await asyncio.gather(*parallel_tasks)

    # Verify results
    for task_id, embedding_count in parallel_results:
        if task_id.startswith("query"):
            assert embedding_count > 0, f"No embeddings for query {task_id}"
        else:
            assert embedding_count > 0, (
                f"No embeddings for image batch {task_id}"
            )


@pytest.mark.asyncio
async def test_colpali_embed_images_parallel(colpali_retriever, test_images):
    """Test parallel image embeddings using fixture."""

    test_images = await test_images

    assert len(test_images) >= 2, (
        f"Need at least 2 images, got {len(test_images)}"
    )

    # Split images into two halves for parallel processing
    mid_point = len(test_images) // 2
    first_half = test_images[:mid_point]
    second_half = test_images[mid_point:]

    parallel_tasks = [
        embed_image_batch(colpali_retriever, first_half, 0),
        embed_image_batch(colpali_retriever, second_half, 1),
    ]
    parallel_results = await asyncio.gather(*parallel_tasks)

    for task_id, embedding_count in parallel_results:
        assert embedding_count > 0, f"No embeddings for task {task_id}"
