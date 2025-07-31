import json
import re

import pytest
from fastapi.testclient import TestClient

from aidial_rag.app import create_app
from aidial_rag.app_config import AppConfig
from aidial_rag.retrieval_api import Page, RetrievalResults, Source
from tests.utils.config_override import (
    description_index_retries_override,  # noqa: F401
)
from tests.utils.e2e_decorator import e2e_test
from tests.utils.response_helpers import get_stage_names

middleware_host = "http://localhost:8081"

pytestmark = pytest.mark.usefixtures("description_index_retries_override")


@pytest.mark.asyncio
@e2e_test(filenames=["alps_wiki.html", "test_image.png"])
async def test_indexing_request(attachments):
    app = create_app(
        app_config=AppConfig(
            dial_url=middleware_host,
        )
    )
    client = TestClient(app)

    indexing_response = client.post(
        "/openai/deployments/dial-rag/chat/completions",
        headers={"Api-Key": "api-key"},
        json={
            "model": "dial-rag",
            "messages": [
                {
                    "role": "user",
                    "custom_content": {"attachments": attachments},
                }
            ],
            "custom_fields": {
                "configuration": {
                    "request": {
                        "type": "indexing",
                    }
                }
            },
        },
        timeout=60.0,
    )

    assert indexing_response.status_code == 200

    indexing_response_attachments = indexing_response.json()["choices"][0][
        "message"
    ]["custom_content"]["attachments"]
    index_attachments = [
        attachment
        for attachment in indexing_response_attachments
        if attachment["type"] == "application/x.aidial-rag-index.v0"
    ]
    assert len(index_attachments) == 2

    indexing_json = json.loads(indexing_response.text)
    indexing_stage_names = get_stage_names(indexing_json)
    assert sorted(indexing_stage_names) == [
        "Access document 'alps_wiki.html'",
        "Access document 'test_image.png'",
        "Load indexes for 'alps_wiki.html'",
        "Load indexes for 'test_image.png'",
        "Processing document 'alps_wiki.html'",
        "Processing document 'test_image.png'",
        "Store indexes for 'alps_wiki.html'",
        "Store indexes for 'test_image.png'",
    ]

    # Retrieval request with pre-computed indexes
    retrieval_response = client.post(
        "/openai/deployments/dial-rag/chat/completions",
        headers={"Api-Key": "api-key"},
        json={
            "model": "dial-rag",
            "messages": [
                {
                    "role": "user",
                    "content": "What is the shape of the infographic?",
                    "custom_content": {
                        "attachments": attachments + index_attachments
                    },
                }
            ],
            "custom_fields": {
                "configuration": {
                    "request": {
                        "type": "retrieval",
                    }
                }
            },
        },
        timeout=60.0,
    )

    assert retrieval_response.status_code == 200
    json_response = json.loads(retrieval_response.text)
    retrieval_stage_names = get_stage_names(json_response)

    # No Processing document or Store indexes stages in retrieval request
    # because the documents were already indexed in the previous request.
    assert sorted(retrieval_stage_names) == [
        "Access document 'alps_wiki.html'",
        "Access document 'test_image.png'",
        "Combined search",
        "Embeddings search",
        "Keywords search",
        "Load indexes for 'alps_wiki.html'",
        "Load indexes for 'test_image.png'",
        "Page image search",
        "Prepare indexes for search",
        "Standalone question",
    ]

    attachments = json_response["choices"][0]["message"]["custom_content"][
        "attachments"
    ]
    assert len(attachments) == 1
    attachment = attachments[0]
    assert attachment["type"] == RetrievalResults.CONTENT_TYPE
    assert attachment["title"] == "Retrieval results"

    retrieval_results = RetrievalResults.model_validate_json(attachment["data"])
    assert len(retrieval_results.chunks) == 13
    chunk_from_image = retrieval_results.chunks[0]
    assert (
        chunk_from_image.attachment_url
        == "files/6iTkeGUs2CvUehhYLmMYXB/test_image.png"
    )
    assert chunk_from_image.text == ""
    assert chunk_from_image.source == Source(
        url="files/6iTkeGUs2CvUehhYLmMYXB/test_image.png",
        display_name="test_image.png",
    )
    assert chunk_from_image.page == Page(
        number=1,
        image_index=0,
    )

    assert len(retrieval_results.images) == 1
    assert retrieval_results.images[0].mime_type == "image/png"
    assert re.match(
        r"^iVBORw0KGgoAAAA.*CYII=$", retrieval_results.images[0].data
    )

    for chunk in retrieval_results.chunks[1:]:
        assert (
            chunk.attachment_url
            == "files/6iTkeGUs2CvUehhYLmMYXB/alps_wiki.html"
        )
        assert chunk.source == Source(
            url="files/6iTkeGUs2CvUehhYLmMYXB/alps_wiki.html",
            display_name="alps_wiki.html",
        )
        assert chunk.page is None

    assert retrieval_results.chunks[1].text is not None
    assert retrieval_results.chunks[1].text.startswith(
        "Techniques and tools Quantitative Cartography"
    )


@pytest.mark.asyncio
@e2e_test(filenames=["alps_wiki.html", "test_file.csv"])
async def test_unsupported_file(attachments):
    assert len(attachments) == 2

    app = create_app(
        app_config=AppConfig(
            dial_url=middleware_host,
        )
    )
    client = TestClient(app)

    indexing_response = client.post(
        "/openai/deployments/dial-rag/chat/completions",
        headers={"Api-Key": "api-key"},
        json={
            "model": "dial-rag",
            "messages": [
                {
                    "role": "user",
                    "custom_content": {"attachments": attachments},
                }
            ],
            "custom_fields": {
                "configuration": {
                    "request": {
                        "type": "indexing",
                    }
                }
            },
        },
        timeout=60.0,
    )

    assert indexing_response.status_code == 200
    indexing_response_attachments = indexing_response.json()["choices"][0][
        "message"
    ]["custom_content"]["attachments"]

    # alps_wiki.html should be indexed successfully
    index_attachments = [
        attachment
        for attachment in indexing_response_attachments
        if attachment["type"] == "application/x.aidial-rag-index.v0"
    ]
    assert len(index_attachments) == 1
    assert index_attachments[0]["reference_url"] == attachments[0]["url"]

    indexing_result_attachment = next(
        attachment
        for attachment in indexing_response_attachments
        if attachment["type"]
        == "application/x.aidial-rag.indexing-response+json"
    )

    # test_file.csv should have an error, since it is not supported
    indexing_result_json = json.loads(indexing_result_attachment["data"])
    assert indexing_result_json == {
        "indexing_result": {
            attachments[1]["url"]: {
                "errors": [
                    {
                        "message": "Unable to load document content. Try another document format.",
                    }
                ]
            }
        },
    }


@pytest.mark.asyncio
@e2e_test(filenames=["alps_wiki.html"])
async def test_custom_index_path(attachments):
    assert len(attachments) == 1

    app = create_app(
        app_config=AppConfig(
            dial_url=middleware_host,
        )
    )
    client = TestClient(app)

    index_attachments = [
        {
            "type": "application/x.aidial-rag-index.v0",
            "url": "files/6iTkeGUs2CvUehhYLmMYXB/appdata/dial-rag/index/my_index_1",
            "reference_url": attachments[0]["url"],
        }
    ]

    indexing_response = client.post(
        "/openai/deployments/dial-rag/chat/completions",
        headers={"Api-Key": "api-key"},
        json={
            "model": "dial-rag",
            "messages": [
                {
                    "role": "user",
                    "custom_content": {
                        "attachments": attachments + index_attachments,
                    },
                }
            ],
            "custom_fields": {
                "configuration": {
                    "request": {
                        "type": "indexing",
                    }
                }
            },
        },
        timeout=60.0,
    )

    assert indexing_response.status_code == 200
    indexing_response_attachments = indexing_response.json()["choices"][0][
        "message"
    ]["custom_content"]["attachments"]
    index_attachments_result = [
        attachment
        for attachment in indexing_response_attachments
        if attachment["type"] == "application/x.aidial-rag-index.v0"
    ]
    assert len(index_attachments_result) == 1
    assert index_attachments_result == index_attachments
