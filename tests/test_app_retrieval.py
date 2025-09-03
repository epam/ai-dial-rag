import json
import re

import pytest
from fastapi.testclient import TestClient

from aidial_rag.app import create_app
from aidial_rag.app_config import AppConfig
from aidial_rag.retrieval_api import Page, RetrievalResponse, Source
from tests.utils.config_override import (
    description_index_retries_override,  # noqa: F401
)
from tests.utils.e2e_decorator import e2e_test
from tests.utils.response_helpers import (
    get_attachments,
    get_retrieval_response_json,
)

middleware_host = "http://localhost:8081"

pytestmark = pytest.mark.usefixtures("description_index_retries_override")


@pytest.mark.asyncio
@e2e_test(filenames=["alps_wiki.html", "test_image.png"])
async def test_retrieval_request(attachments):
    app = create_app(
        app_config=AppConfig(
            dial_url=middleware_host,
        )
    )
    client = TestClient(app)

    response = client.post(
        "/openai/deployments/dial-rag/chat/completions",
        headers={"Api-Key": "api-key"},
        json={
            "model": "dial-rag",
            "messages": [
                {
                    "role": "user",
                    "content": "What is the shape of the infographic?",
                    "custom_content": {"attachments": attachments},
                }
            ],
            "custom_fields": {
                "configuration": {
                    "request": {
                        "type": "retrieval",
                        "allow_indexing": True,
                    }
                }
            },
        },
        timeout=60.0,
    )

    assert response.status_code == 200
    json_response = json.loads(response.text)
    attachments = json_response["choices"][0]["message"]["custom_content"][
        "attachments"
    ]
    assert len(attachments) == 1
    attachment = attachments[0]
    assert attachment["type"] == RetrievalResponse.CONTENT_TYPE
    assert attachment["title"] == "Retrieval response"

    retrieval_response = RetrievalResponse.model_validate_json(
        attachment["data"]
    )
    assert len(retrieval_response.chunks) == 13
    chunk_from_image = retrieval_response.chunks[0]
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

    assert len(retrieval_response.images) == 1
    assert retrieval_response.images[0].mime_type == "image/png"
    assert re.match(
        r"^iVBORw0KGgoAAAA.*CYII=$", retrieval_response.images[0].data
    )

    for chunk in retrieval_response.chunks[1:]:
        assert (
            chunk.attachment_url
            == "files/6iTkeGUs2CvUehhYLmMYXB/alps_wiki.html"
        )
        assert chunk.source == Source(
            url="files/6iTkeGUs2CvUehhYLmMYXB/alps_wiki.html",
            display_name="alps_wiki.html",
        )
        assert chunk.page is None

    assert retrieval_response.chunks[1].text is not None
    assert retrieval_response.chunks[1].text.startswith(
        "Techniques and tools Quantitative Cartography"
    )


@pytest.mark.asyncio
@e2e_test(filenames=["test_file.csv"])
async def test_retrieval_request_with_unsupported_csv(attachments):
    app = create_app(
        app_config=AppConfig(
            dial_url=middleware_host,
        )
    )
    client = TestClient(app)

    response = client.post(
        "/openai/deployments/dial-rag/chat/completions",
        headers={"Api-Key": "api-key"},
        json={
            "model": "dial-rag",
            "messages": [
                {
                    "role": "user",
                    "content": "What is this csv about?",
                    "custom_content": {"attachments": attachments},
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

    assert response.status_code == 200
    retrieval_response_attachments = get_attachments(response.json())
    retrieval_response_json = get_retrieval_response_json(
        retrieval_response_attachments
    )
    assert retrieval_response_json["indexing_results"] == {
        attachments[0]["url"]: {
            "errors": [
                {
                    "message": "Unable to load document content. Try another document format.",
                }
            ]
        }
    }


@pytest.mark.asyncio
@e2e_test(filenames=["alps_wiki.html"])
async def test_retrieval_request_with_disabled_indexing(attachments):
    app = create_app(
        app_config=AppConfig(
            dial_url=middleware_host,
        )
    )
    client = TestClient(app)

    response = client.post(
        "/openai/deployments/dial-rag/chat/completions",
        headers={"Api-Key": "api-key"},
        json={
            "model": "dial-rag",
            "messages": [
                {
                    "role": "user",
                    "content": "What is the highest peak in the Alps?",
                    "custom_content": {"attachments": attachments},
                }
            ],
            "custom_fields": {
                "configuration": {
                    "request": {
                        "type": "retrieval",
                        "allow_indexing": False,
                    }
                }
            },
        },
        timeout=60.0,
    )

    assert response.status_code == 200
    retrieval_response_attachments = get_attachments(response.json())
    retrieval_response_json = get_retrieval_response_json(
        retrieval_response_attachments
    )
    assert retrieval_response_json["indexing_results"] == {
        attachments[0]["url"]: {
            "errors": [
                {
                    "message": "Index is missing.",
                    "type": "index_missing",
                }
            ]
        }
    }
