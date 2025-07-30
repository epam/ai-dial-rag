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
        assert chunk.attachment_url == "files/6iTkeGUs2CvUehhYLmMYXB/alps_wiki.html"
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

    # TODO: Add machine-readable per-document errors
    assert response.status_code == 400
    json_response = json.loads(response.text)
    assert json_response["error"]["message"] == (
        "I'm sorry, but I can't process the documents because of the following errors:\n\n"
        "|Document|Error|\n"
        "|---|---|\n"
        "|test_file.csv|Unable to load document content. Try another document format.|\n\n"
        "Please try again with different documents."
    )
