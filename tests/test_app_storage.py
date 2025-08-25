import json

import pytest
from fastapi.testclient import TestClient

from aidial_rag.app import create_app
from aidial_rag.app_config import AppConfig
from tests.utils.config_override import (
    description_index_retries_override,  # noqa: F401
)
from tests.utils.e2e_decorator import e2e_test
from tests.utils.response_helpers import get_stage_names

middleware_host = "http://localhost:8081"

pytestmark = pytest.mark.usefixtures("description_index_retries_override")


@pytest.mark.asyncio
@e2e_test(filenames=["alps_wiki.html"])
async def test_retrieval_request(attachments):
    app = create_app(
        app_config=AppConfig(
            dial_url=middleware_host,
        )
    )
    client = TestClient(app)

    response1 = client.post(
        "/openai/deployments/dial-rag/chat/completions",
        headers={"Api-Key": "api-key"},
        json={
            "model": "dial-rag",
            "messages": [
                {
                    "role": "user",
                    "content": "What is Alps?",
                    "custom_content": {"attachments": attachments},
                }
            ],
        },
        timeout=60.0,
    )

    assert response1.status_code == 200
    json_response1 = json.loads(response1.text)

    assert (
        "mountain range" in json_response1["choices"][0]["message"]["content"]
    )

    assert sorted(get_stage_names(json_response1)) == [
        "Access document 'alps_wiki.html'",
        "Combined search",
        "Embeddings search",
        "Keywords search",
        "Load indexes for 'alps_wiki.html'",
        "Prepare indexes for search",
        "Processing document 'alps_wiki.html'",
        "Standalone question",
        "Store indexes for 'alps_wiki.html'",
    ]

    # Second request with the same document
    response2 = client.post(
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
        },
        timeout=60.0,
    )
    assert response2.status_code == 200
    json_response2 = json.loads(response2.text)

    assert "Mont Blanc" in json_response2["choices"][0]["message"]["content"]

    # No Processing and Storing indexes for the second request
    # because RAG uses cached indexes
    assert sorted(get_stage_names(json_response2)) == [
        "Access document 'alps_wiki.html'",
        "Combined search",
        "Embeddings search",
        "Keywords search",
        "Load indexes for 'alps_wiki.html'",
        "Prepare indexes for search",
        "Standalone question",
    ]
