import pytest
from fastapi.testclient import TestClient

from aidial_rag.app import create_app
from aidial_rag.app_config import AppConfig
from tests.utils.config_override import (
    description_index_retries_override,  # noqa: F401
)
from tests.utils.e2e_decorator import e2e_test

MIDDLEWARE_HOST = "http://localhost:8081"

pytestmark = pytest.mark.usefixtures("description_index_retries_override")


@pytest.mark.asyncio
@e2e_test(filenames=["alps_wiki.pdf"])
async def test_description_retriever_azure(attachments):
    app = create_app(
        app_config=AppConfig(
            dial_url=MIDDLEWARE_HOST,
            config_path="config/azure_description.yaml",
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
                    "content": "At what page there is an image of butterfly?",
                    "custom_content": {"attachments": attachments},
                }
            ],
        },
        timeout=100.0,
    )
    assert response.status_code == 200

    json_response = response.json()
    assert "page 13" in json_response["choices"][0]["message"]["content"]


@pytest.mark.asyncio
@e2e_test(filenames=["alps_wiki.pdf"])
async def test_description_retriever_gcp(attachments):
    app = create_app(
        app_config=AppConfig(
            dial_url=MIDDLEWARE_HOST,
            config_path="config/gcp_description.yaml",
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
                    "content": "At what page there is an image of butterfly?",
                    "custom_content": {"attachments": attachments},
                }
            ],
        },
        timeout=100.0,
    )
    assert response.status_code == 200

    json_response = response.json()
    assert "page 13" in json_response["choices"][0]["message"]["content"]


@pytest.mark.asyncio
@e2e_test(filenames=["alps_wiki.pdf"])
async def test_description_retriever_aws(attachments):
    app = create_app(
        app_config=AppConfig(
            dial_url=MIDDLEWARE_HOST,
            config_path="config/aws_description.yaml",
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
                    "content": "At what page there is an image of butterfly?",
                    "custom_content": {"attachments": attachments},
                }
            ],
        },
        timeout=100.0,
    )
    assert response.status_code == 200

    json_response = response.json()
    assert "page 13" in json_response["choices"][0]["message"]["content"]


@pytest.mark.asyncio
@e2e_test(filenames=["alps_wiki.pdf"])
async def test_description_retriever_gpt5(attachments):
    app = create_app(
        app_config=AppConfig(
            dial_url=MIDDLEWARE_HOST,
            config_path="config/azure_description.yaml",
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
                    "content": "At what page there is an image of butterfly?",
                    "custom_content": {"attachments": attachments},
                }
            ],
            "custom_fields": {
                "configuration": {
                    "indexing": {
                        "description_index": {
                            "llm": {
                                "deployment_name": "gpt-5-mini-2025-08-07",
                                "temperature": 1.0,
                            }
                        }
                    },
                    "qa_chain": {
                        "chat_chain": {
                            "llm": {
                                "deployment_name": "gpt-5-2025-08-07",
                                "temperature": 1.0,
                            }
                        }
                    },
                }
            },
        },
        timeout=100.0,
    )
    assert response.status_code == 200

    json_response = response.json()
    assert "Page 13" in json_response["choices"][0]["message"]["content"]
