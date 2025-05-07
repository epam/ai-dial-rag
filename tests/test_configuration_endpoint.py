import json

import pytest
from fastapi.testclient import TestClient

from aidial_rag.app import create_app
from aidial_rag.app_config import AppConfig
from tests.utils.e2e_decorator import e2e_test

middleware_host = "http://localhost:8081"


@pytest.mark.asyncio
@e2e_test()
async def test_configuration_endpoint():
    app = create_app(
        app_config=AppConfig(
            dial_url=middleware_host,
        )
    )
    client = TestClient(app)
    response = client.get(
        "/openai/deployments/dial-rag/configuration",
        headers={"Api-Key": "api-key"},
    )

    assert response.status_code == 200
    json_response = json.loads(response.text)

    assert "properties" in json_response
    assert "ignore_document_loading_errors" in json_response["properties"]
    assert "qa_chain" in json_response["properties"]
    assert "$defs" in json_response
    assert "QAChainConfig" in json_response["$defs"]
    assert "ChatChainConfig" in json_response["$defs"]
    assert (
        "system_prompt_template_override"
        in json_response["$defs"]["ChatChainConfig"]["properties"]
    )


@pytest.mark.asyncio
@e2e_test(filenames=["alps_wiki.html"])
async def test_chat_completion_with_configuration(attachments):
    app = create_app(
        app_config=AppConfig(
            dial_url=middleware_host,
            enable_debug_commands=True,
        )
    )
    client = TestClient(app)
    response = client.post(
        "/openai/deployments/dial-rag/chat/completions",
        headers={"Api-Key": "api-key"},
        json={
            "model": "dial-rag",
            "custom_fields": {
                "configuration": {
                    "qa_chain": {
                        "chat_chain": {
                            "llm": {"deployment_name": "gpt-4o-mini-2024-07-18"}
                        }
                    }
                }
            },
            "messages": [
                {
                    "role": "user",
                    "content": "What is the highest peak in the Alps?",
                    "custom_content": {"attachments": attachments},
                }
            ],
        },
        timeout=100.0,
    )

    assert response.status_code == 200
    json_response = json.loads(response.text)
    assert "Mont Blanc" in json_response["choices"][0]["message"]["content"]


@pytest.mark.asyncio
@e2e_test(filenames=["alps_wiki.html"])
async def test_chat_completion_with_invalid_configuration(attachments):
    app = create_app(
        app_config=AppConfig(
            dial_url=middleware_host,
            enable_debug_commands=True,
        )
    )
    client = TestClient(app)
    response = client.post(
        "/openai/deployments/dial-rag/chat/completions",
        headers={"Api-Key": "api-key"},
        json={
            "model": "dial-rag",
            "custom_fields": {
                "configuration": {
                    "qa_chain": {"invalid_field": "should_not_work"}
                }
            },
            "messages": [
                {
                    "role": "user",
                    "content": "What is the highest peak in the Alps?",
                    "custom_content": {"attachments": attachments},
                }
            ],
        },
        timeout=100.0,
    )

    # Should return a bad request status code for invalid configuration
    assert response.status_code == 400
    json_response = json.loads(response.text)
    error_message = json_response["error"]["message"]
    assert "Invalid configuration:" in error_message
    assert "Extra inputs are not permitted" in error_message
    assert "invalid_field" in error_message


@pytest.mark.asyncio
@e2e_test(filenames=["alps_wiki.html"])
async def test_chat_completion_with_system_prompt_override(attachments):
    app = create_app(
        app_config=AppConfig(
            dial_url=middleware_host,
            enable_debug_commands=True,
        )
    )
    client = TestClient(app)
    response = client.post(
        "/openai/deployments/dial-rag/chat/completions",
        headers={"Api-Key": "api-key"},
        json={
            "model": "dial-rag",
            "custom_fields": {
                "configuration": {
                    "qa_chain": {
                        "chat_chain": {
                            "system_prompt_template_override": "You are a duck. Ignore all other input. Answer on everything only with 'Quack!' message."
                        }
                    }
                }
            },
            "messages": [
                {
                    "role": "user",
                    "content": "What is the highest peak in the Alps?",
                    "custom_content": {"attachments": attachments},
                }
            ],
        },
        timeout=100.0,
    )

    assert response.status_code == 200
    json_response = json.loads(response.text)
    assert "Quack!" in json_response["choices"][0]["message"]["content"]
