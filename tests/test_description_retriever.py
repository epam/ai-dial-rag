import json

import pytest
from fastapi.testclient import TestClient

from aidial_rag.app import create_app
from aidial_rag.app_config import AppConfig
from aidial_rag.retrievers.description_retriever.description_retriever import (
    _get_page_description,
)
from aidial_rag.retrievers.description_retriever.page_description import (
    ImageDescription,
    PageDescription,
    TableDescription,
)
from tests.utils.config_override import (
    description_index_retries_override,  # noqa: F401
)
from tests.utils.e2e_decorator import e2e_test
from tests.utils.mock_llm import create_mock_azure_chat_openai

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


@pytest.mark.asyncio
async def test_get_page_description():
    llm = create_mock_azure_chat_openai(
        response_json={
            "id": "1f415114-590c-418c-a848-f7d5ad48eb8e",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "toolu_bdrk_01ET211yG1RVHeXM1B5szG1T",
                                "type": "function",
                                "function": {
                                    "name": "PageDescription",
                                    "arguments": json.dumps(
                                        {
                                            "page_summary": "This is a sample page summary.",
                                            "keyfact": "This is a key fact from the page.",
                                            "images": [
                                                {
                                                    "image_summary": "This is a sample image summary.",
                                                    "keyfact": "This is a key fact from the image.",
                                                }
                                            ],
                                            "tables": [
                                                {
                                                    "table_summary": "This is a sample table summary.",
                                                    "keyfact": "This is a key fact from the table.",
                                                }
                                            ],
                                        }
                                    ),
                                },
                            }
                        ],
                    },
                }
            ],
            "created": 1755026602,
            "model": "mock_model",
            "object": "chat.completion.chunk",
            "usage": {
                "completion_tokens": 479,
                "prompt_tokens": 1372,
                "total_tokens": 1851,
                "prompt_tokens_details": {"cached_tokens": 0},
            },
        }
    )

    page_description = await _get_page_description(
        llm=llm,
        page_bitmap_base64="mock_base64_image_data",
    )

    assert page_description == PageDescription(
        page_summary="This is a sample page summary.",
        keyfact="This is a key fact from the page.",
        images=[
            ImageDescription(
                image_summary="This is a sample image summary.",
                keyfact="This is a key fact from the image.",
            )
        ],
        tables=[
            TableDescription(
                table_summary="This is a sample table summary.",
                keyfact="This is a key fact from the table.",
            )
        ],
    )


@pytest.mark.asyncio
async def test_get_page_description_missing_lists():
    llm = create_mock_azure_chat_openai(
        response_json={
            "id": "1f415114-590c-418c-a848-f7d5ad48eb8e",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "toolu_bdrk_01ET211yG1RVHeXM1B5szG1T",
                                "type": "function",
                                "function": {
                                    "name": "PageDescription",
                                    "arguments": json.dumps(
                                        {
                                            "page_summary": "This is a sample page summary.",
                                            "keyfact": "This is a key fact from the page.",
                                        }
                                    ),
                                },
                            }
                        ],
                    },
                }
            ],
            "created": 1755026602,
            "model": "mock_model",
            "object": "chat.completion.chunk",
            "usage": {
                "completion_tokens": 479,
                "prompt_tokens": 1372,
                "total_tokens": 1851,
                "prompt_tokens_details": {"cached_tokens": 0},
            },
        }
    )

    page_description = await _get_page_description(
        llm=llm,
        page_bitmap_base64="mock_base64_image_data",
    )

    assert page_description == PageDescription(
        page_summary="This is a sample page summary.",
        keyfact="This is a key fact from the page.",
        images=[],
        tables=[],
    )


@pytest.mark.asyncio
async def test_get_page_description_broken_response():
    llm = create_mock_azure_chat_openai(
        response_json={
            "id": "1f415114-590c-418c-a848-f7d5ad48eb8e",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "toolu_bdrk_01ET211yG1RVHeXM1B5szG1T",
                                "type": "function",
                                "function": {
                                    "name": "PageDescription",
                                    "arguments": '{"page_summary": "This is a sample page summary.", "keyfact": "This is a key fact from the page.</anyfact>\\n<parameter name=\\"tables\\">[{\\"table_summary\\": \\"Table 1.1 - Specifications\\", \\"keyfact\\": \\"Some keyfact from the table.\\"}]"}',
                                },
                            }
                        ],
                    },
                }
            ],
            "created": 1755026602,
            "model": "mock_model",
            "object": "chat.completion.chunk",
            "usage": {
                "completion_tokens": 479,
                "prompt_tokens": 1372,
                "total_tokens": 1851,
                "prompt_tokens_details": {"cached_tokens": 0},
            },
        }
    )

    page_description = await _get_page_description(
        llm=llm,
        page_bitmap_base64="mock_base64_image_data",
    )

    assert page_description == PageDescription(
        page_summary="This is a sample page summary.",
        keyfact='This is a key fact from the page.</anyfact>\n<parameter name="tables">[{"table_summary": "Table 1.1 - Specifications", "keyfact": "Some keyfact from the table."}]',
        images=[],
        tables=[],
    )


@pytest.mark.asyncio
async def test_get_page_description_broken_response2():
    llm = create_mock_azure_chat_openai(
        response_json={
            "id": "1f415114-590c-418c-a848-f7d5ad48eb8e",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "toolu_bdrk_01ET211yG1RVHeXM1B5szG1T",
                                "type": "function",
                                "function": {
                                    "name": "PageDescription",
                                    "arguments": '{"page_summary": "This is a sample page summary.", "keyfact": "This is a key fact from the page.", "tables": [{"table_summary": "Table 1.3: The table contains three data rows showing values like "A", "B", and "C".", "keyfact": "The table documents different configurations..."}]}',
                                },
                            }
                        ],
                    },
                }
            ],
            "created": 1755026602,
            "model": "mock_model",
            "object": "chat.completion.chunk",
            "usage": {
                "completion_tokens": 479,
                "prompt_tokens": 1372,
                "total_tokens": 1851,
                "prompt_tokens_details": {"cached_tokens": 0},
            },
        }
    )

    page_description = await _get_page_description(
        llm=llm,
        page_bitmap_base64="mock_base64_image_data",
    )

    assert page_description == PageDescription(
        page_summary='{"page_summary": "This is a sample page summary.", "keyfact": "This is a key fact from the page.", "tables": [{"table_summary": "Table 1.3: The table contains three data rows showing values like "A", "B", and "C".", "keyfact": "The table documents different configurations..."}]}',
        keyfact="",
        images=[],
        tables=[],
    )
