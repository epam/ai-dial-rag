from typing import Any

import httpx
from langchain_openai.chat_models import AzureChatOpenAI
from pydantic import SecretStr


def create_mock_azure_chat_openai(response_json: Any) -> AzureChatOpenAI:
    mock_transport = httpx.MockTransport(
        lambda _: httpx.Response(
            200,
            json=response_json,
        )
    )

    llm = AzureChatOpenAI(
        azure_endpoint="http://mock_endpoint",
        api_key=SecretStr("mock_api_key"),
        model="mock_model",
        api_version="2023-03-15-preview",
        openai_api_type="azure",
        temperature=0,
        streaming=False,
        max_retries=1,
        http_client=httpx.Client(transport=mock_transport),
        http_async_client=httpx.AsyncClient(transport=mock_transport),
    )
    return llm
