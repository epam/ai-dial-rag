from langchain_openai.chat_models import AzureChatOpenAI
from pydantic import Field

from aidial_rag.base_config import BaseConfig
from aidial_rag.dial_config import DialConfig


class LlmConfig(BaseConfig):
    deployment_name: str = Field(
        default="gpt-4.1-2025-04-14",
        description=(
            "Used to set the deployment name of the LLM used in the chain. Could be useful "
            "if the model deployments have non-standard names in the Dial Core configuration."
        ),
    )  # model_* names are reserved in pydantic v2, so deployment_name is used instead
    max_prompt_tokens: int = Field(
        default=0,
        description=(
            "Sets `max_prompt_tokens` for the history truncation for the LLM, if history is used. "
            "Requires `DEPLOYMENT_NAME` model to support he history truncation and "
            "`max_prompt_tokens` parameter. Could be set to `0` to disable the history truncation "
            "for models which does not support it, but will cause error it if max model "
            "context window will be reached."
        ),
    )
    max_retries: int = Field(
        default=2,
        description="Sets the number of retries to send the request to the LLM.",
    )
    temperature: float = Field(
        default=0.0,
        description=(
            "Sets the temperature for the LLM, controlling the randomness of the output. "
            "Higher values (e.g., 1.0) make the output more random, while lower values (e.g., 0.0) "
            "make it more deterministic."
        ),
    )


def create_llm(dial_config: DialConfig, llm_config: LlmConfig):
    extra_body = {}
    if llm_config.max_prompt_tokens:
        extra_body["max_prompt_tokens"] = llm_config.max_prompt_tokens

    llm = AzureChatOpenAI(
        azure_endpoint=dial_config.dial_url,
        api_key=dial_config.api_key,
        model=llm_config.deployment_name,
        api_version="2023-03-15-preview",
        openai_api_type="azure",
        temperature=llm_config.temperature,
        streaming=True,
        max_retries=llm_config.max_retries,
        extra_body=extra_body,
    )
    return llm
