from pydantic import Field

from aidial_rag.base_config import BaseConfig
from aidial_rag.llm import LlmConfig
from aidial_rag.query_chain import QueryChainConfig


class ChatChainConfig(BaseConfig):
    llm: LlmConfig = Field(
        default=LlmConfig(),
        description=(
            "Configuration for the LLM used in the chat chain. "
            "The model should support vision if `num_page_images_to_use` is greater than 0."
        ),
    )
    system_prompt_template_override: str | None = Field(
        default=None,
        description="Allow to override the system prompt template.",
    )
    use_history: bool = Field(
        default=True,
        description=(
            "Used to set whether to use the history for the answer generation. "
            "If true, the previous messages from the chat history would be passes to the model. "
            "If false, only the query (last user message or standalone question, depending on the "
            "`query_chain` settings) will be passed to the model for the answer generation."
        ),
    )
    num_page_images_to_use: int = Field(
        default=4,
        description=(
            "Sets number of page images to pass to the model for the answer generation. "
            "If is greater that 0, the model in `llm.deployment_name` should accept images "
            "in the user messages. Could be set to 0 (together with USE_MULTIMODAL_INDEX=False "
            "and USE_DESCRIPTION_INDEX=False) for text-only RAG."
        ),
    )
    page_image_size: int = Field(
        default=1536,
        description="Sets the size of the page images to pass to the model for the answer generation.",
    )


class QAChainConfig(BaseConfig):
    chat_chain: ChatChainConfig = Field(
        default=ChatChainConfig(),
        description=(
            "Configuration for the chat chain which generates the answer for the user question "
            "based on the retrieved context."
        ),
    )
    query_chain: QueryChainConfig = Field(
        default=QueryChainConfig(),
        description=(
            "Configuration for the query chain which reformulates the user question to standalone "
            "question for the retrieval based on the chat history."
        ),
    )
