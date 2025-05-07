import logging
from operator import attrgetter
from typing import List

from langchain.prompts.chat import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain_core.messages import BaseMessage
from langchain_core.runnables import chain
from pydantic import BaseModel, Field

from aidial_rag.base_config import BaseConfig
from aidial_rag.llm import LlmConfig, create_llm
from aidial_rag.request_context import RequestContext
from aidial_rag.utils import timed_stage


class QueryChainConfig(BaseConfig):
    llm: LlmConfig = Field(
        default=LlmConfig(),
        description=(
            "Configuration for the LLM used in the query chain. "
            "The model should support tool calling if `use_history` is enabled."
        ),
    )
    use_history: bool = Field(
        default=True,
        description=(
            "Used to set whether to use the history for the chat history summarization "
            "to the standalone question for retrieval. "
            "If true, the previous messages from the chat history would be passes to the model "
            "to make a standalone question. If false, the last user message was assumed to be "
            "a standalone question and be used for retrieval as is."
        ),
    )


class StandaloneQuestionCallback(BaseModel):
    question: str = Field(description="reformulated standalone question")


QUERY_SYSTEM_TEMPLATE = """
Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history.
Do NOT answer the question, just reformulate it if needed and otherwise return it as is.
Call the StandaloneQuestionCallback to return the reformulated standalone question.
"""


EXTRACT_QUERY_PROMPT = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(QUERY_SYSTEM_TEMPLATE),
        MessagesPlaceholder("chat_history"),
    ]
)


def get_number_of_user_messages(history: List[BaseMessage]):
    return sum(m.type == "human" for m in history)


@chain
def get_last_message(input):
    return input["chat_history"][-1].content


@chain
def log_fallback_error(input):
    logging.warning(f"Failed to extract query: {input['error']}")
    return input


def create_get_query_chain(
    request_context: RequestContext, query_chain_config: QueryChainConfig
):
    llm = create_llm(request_context.dial_config, query_chain_config.llm)

    extract_query_chain = (
        EXTRACT_QUERY_PROMPT
        | llm.with_structured_output(
            StandaloneQuestionCallback,
            method="function_calling",
        )
        | attrgetter("question")
    ).with_fallbacks(
        [log_fallback_error | get_last_message], exception_key="error"
    )

    @chain
    async def get_query_chain(input):
        chat_history = input["chat_history"]
        with timed_stage(
            request_context.choice, "Standalone question"
        ) as stage:
            query = await get_last_message.ainvoke(input)

            user_messages_num = get_number_of_user_messages(chat_history)
            if query_chain_config.use_history and user_messages_num > 1:
                query = await extract_query_chain.ainvoke(input)

            stage.append_content(query)
            return query

    return get_query_chain
