import re
from typing import List

from aidial_sdk.chat_completion import Message
from langchain_core.messages import AIMessage, BaseMessage

from aidial_rag.aidial_to_langchain import to_langchain_messages

REF_HISTORY_PATTERN = re.compile(r"\[(\d+)\]")


def transform_history_message(message: BaseMessage) -> BaseMessage:
    if isinstance(message, AIMessage) and message.content:
        # Restore the references to <[num]> in the assistant messages, because
        # the model may be confused if the format is different from the prompt
        return AIMessage(
            content=REF_HISTORY_PATTERN.sub(r"<[\1]>", str(message.content))
        )

    return message


def transform_history(messages: List[Message]) -> List[BaseMessage]:
    # We may have empty messages after command processing
    # Some models (like claude) do not support empty messages
    return [
        transform_history_message(message)
        for message in to_langchain_messages(messages)
        if message.content
    ]
