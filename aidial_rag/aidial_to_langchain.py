from typing import Iterable, List

from aidial_sdk.chat_completion import Message
from langchain_core.messages import BaseMessage
from langchain_core.messages.utils import convert_to_messages


def to_langchain_messages(messages: Iterable[Message]) -> List[BaseMessage]:
    return convert_to_messages(
        [(message.role.value, str(message.content)) for message in messages]
    )
