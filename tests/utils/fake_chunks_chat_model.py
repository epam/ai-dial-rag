from typing import Any, AsyncIterator, Iterator, List, Optional

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import SimpleChatModel
from langchain_core.messages import AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGenerationChunk


class FakeChunksChatModel(SimpleChatModel):
    """Fake chat model that streams a fixed list of chunks verbatim.

    Unlike ``langchain_core``'s ``FakeListChatModel`` (which splits a single
    string character by character), this model lets tests control the exact
    chunk boundaries emitted while streaming.
    """

    chunks: List[str]

    @property
    def _llm_type(self) -> str:
        return "fake-chunks-chat-model"

    def _call(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        return "".join(self.chunks)

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        for chunk in self.chunks:
            yield ChatGenerationChunk(message=AIMessageChunk(content=chunk))

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        for chunk in self.chunks:
            yield ChatGenerationChunk(message=AIMessageChunk(content=chunk))
