from typing import Callable, List

from aidial_sdk.chat_completion import Choice, Stage
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.manager import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain.schema import BaseRetriever, Document
from pydantic import Field

from aidial_rag.document_record import DocumentRecord
from aidial_rag.utils import timed_stage


class ContentCallbackHandler(BaseCallbackHandler):
    def __init__(self, content_callback) -> None:
        super().__init__()
        self.callback = content_callback

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        if token:
            self.callback(content=token)


class RetrieverStage(BaseRetriever):
    choice: Choice
    stage_name: str
    retriever: BaseRetriever = Field(repr=False)
    document_records: List[DocumentRecord] = Field(repr=False)
    doc_to_attach: Callable[[Document, List[DocumentRecord]], dict | None]

    def _report_stage(self, stage: Stage, attached_docs: List[Document]):
        for doc in attached_docs:
            if attachment := self.doc_to_attach(doc, self.document_records):
                stage.add_attachment(**attachment)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        with timed_stage(self.choice, self.stage_name) as stage:
            attached_docs = self.retriever.get_relevant_documents(query)
            self._report_stage(stage, attached_docs)
            return attached_docs

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> List[Document]:
        with timed_stage(self.choice, self.stage_name) as stage:
            attached_docs = await self.retriever.ainvoke(query)
            self._report_stage(stage, attached_docs)
            return attached_docs
