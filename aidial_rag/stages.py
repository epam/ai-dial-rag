from typing import Callable, List

from aidial_sdk.chat_completion import Choice, Stage
from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field

from aidial_rag.document_record import DocumentRecord
from aidial_rag.utils import timed_stage


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
        raise NotImplementedError(
            "RetrieverStage only supports async retrieval."
        )

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> List[Document]:
        with timed_stage(self.choice, self.stage_name) as stage:
            attached_docs = await self.retriever.ainvoke(query)
            self._report_stage(stage, attached_docs)
            return attached_docs
