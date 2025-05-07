from typing import ClassVar, List

from langchain.schema import BaseRetriever, Document

from aidial_rag.document_record import Chunk, DocumentRecord
from aidial_rag.index_record import RetrievalType, to_metadata_doc
from aidial_rag.qa_chain import format_attributes


class AllDocumentsRetriever(BaseRetriever):
    _MAX_LENGTH_IN_BYTES: ClassVar[int] = 12000
    _CHUNK_PROMPT_OVERHEAD: ClassVar[int] = 30

    metadata_chunks: List[Document]

    @staticmethod
    def _estimated_size(i: int, chunk: Chunk) -> int:
        # For small chunks the overhead for the document name or file path is significant
        # So we have to include the size of the metadata in the size estimation
        return (
            len(chunk.text)
            + len(format_attributes(i, chunk.metadata))
            + AllDocumentsRetriever._CHUNK_PROMPT_OVERHEAD
        )

    @staticmethod
    def is_within_limit(document_records: List[DocumentRecord]) -> bool:
        total_length = sum(
            AllDocumentsRetriever._estimated_size(i, chunk)
            for i, chunk in enumerate(
                chunk for doc in document_records for chunk in doc.chunks
            )
        )
        return total_length <= AllDocumentsRetriever._MAX_LENGTH_IN_BYTES

    @classmethod
    def from_doc_records(
        cls,
        document_records: List[DocumentRecord] | None = None,
    ) -> "AllDocumentsRetriever":
        if document_records is None:
            document_records = []

        metadata_chunks = [
            to_metadata_doc(i, j, RetrievalType.TEXT)
            for i, doc in enumerate(document_records)
            for j in range(len(doc.chunks))
        ]

        return cls(
            metadata_chunks=metadata_chunks,
        )

    def _get_relevant_documents(self, query: str, *args, **kwargs):
        return self.metadata_chunks

    async def _aget_relevant_documents(self, query: str, *args, **kwargs):
        return self.metadata_chunks
