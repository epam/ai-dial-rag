from enum import StrEnum
from typing import List, TypedDict

from docarray import BaseDoc
from docarray.typing import ID
from langchain.schema import Document


class IndexItem(BaseDoc):
    id: ID | None = None  # Disable random ID generation for performance reasons
    chunk_index: int  # TODO: id?


class TextIndexItem(IndexItem):
    tokenized_text: List[str]


class RetrievalType(StrEnum):
    TEXT = "text"
    IMAGE = "image"


class ChunkMetadata(TypedDict):
    doc_id: int
    chunk_id: int
    retrieval_type: RetrievalType


def to_metadata_doc(
    doc_id: int, chunk_id: int, retrieval_type: RetrievalType
) -> Document:
    return Document(
        # EnsembleRetriever uses page_content as the key for the document
        page_content=f"{doc_id}_{chunk_id}",
        metadata=ChunkMetadata(
            doc_id=doc_id, chunk_id=chunk_id, retrieval_type=retrieval_type
        ),
    )
