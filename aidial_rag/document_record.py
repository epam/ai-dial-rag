import logging
from typing import List, TypeAlias

from docarray import BaseDoc, DocList
from docarray.typing import ID, NdArray
from langchain.schema import Document

from aidial_rag.index_record import TextIndexItem

# The format version should be incremented on every incompatible change
# to the DocumentRecord structure or serialization method.
# Try to keep backward compatibility when possible (e.g. by adding new fields
# with default values) so that we don't need to re-index all documents on
# every change.
# See tests/test_doc_record.py for tests that ensure backward compatibility.
FORMAT_VERSION: int = 13

SERIALIZATION_CONFIG = {"protocol": "protobuf", "compress": "gzip"}

logger = logging.getLogger(__name__)


class Chunk(BaseDoc):
    id: ID | None = None  # Disable random ID generation for performance reasons
    text: str
    metadata: dict

    def to_langchain_doc(self) -> Document:
        return Document(
            page_content=self.text,
            metadata=self.metadata,
        )


class IndexSettings(BaseDoc):
    id: ID | None = None  # Disable random ID generation for performance reasons
    indexes: dict = {}


class ItemEmbeddings(BaseDoc):
    """Represents a list of embeddings associated with a single item (chunk or page)."""

    id: ID | None = None  # Disable random ID generation for performance reasons
    embeddings: NdArray["i", "x"]  # type: ignore # noqa: F821


MultiEmbeddings: TypeAlias = DocList[ItemEmbeddings]


class ModificationMetadata(BaseDoc):
    id: ID | None = None  # Disable random ID generation for performance reasons
    last_modified: int | None = None
    etag: str | None = None


# Note: the default values should be added for new fields to keep backward compatibility
class DocumentRecord(BaseDoc):
    format_version: int | None
    index_settings: IndexSettings
    id: ID | None = None  # Disable random ID generation for performance reasons
    chunks: DocList[Chunk]
    text_index: DocList[TextIndexItem] | None
    embeddings_index: MultiEmbeddings | None
    multimodal_embeddings_index: MultiEmbeddings | None = None
    description_embeddings_index: MultiEmbeddings | None = None
    colpali_embeddings_index: MultiEmbeddings | None = None
    mime_type: str
    document_bytes: bytes  # Could be attached document or converted document
    modification_metadata: ModificationMetadata = ModificationMetadata()


async def build_chunks_list(chunk_docs: List[Document]) -> DocList:
    logger.debug("Cutting text chunks.")
    chunks = DocList(
        [
            Chunk(
                text=chunk.page_content,
                metadata=chunk.metadata,
            )
            for chunk in chunk_docs
        ]
    )

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i

    return chunks
