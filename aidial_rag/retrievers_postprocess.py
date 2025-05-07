from typing import List

from langchain.schema import Document
from langchain_core.runnables import chain

from aidial_rag.documents import DocumentRecord
from aidial_rag.index_record import ChunkMetadata


def get_chunk(
    doc_records: List[DocumentRecord], chunk_metadata: ChunkMetadata
) -> Document:
    chunk_doc = (
        doc_records[chunk_metadata["doc_id"]]
        .chunks[chunk_metadata["chunk_id"]]
        .to_langchain_doc()
    )

    chunk_doc.metadata.update(chunk_metadata)
    return chunk_doc


@chain
def get_text_chunks(input: dict):
    doc_records: List[DocumentRecord] = input.get("doc_records", [])
    index_items: List[Document] = input.get("found_items", [])

    chunks_metadata = [
        ChunkMetadata(**index_item.metadata) for index_item in index_items
    ]
    result: List[Document] = [
        get_chunk(doc_records, chunk_metadata)
        for chunk_metadata in chunks_metadata
    ]
    return result
