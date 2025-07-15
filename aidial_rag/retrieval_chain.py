from itertools import groupby
from operator import itemgetter
from typing import List

from langchain.schema import Document
from langchain_core.runnables import chain

from aidial_rag.document_record import DocumentRecord
from aidial_rag.image_processor.base64 import pil_image_as_base64
from aidial_rag.image_processor.extract_pages import (
    are_image_pages_supported,
    extract_pages_gen,
)
from aidial_rag.index_record import ChunkMetadata, RetrievalType
from aidial_rag.qa_chain_config import ChatChainConfig


def collect_pages_with_images(
    doc_records: List[DocumentRecord], chunks_metadatas
):
    # RetrievalType.IMAGE has higher priority
    for chunk_metadata in chunks_metadatas:
        doc_record = doc_records[chunk_metadata["doc_id"]]
        if not are_image_pages_supported(doc_record.mime_type):
            continue
        chunk = doc_record.chunks[chunk_metadata["chunk_id"]]
        if (
            chunk_metadata["retrieval_type"] == RetrievalType.IMAGE
            and "page_number" in chunk.metadata
        ):
            yield (chunk_metadata["doc_id"], chunk.metadata["page_number"])

    for chunk_metadata in chunks_metadatas:
        doc_record = doc_records[chunk_metadata["doc_id"]]
        if not are_image_pages_supported(doc_record.mime_type):
            continue
        chunk = doc_record.chunks[chunk_metadata["chunk_id"]]
        if (
            chunk_metadata["retrieval_type"] != RetrievalType.IMAGE
            and "page_number" in chunk.metadata
        ):
            yield (chunk_metadata["doc_id"], chunk.metadata["page_number"])


async def make_image_by_page(
    doc_records: List[DocumentRecord],
    chunks_metadatas,
    num_pages_to_use: int,
    page_image_size: int,
) -> dict:
    required_pages = set()
    for doc_id, page_number in collect_pages_with_images(
        doc_records, chunks_metadatas
    ):
        if len(required_pages) >= num_pages_to_use:
            break
        required_pages.add((doc_id, page_number))

    image_by_page = {}
    for doc_id, pages_iter in groupby(sorted(required_pages), itemgetter(0)):
        page_numbers = [page_number for _, page_number in pages_iter]
        doc_record = doc_records[doc_id]
        page_images_gen = extract_pages_gen(
            doc_record.mime_type,
            doc_record.document_bytes,
            page_numbers,
            scaled_size=page_image_size,
        )
        page_numbers_it = iter(page_numbers)
        async for page_image in page_images_gen:
            image_by_page[doc_id, next(page_numbers_it)] = pil_image_as_base64(
                page_image, format="PNG"
            )

    return image_by_page


@chain
async def create_image_by_page(
    input: dict,
) -> dict:
    config: ChatChainConfig = input["chat_chain_config"]
    doc_records: List[DocumentRecord] = input.get("doc_records", [])
    index_items: List[Document] = input.get("found_items", [])

    chunks_metadatas = [
        ChunkMetadata(**index_item.metadata) for index_item in index_items
    ]

    image_by_page = await make_image_by_page(
        doc_records,
        chunks_metadatas,
        config.num_page_images_to_use,
        config.page_image_size,
    )
    return image_by_page


# TODO: Implement retrieval chain here
