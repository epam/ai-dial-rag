import asyncio
from dataclasses import dataclass
from itertools import groupby
from operator import itemgetter
from typing import Any, Callable, Dict, List, Set

from langchain.retrievers import EnsembleRetriever
from langchain.schema import BaseRetriever, Document
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.retrievers import RetrieverLike
from langchain_core.runnables import Runnable, chain

from aidial_rag.attachment_link import AttachmentLink
from aidial_rag.dial_config import DialConfig
from aidial_rag.document_record import DocumentRecord
from aidial_rag.image_processor.base64 import pil_image_as_base64
from aidial_rag.image_processor.extract_pages import (
    are_image_pages_supported,
    extract_pages_gen,
)
from aidial_rag.index_record import ChunkMetadata, RetrievalType
from aidial_rag.indexing_config import IndexingConfig
from aidial_rag.qa_chain_config import ChatChainConfig
from aidial_rag.retrieval_api import (
    Chunk,
    Image,
    Page,
    RetrievalResults,
    Source,
)
from aidial_rag.retrievers.all_documents_retriever import AllDocumentsRetriever
from aidial_rag.retrievers.bm25_retriever import BM25Retriever
from aidial_rag.retrievers.description_retriever.description_retriever import (
    DescriptionRetriever,
)
from aidial_rag.retrievers.multimodal_retriever import MultimodalRetriever
from aidial_rag.retrievers.semantic_retriever import SemanticRetriever


@dataclass(frozen=True, order=True)
class PageKey:
    doc_id: int
    page_number: int


def collect_pages_with_images(
    doc_records: List[DocumentRecord],
    chunks_metadatas: List[ChunkMetadata],
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
    chunks_metadatas: List[ChunkMetadata],
    num_pages_to_use: int,
    page_image_size: int,
) -> Dict[PageKey, str]:
    required_pages: Set[PageKey] = set()
    for doc_id, page_number in collect_pages_with_images(
        doc_records, chunks_metadatas
    ):
        if len(required_pages) >= num_pages_to_use:
            break
        required_pages.add(PageKey(doc_id, page_number))

    image_by_page: Dict[PageKey, str] = {}
    for doc_id, pages_iter in groupby(
        sorted(required_pages), lambda key: key.doc_id
    ):
        page_keys = list(pages_iter)
        doc_record = doc_records[doc_id]
        page_images_gen = extract_pages_gen(
            doc_record.mime_type,
            doc_record.document_bytes,
            page_numbers=[key.page_number for key in page_keys],
            scaled_size=page_image_size,
        )
        page_keys_it = iter(page_keys)
        async for page_image in page_images_gen:
            image_by_page[next(page_keys_it)] = pil_image_as_base64(
                page_image, format="PNG"
            )

    return image_by_page


@chain
async def create_image_by_page(
    input: Dict[str, Any],
) -> Dict[PageKey, str]:
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


@chain
async def create_retrieval_results(
    input: Dict[str, Any],
) -> RetrievalResults:
    """Create retrieval results from the input data."""
    doc_records: List[DocumentRecord] = input.get("doc_records", [])
    doc_records_links: List[AttachmentLink] = input.get("doc_records_links", [])
    index_items: List[Document] = input.get("found_items", [])
    image_by_page: Dict[PageKey, str] = input.get("image_by_page", {})

    images: List[Image] = []
    chunks: List[Chunk] = []
    used_image_keys: Set[PageKey] = set()

    for index_item in index_items:
        chunk_metadata = ChunkMetadata(**index_item.metadata)
        doc_id = chunk_metadata["doc_id"]
        chunk_id = chunk_metadata["chunk_id"]
        doc_record = doc_records[doc_id]
        doc_record_link = doc_records_links[doc_id]
        doc_record_chunk = doc_record.chunks[chunk_id]
        chunk_data = Chunk(
            attachment_url=doc_record_link.dial_link,
            text=doc_record_chunk.text,
            source=Source(
                url=doc_record_chunk.metadata["source"],
                display_name=doc_record_chunk.metadata.get(
                    "source_display_name"
                ),
            ),
            page=None,
        )

        if (
            page_number := doc_record_chunk.metadata.get("page_number")
        ) is not None:
            chunk_data.page = Page(
                number=page_number,
                image_index=None,
            )

            page_key = PageKey(doc_id, page_number)
            if page_key in image_by_page and page_key not in used_image_keys:
                used_image_keys.add(page_key)
                image_index = len(images)
                images.append(Image(data=image_by_page[page_key]))
                chunk_data.page.image_index = image_index

        chunks.append(chunk_data)

    return RetrievalResults(
        chunks=chunks,
        images=images,
    )


def _make_retrieval_stage_default(
    retriever: BaseRetriever, stage_name: str
) -> BaseRetriever:
    """Default stage maker that returns the retriever as is."""
    return retriever


def create_retriever(
    dial_config: DialConfig,
    document_records: List[DocumentRecord],
    indexing_config: IndexingConfig,
    make_retrieval_stage: Callable[
        [BaseRetriever, str], BaseRetriever
    ] = _make_retrieval_stage_default,
) -> BaseRetriever:
    if not AllDocumentsRetriever.is_within_limit(document_records):
        semantic_retriever = make_retrieval_stage(
            SemanticRetriever.from_doc_records(document_records, 7),
            "Embeddings search",
        )
        retrievers: List[RetrieverLike] = [semantic_retriever]
        weights = [1.0]

        if BM25Retriever.has_index(document_records):
            bm25_retriever = make_retrieval_stage(
                BM25Retriever.from_doc_records(document_records, 7),
                "Keywords search",
            )
            retrievers.append(bm25_retriever)
            weights.append(1.0)

        if MultimodalRetriever.has_index(document_records):
            assert indexing_config.multimodal_index
            multimodal_retriever = make_retrieval_stage(
                MultimodalRetriever.from_doc_records(
                    dial_config,
                    indexing_config.multimodal_index,
                    document_records,
                    7,
                ),
                "Multimodal search",
            )
            retrievers.append(multimodal_retriever)
            weights.append(1.0)

        if DescriptionRetriever.has_index(document_records):
            description_retriever = make_retrieval_stage(
                DescriptionRetriever.from_doc_records(document_records, 7),
                "Page image search",
            )
            retrievers.append(description_retriever)
            weights.append(1.0)

        retriever = make_retrieval_stage(
            EnsembleRetriever(
                retrievers=retrievers,
                weights=weights,
            ),
            "Combined search",
        )
    else:
        retriever = make_retrieval_stage(
            AllDocumentsRetriever.from_doc_records(document_records),
            "All documents",
        )

    return retriever


async def create_retrieval_chain(
    dial_config: DialConfig,
    indexing_config: IndexingConfig,
    document_records: List[DocumentRecord],
    query_chain: Runnable[Dict[str, Any], str],
    make_retrieval_stage: Callable[
        [BaseRetriever, str], BaseRetriever
    ] = _make_retrieval_stage_default,
) -> Runnable[Dict[str, Any], Dict[str, Any]]:
    retriever = await asyncio.get_running_loop().run_in_executor(
        None,
        create_retriever,
        dial_config,
        document_records,
        indexing_config,
        make_retrieval_stage,
    )

    retrieval_chain = (
        RunnablePassthrough()
        .assign(query=query_chain)
        .assign(found_items=(itemgetter("query") | retriever))
        .assign(image_by_page=create_image_by_page)
        .assign(retrieval_results=create_retrieval_results)
    )
    return retrieval_chain
