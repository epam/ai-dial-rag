import asyncio
from itertools import groupby
from operator import itemgetter
from typing import Any, Callable, Dict, List

from langchain.retrievers import EnsembleRetriever
from langchain.schema import BaseRetriever, Document
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.retrievers import RetrieverLike
from langchain_core.runnables import Runnable, chain

from aidial_rag.configuration_endpoint import Configuration
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
from aidial_rag.retrieval_api import RetrievalResults
from aidial_rag.retrievers.all_documents_retriever import AllDocumentsRetriever
from aidial_rag.retrievers.bm25_retriever import BM25Retriever
from aidial_rag.retrievers.description_retriever.description_retriever import (
    DescriptionRetriever,
)
from aidial_rag.retrievers.multimodal_retriever import MultimodalRetriever
from aidial_rag.retrievers.semantic_retriever import SemanticRetriever


def _make_stage_default(
    retriever: BaseRetriever, *args, **kwargs
) -> BaseRetriever:
    """Default stage maker that returns the retriever as is."""
    return retriever


def create_retriever(
    dial_config: DialConfig,
    document_records: List[DocumentRecord],
    indexing_config: IndexingConfig,
    make_stage: Callable[
        [BaseRetriever, str], BaseRetriever
    ] = _make_stage_default,
) -> BaseRetriever:
    if not AllDocumentsRetriever.is_within_limit(document_records):
        semantic_retriever = make_stage(
            SemanticRetriever.from_doc_records(document_records, 7),
            "Embeddings search",
        )
        retrievers: List[RetrieverLike] = [semantic_retriever]
        weights = [1.0]

        if BM25Retriever.has_index(document_records):
            bm25_retriever = make_stage(
                BM25Retriever.from_doc_records(document_records, 7),
                "Keywords search",
            )
            retrievers.append(bm25_retriever)
            weights.append(1.0)

        if MultimodalRetriever.has_index(document_records):
            assert indexing_config.multimodal_index
            multimodal_retriever = make_stage(
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
            description_retriever = make_stage(
                DescriptionRetriever.from_doc_records(document_records, 7),
                "Page image search",
            )
            retrievers.append(description_retriever)
            weights.append(1.0)

        retriever = make_stage(
            EnsembleRetriever(
                retrievers=retrievers,
                weights=weights,
            ),
            "Combined search",
        )
    else:
        retriever = make_stage(
            AllDocumentsRetriever.from_doc_records(document_records),
            "All documents",
        )

    return retriever


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


@chain
async def create_retrieval_results(
    input: dict,
) -> RetrievalResults:
    """Create retrieval results from the input data."""
    doc_records: List[DocumentRecord] = input.get("doc_records", [])
    index_items: List[Document] = input.get("found_items", [])
    image_by_page: dict = input.get("image_by_page", {})

    chunks_metadatas = [
        ChunkMetadata(**index_item.metadata) for index_item in index_items
    ]

    images = []
    used_image_keys = set()

    chunks = []

    for chunk_metadata in chunks_metadatas:
        doc_id = chunk_metadata["doc_id"]
        chunk_id = chunk_metadata["chunk_id"]
        doc_record = doc_records[doc_id]
        chunk = doc_record.chunks[chunk_id]
        chunk_data = RetrievalResults.Chunk(
            doc_id=doc_id,
            chunk_id=chunk_id,
            text=chunk.text,
            source=chunk.metadata["source"],
            source_display_name=chunk.metadata.get("source_display_name"),
            page_number=chunk.metadata.get("page_number"),
        )

        image_key = (
            doc_id,
            chunk.metadata.get("page_number"),
        )
        if image_key in image_by_page and image_key not in used_image_keys:
            used_image_keys.add(image_key)
            image_index = len(images)
            images.append(
                RetrievalResults.Image(
                    data=image_by_page[image_key],
                    mime_type="image/png",
                )
            )
            chunk_data.page_image = image_index

        chunks.append(chunk_data)

    return RetrievalResults(
        chunks=chunks,
        images=images,
    )


async def create_retrieval_chain(
    dial_config: DialConfig,
    request_config: Configuration,
    document_records: List[DocumentRecord],
    query_chain: Runnable[Dict[str, Any], str],
    make_stage: Callable[
        [BaseRetriever, str], BaseRetriever
    ] = _make_stage_default,
) -> Runnable[Dict[str, Any], Dict]:
    retriever = await asyncio.get_running_loop().run_in_executor(
        None,
        create_retriever,
        dial_config,
        document_records,
        request_config.indexing,
        make_stage,
    )

    retrieval_chain = (
        RunnablePassthrough()
        .assign(query=query_chain)
        .assign(found_items=(itemgetter("query") | retriever))
        .assign(image_by_page=create_image_by_page)
        .assign(retrieval_results=create_retrieval_results)
    )
    return retrieval_chain
