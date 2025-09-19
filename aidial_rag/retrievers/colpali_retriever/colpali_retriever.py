import logging
import sys
from collections import defaultdict
from typing import List, Tuple

import numpy as np
import torch
from langchain.schema import BaseRetriever, Document
from torch import Tensor

from aidial_rag.batched import TqdmProgressBar, batched_map_with_progress
from aidial_rag.content_stream import SupportsWriteStr
from aidial_rag.document_record import (
    DocumentRecord,
    ItemEmbeddings,
    MultiEmbeddings,
)
from aidial_rag.image_processor.base64 import pil_image_from_base64
from aidial_rag.index_record import RetrievalType, to_metadata_doc
from aidial_rag.resources.cpu_pools import (
    run_in_heavy_indexing_embeddings_pool,
    run_in_heavy_query_embeddings_pool,
)
from aidial_rag.resources.dial_limited_resources import AsyncGeneratorWithTotal
from aidial_rag.retrievers.colpali_retriever.colpali_model_resource import (
    ColpaliModelResource,
)
from aidial_rag.retrievers.embeddings_index import (
    to_ndarray,
)
from aidial_rag.retrievers.page_image_retriever_utils import extract_page_images
from aidial_rag.utils import timed_block

logger = logging.getLogger(__name__)


class DocumentPageEmbedding:
    """Structure to hold document page embedding and chunk IDs."""

    embedding: np.ndarray
    chunk_ids: List[int]
    doc_idx_page_idx: Tuple[int, int]  # [doc_idx, page_idx]

    def __init__(
        self,
        embedding: np.ndarray,
        chunk_ids: List[int],
        doc_idx_page_idx: Tuple[int, int],
    ):
        self.embedding = embedding
        self.chunk_ids = chunk_ids
        self.doc_idx_page_idx = doc_idx_page_idx


class ColpaliRetriever(BaseRetriever):
    """ColPali retriever, calculates embeddings for documents and queries and scores documents against the query"""

    document_embeddings: List[DocumentPageEmbedding]
    k: int
    model_resource: ColpaliModelResource

    def _score_documents_with_embeddings(
        self, query_embeddings: Tensor
    ) -> List[Tuple[float, Tuple[int, int]]]:
        """Score all documents against the query embeddings and return sorted (score, (doc_idx, page_idx)) pairs."""
        query_embeddings = query_embeddings.half()

        page_scores = []
        page_indices = []

        for doc_embedding in self.document_embeddings:
            doc_idx_page_idx = doc_embedding.doc_idx_page_idx
            image_embedding = torch.from_numpy(doc_embedding.embedding).half()
            score = (
                self.model_resource.calculate_scores(
                    [query_embeddings], [image_embedding]
                )
                .squeeze()
                .item()
            )
            page_scores.append(score)
            page_indices.append(doc_idx_page_idx)

        if not page_scores:
            return []

        doc_scores = list(zip(page_scores, page_indices, strict=True))
        doc_scores.sort(key=lambda x: x, reverse=True)
        return doc_scores

    def _score_documents(
        self, query: str
    ) -> List[Tuple[float, Tuple[int, int]]]:
        """Score all documents against the query and return sorted (score, doc_idx) pairs."""
        query_embeddings_list = self.embed_queries([query])
        query_embeddings = query_embeddings_list[0]  # Get the single embedding
        return self._score_documents_with_embeddings(query_embeddings)

    async def _ascore_documents(
        self, query: str
    ) -> List[Tuple[float, Tuple[int, int]]]:
        """Async version of _score_documents"""
        query_embeddings_list = await self.aembed_queries([query])
        query_embeddings = query_embeddings_list[0]  # Get the single embedding
        return self._score_documents_with_embeddings(query_embeddings)

    def _collect_top_k_chunks(
        self, doc_scores: List[Tuple[float, Tuple[int, int]]]
    ) -> List[Document]:
        """Collect top k chunks from sorted document scores."""
        metadata_chunks = []
        for _, doc_idx_page_idx in doc_scores:
            doc_idx, page_idx = doc_idx_page_idx
            doc_embedding = self.document_embeddings[page_idx]

            # Add chunks from this document page one by one until we reach top k
            for chunk_id in doc_embedding.chunk_ids:
                if len(metadata_chunks) >= self.k:
                    return metadata_chunks

                metadata_chunks.append(
                    to_metadata_doc(
                        doc_idx,
                        chunk_id,
                        retrieval_type=RetrievalType.IMAGE,
                    )
                )

        return metadata_chunks

    def _get_relevant_documents(
        self, query: str, *args, **kwargs
    ) -> List[Document]:
        """Get relevant top k documents for a given query"""
        doc_scores = self._score_documents(query)
        return self._collect_top_k_chunks(doc_scores)

    async def _aget_relevant_documents(
        self, query: str, *args, **kwargs
    ) -> List[Document]:
        """Async version of _get_relevant_documents"""
        doc_scores = await self._ascore_documents(query)
        return self._collect_top_k_chunks(doc_scores)

    @classmethod
    def from_doc_records(
        cls,
        colpali_model_resouce: ColpaliModelResource,
        document_records: List[DocumentRecord],
        k: int = 1,
    ) -> "ColpaliRetriever":
        """Create ColPali retriever from document records"""
        if document_records is None:
            document_records = []

        document_embeddings = []
        for doc_idx, doc in enumerate(document_records):
            if doc.colpali_embeddings_index is not None:
                # Precalculate chunks per page
                chunks_per_page = defaultdict(list)
                for chunk_idx, chunk in enumerate(doc.chunks):
                    page_num = (
                        chunk.metadata["page_number"] - 1
                    )  # page_number is 1-indexed
                    chunks_per_page[page_num].append(chunk_idx)

                # Each page of the document has one list of embeddings that represent the page
                for page_idx, page_embedding in enumerate(
                    doc.colpali_embeddings_index
                ):
                    chunks_in_page = chunks_per_page.get(page_idx, [])

                    document_embeddings.append(
                        DocumentPageEmbedding(
                            embedding=page_embedding.embeddings,
                            chunk_ids=chunks_in_page,
                            doc_idx_page_idx=(doc_idx, page_idx),
                        )
                    )

        return cls(
            document_embeddings=document_embeddings,
            k=k,
            model_resource=colpali_model_resouce,
        )

    def embed_queries(self, queries: List[str]) -> List[Tensor]:
        """Embed queries using the ColPali model."""
        return self.model_resource.calculate_queries_embeddings(queries)

    async def aembed_queries(self, queries: List[str]) -> List[Tensor]:
        """Async version of embed_queries with batching support."""

        # Process queries in batches using batched_map_with_progress
        async def process_batch(batch: List[str]) -> List[Tensor]:
            return await run_in_heavy_query_embeddings_pool(
                self.embed_queries, batch
            )

        batch_size = self.model_resource.get_batch_size()
        batch_results = await batched_map_with_progress(
            queries,
            process_batch,  # Use CPU pool for heavy tasks
            batch_size=batch_size,
            file=sys.stdout,  # Use stdout for progress bar
        )

        query_embeddings_list = list(batch_results)

        return query_embeddings_list

    @staticmethod
    def _process_images_batch(
        images_batch: List[str], model_resource: ColpaliModelResource
    ) -> List[torch.Tensor]:
        """Process a batch of images using the ColPali model on GPU."""
        # Convert base64 images to PIL images

        # process images
        pil_images = []
        for image in images_batch:
            pil_image = pil_image_from_base64(image)
            pil_images.append(pil_image)

        return model_resource.calculate_images_embeddings(pil_images)

    @staticmethod
    async def embed_images(
        colpali_model_resource: ColpaliModelResource,
        images: AsyncGeneratorWithTotal,
        stageio,
    ) -> List[Tensor]:
        stageio.write("Processing images\n")

        # Process images in batches manually to avoid memory issues
        batch = []
        image_embeddings_list = []

        batch_size = colpali_model_resource.get_batch_size()
        # here cant use batched_map_with_progress because it loads all images into memory and doesnt support async generators
        with TqdmProgressBar(total=images.total, file=stageio) as pbar:
            async for image in images.agen:
                batch.append(image)

                if (
                    len(batch) >= batch_size
                ):  # Process batch when it reaches configured batch size
                    batch_results = await run_in_heavy_indexing_embeddings_pool(
                        ColpaliRetriever._process_images_batch,
                        batch,
                        colpali_model_resource,
                    )
                    image_embeddings_list.extend(batch_results)
                    pbar.update(len(batch))
                    batch = []  # Reset batch

            # Process remaining images in the last batch
            if batch:
                batch_results = await run_in_heavy_indexing_embeddings_pool(
                    ColpaliRetriever._process_images_batch,
                    batch,
                    colpali_model_resource,
                )
                image_embeddings_list.extend(batch_results)
                pbar.update(len(batch))

        return image_embeddings_list

    @staticmethod
    def has_index(document_records: List[DocumentRecord]) -> bool:
        return any(
            doc.colpali_embeddings_index is not None for doc in document_records
        )

    @staticmethod
    async def build_index(
        model_resource,
        stageio: SupportsWriteStr,
        mime_type: str,
        original_document: bytes,
    ) -> MultiEmbeddings | None:
        """Build ColPali indexes from a given document"""
        async with timed_block("Building ColPali indexes", stageio):
            logger.debug("Building Colpali indexes.")

            extract_pages_kwargs = {}

            extracted_images = await extract_page_images(
                mime_type,
                original_document,
                extract_pages_kwargs,
                stageio,
            )

            if extracted_images is None:
                return None

            all_embeddings = await ColpaliRetriever.embed_images(
                model_resource, extracted_images, stageio
            )
        return MultiEmbeddings(
            [
                ItemEmbeddings(
                    embeddings=to_ndarray(embeddings.cpu().float().numpy())
                )
                for embeddings in all_embeddings
            ]
        )
