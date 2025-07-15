import logging
from collections import defaultdict
from typing import Any, List, Tuple

import numpy as np
import torch
from langchain.schema import BaseRetriever
from langchain_core.documents import Document
from torch import Tensor

from aidial_rag.content_stream import SupportsWriteStr
from aidial_rag.document_record import (
    DocumentRecord,
    ItemEmbeddings,
    MultiEmbeddings,
)
from aidial_rag.image_processor.base64 import pil_image_from_base64
from aidial_rag.index_record import RetrievalType, to_metadata_doc
from aidial_rag.resources.cpu_pools import run_in_indexing_cpu_pool
from aidial_rag.resources.dial_limited_resources import AsyncGeneratorWithTotal
from aidial_rag.retrievers.colpali_retriever.colpali_index_config import (
    ColpaliIndexConfig,
)
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
    doc_idx: int

    def __init__(
        self, embedding: np.ndarray, chunk_ids: List[int], doc_idx: int
    ):
        self.embedding = embedding
        self.chunk_ids = chunk_ids
        self.doc_idx = doc_idx


class ColpaliRetriever(BaseRetriever):
    document_embeddings: List[DocumentPageEmbedding]
    model: Any
    processor: Any
    device: torch.device
    k: int
    model_resource: ColpaliModelResource

    def _score_documents_with_embeddings(
        self, query_embeddings: Tensor
    ) -> List[Tuple[float, int]]:
        """Score all documents against the query embeddings and return sorted (score, doc_idx) pairs."""
        query_embeddings = query_embeddings.half()

        page_scores = []
        page_indices = []

        for doc_id, doc_embedding in enumerate(self.document_embeddings):
            image_embedding = torch.from_numpy(doc_embedding.embedding).half()
            score = (
                self.processor.score_multi_vector(
                    query_embeddings, [image_embedding]
                )
                .squeeze()
                .item()
            )
            page_scores.append(score)
            page_indices.append(doc_id)

        if not page_scores:
            return []

        doc_scores = list(zip(page_scores, page_indices, strict=True))
        doc_scores.sort(key=lambda x: x, reverse=True)
        return doc_scores

    def _score_documents(self, query: str) -> List[Tuple[float, int]]:
        """Score all documents against the query and return sorted (score, doc_idx) pairs."""
        query_embeddings = self.embed_queries([query])
        return self._score_documents_with_embeddings(query_embeddings)

    async def _ascore_documents(self, query: str) -> List[Tuple[float, int]]:
        """Async version of _score_documents"""
        query_embeddings = await self.aembed_queries([query])
        return self._score_documents_with_embeddings(query_embeddings)

    def _collect_top_k_chunks(
        self, doc_scores: List[Tuple[float, int]]
    ) -> List[Document]:
        """Collect top k chunks from sorted document scores."""
        metadata_chunks = []
        for _, page_idx in doc_scores:
            doc_embedding = self.document_embeddings[page_idx]

            # Add chunks from this document one by one until we reach top k
            for chunk_id in doc_embedding.chunk_ids:
                if len(metadata_chunks) >= self.k:
                    return metadata_chunks

                metadata_chunks.append(
                    to_metadata_doc(
                        doc_embedding.doc_idx,
                        chunk_id,
                        retrieval_type=RetrievalType.IMAGE,
                    )
                )

        return metadata_chunks

    def _get_relevant_documents(
        self, query: str, *args, **kwargs
    ) -> List[Document]:
        doc_scores = self._score_documents(query)
        return self._collect_top_k_chunks(doc_scores)

    async def _aget_relevant_documents(
        self, query: str, *args, **kwargs
    ) -> List[Document]:
        doc_scores = await self._ascore_documents(query)
        return self._collect_top_k_chunks(doc_scores)

    @classmethod
    def from_doc_records(
        cls,
        colpali_model_resouce: ColpaliModelResource,
        colpali_index_config: ColpaliIndexConfig,
        document_records: List[DocumentRecord],
        k: int = 1,
    ) -> "ColpaliRetriever":
        model, processor, device = (
            colpali_model_resouce.get_model_processor_device()
        )
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
                            doc_idx=doc_idx,
                        )
                    )

        return cls(
            document_embeddings=document_embeddings,
            model=model,
            processor=processor,
            device=device,
            k=k,
            model_resource=colpali_model_resouce,
        )

    def embed_queries(self, queries: List[str]) -> Tensor:
        if self.processor is None:
            raise RuntimeError("Processor is not initialized.")
        batch_queries = self.processor.process_queries(queries).to(self.device)
        with torch.no_grad():
            with self.model_resource.get_gpu_lock():
                query_embeddings = self.model(**batch_queries)
        return query_embeddings

    async def aembed_queries(self, queries: List[str]) -> Tensor:
        """Async version of embed_queries that runs in CPU pool."""
        return await run_in_indexing_cpu_pool(self.embed_queries, queries)

    @staticmethod
    def pad_embeddings(tensor: Tensor, target_shape: Tuple) -> Tensor:
        padding_dims = [
            (0, target_shape[2] - tensor.shape[2]),
            (0, target_shape[1] - tensor.shape[1]),
            (0, 0),
        ]
        padding_dims_flat = [
            item for sublist in padding_dims[::-1] for item in sublist
        ]
        return torch.nn.functional.pad(
            tensor, pad=padding_dims_flat, mode="constant", value=0
        )

    @staticmethod
    async def embed_images(
        colpali_model_resource: ColpaliModelResource,
        colpali_index_confis: ColpaliIndexConfig,
        images: AsyncGeneratorWithTotal,
        stageio,
    ) -> list[Tensor]:
        model, processor, device = (
            colpali_model_resource.get_model_processor_device()
        )
        if device is None:
            raise RuntimeError(
                "ColpaliModelResource did not return a valid device."
            )
        if model is None:
            raise RuntimeError(
                "ColpaliModelResource did not return a valid model."
            )

        def process_image_gpu(image_base64):
            image = pil_image_from_base64(image_base64)
            batch_images = processor.process_images([image]).to(device)  # pyright: ignore
            with torch.no_grad():
                with colpali_model_resource.get_gpu_lock():
                    return model(**batch_images)

        image_embeddings_list = []
        counter = 1

        async for image in images.agen:
            stageio.write(f"Processing page {counter}/{images.total}\n")
            image_embeddings = await run_in_indexing_cpu_pool(
                process_image_gpu, image
            )
            image_embeddings_list.append(image_embeddings)
            counter += 1

        max_shape = (
            max(embed.shape[0] for embed in image_embeddings_list),
            max(embed.shape[1] for embed in image_embeddings_list),
            max(embed.shape[2] for embed in image_embeddings_list),
        )

        padded_embeddings = [
            torch.squeeze(ColpaliRetriever.pad_embeddings(embed, max_shape), 0)
            for embed in image_embeddings_list
        ]

        return padded_embeddings

    @staticmethod
    def has_index(document_records: List[DocumentRecord]) -> bool:
        return any(
            doc.colpali_embeddings_index is not None for doc in document_records
        )

    @staticmethod
    async def build_index(
        model_resource,
        colpali_index_config: ColpaliIndexConfig,
        stageio: SupportsWriteStr,
        mime_type: str,
        original_document: bytes,
    ) -> MultiEmbeddings | None:
        async with timed_block("Building ColPali indexes", stageio):
            logger.debug("Building Colpali indexes.")

            extract_pages_kwargs = {
                "scaled_size": colpali_index_config.image_size
            }

            extracted_images = await extract_page_images(
                mime_type,
                original_document,
                extract_pages_kwargs,
                stageio,
            )

            if extracted_images is None:
                return None

            all_embeddings = await ColpaliRetriever.embed_images(
                model_resource, colpali_index_config, extracted_images, stageio
            )
        return MultiEmbeddings(
            [
                ItemEmbeddings(
                    embeddings=to_ndarray(embeddings.cpu().float().numpy())
                )
                for embeddings in all_embeddings
            ]
        )
