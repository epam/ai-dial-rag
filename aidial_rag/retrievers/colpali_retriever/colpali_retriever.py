import logging
from typing import Any, List, Tuple

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
from aidial_rag.retrievers.colpali_retriever.colpali_model_resource import ColpaliModelResource
from aidial_rag.resources.dial_limited_resources import AsyncGeneratorWithTotal
from aidial_rag.retrievers.colpali_retriever.colpali_index_config import (
    ColpaliIndexConfig,
)
from aidial_rag.retrievers.embeddings_index import (
    EmbeddingsIndex,
    create_index_by_page,
    to_ndarray,
)
from aidial_rag.retrievers.page_image_retriever_utils import extract_page_images
from aidial_rag.utils import timed_block

logger = logging.getLogger(__name__)


class ColpaliRetriever(BaseRetriever):
    index: EmbeddingsIndex
    model: Any
    processor: Any
    device: torch.device

    def _get_relevant_documents(
        self, query: str, *args, **kwargs
    ) -> List[Document]:
        query_embeddings = self.embed_queries([query])

        scores = []
        document_ids = []
        original_chunk_ids = []
        for doc_id, document in enumerate(self.index.doc_indexes):
            chunk_count = (
                document.chunk_ids.max() - document.chunk_ids.min() + 1
            )
            chunk_size = (document.embeddings.shape[0] // chunk_count).item()
            image_embeddings = torch.from_numpy(document.embeddings).bfloat16()
            embeddings_per_chunk = list(
                torch.split(image_embeddings, chunk_size)
            )

            document_ids.extend([doc_id] * chunk_count)
            original_chunk_ids.extend(list(range(chunk_count)))
            scores.append(
                torch.squeeze(
                    self.processor.score_multi_vector(
                        query_embeddings, embeddings_per_chunk
                    )
                )
            )

        scores = torch.cat(scores)
        matching_chunk_ids = torch.topk(
            scores, 2, dim=0, largest=True, sorted=True
        )[1]

        metadata_chunks = []
        for chunk_id in matching_chunk_ids:
            matching_document = document_ids[chunk_id]
            original_chunk_id = original_chunk_ids[chunk_id]
            metadata_chunks.append(
                to_metadata_doc(
                    matching_document,
                    original_chunk_id,
                    retrieval_type=RetrievalType.IMAGE,
                )
            )

        return metadata_chunks

    async def _aget_relevant_documents(
        self, query: str, *args, **kwargs
    ) -> List[Document]:
        return self._get_relevant_documents(query, *args, **kwargs)

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

        indexes = [
            create_index_by_page(doc.chunks, doc.colpali_embeddings_index)
            for doc in document_records
            if doc.colpali_embeddings_index is not None
        ]

        return cls(
            index=EmbeddingsIndex(
                indexes=indexes, retrieval_type=RetrievalType.IMAGE
            ),
            model=model,
            processor=processor,
            device=device,
        )

    def embed_queries(self, queries: List[str]) -> Tensor:
        batch_queries = self.processor.process_queries(queries).to(self.device)
        with torch.no_grad():
            query_embeddings = self.model(**batch_queries)
        return query_embeddings

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
        image_embeddings_list = []
        counter = 1
        async for image in images.agen:
            image = pil_image_from_base64(image)
            batch_images = processor.process_images([image]).to(device)  # pyright: ignore
            stageio.write(f"Processing page {counter}/{images.total}\n")
            counter += 1
            with torch.no_grad():
                image_embeddings = model(**batch_images)
            image_embeddings_list.append(image_embeddings)

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

            extract_pages_kwargs = {"scaled_size": colpali_index_config.image_size}

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
