import asyncio
import logging
import sys
from typing import Generator, List, Tuple

import numpy as np
from docarray import DocList
from langchain.schema import BaseRetriever, Document
from pydantic import Field
from rank_bm25 import BM25Okapi

from aidial_rag.content_stream import SupportsWriteStr
from aidial_rag.document_record import Chunk, DocumentRecord
from aidial_rag.index_record import (
    RetrievalType,
    TextIndexItem,
    to_metadata_doc,
)
from aidial_rag.keywords_search import keywords_preprocess
from aidial_rag.resources.cpu_pools import run_in_indexing_cpu_pool
from aidial_rag.utils import timed_block

logger = logging.getLogger(__name__)


class SearchItem(TextIndexItem):
    doc_index: int


def _build_text_index_chunks(chunks: List[Chunk]):
    return DocList(
        [
            TextIndexItem(
                chunk_index=i,
                tokenized_text=keywords_preprocess(chunk.text),
            )
            for i, chunk in enumerate(chunks)
        ]
    )


class BM25Retriever(BaseRetriever):
    k: int
    text_indexes: List[SearchItem] = Field(repr=False)
    bm25: BM25Okapi = Field(repr=False)

    @staticmethod
    def _get_text_index_gen(
        doc_records: List[DocumentRecord],
    ) -> Generator[Tuple[int, TextIndexItem], None, None]:
        for i, doc in enumerate(doc_records):
            if doc.text_index is not None:
                for item in doc.text_index:
                    yield i, item

    @staticmethod
    def has_index(document_records: List[DocumentRecord]) -> bool:
        total_tokenized_text_len = sum(
            len(item.tokenized_text)
            for _, item in BM25Retriever._get_text_index_gen(document_records)
        )
        return total_tokenized_text_len > 0

    @classmethod
    def from_doc_records(
        cls, doc_records: List[DocumentRecord], k: int = 4
    ) -> "BM25Retriever":
        text_indexes = [
            SearchItem(doc_index=i, **item.model_dump())
            for i, item in BM25Retriever._get_text_index_gen(doc_records)
        ]
        tokenized_texts = [item.tokenized_text for item in text_indexes]

        # bm25 requires at least one text token to avoid division by zero
        if sum(map(len, tokenized_texts)) == 0:
            raise ValueError("Text index is empty.")

        bm25 = BM25Okapi(tokenized_texts)
        return cls(text_indexes=text_indexes, k=k, bm25=bm25)

    def _get_top_n_indexes(self, query: List[str], n: int = 5) -> np.ndarray:
        # bm25.get_top_n uses non-stable argsort
        scores = self.bm25.get_scores(query)
        return np.argsort(scores, kind="stable")[::-1][:n]

    def get_metadata_doc(self, index: int) -> Document:
        item = self.text_indexes[index]
        return to_metadata_doc(
            item.doc_index, item.chunk_index, RetrievalType.TEXT
        )

    def _get_relevant_documents(
        self, query: str, *args, **kwargs
    ) -> List[Document]:
        processed_query = keywords_preprocess(query)
        top_n = self._get_top_n_indexes(processed_query, self.k)
        return [self.get_metadata_doc(i) for i in top_n]

    async def _aget_relevant_documents(
        self, query: str, *args, **kwargs
    ) -> List[Document]:
        return await asyncio.get_running_loop().run_in_executor(
            None, self._get_relevant_documents, query, *args, **kwargs
        )

    @staticmethod
    async def build_index(
        chunks: List[Chunk], stageio: SupportsWriteStr = sys.stderr
    ) -> DocList[TextIndexItem]:
        async with timed_block("Building BM25 indexes", stageio):
            logger.debug("Building BM25 indexes.")
            return await run_in_indexing_cpu_pool(
                _build_text_index_chunks, chunks
            )
