from typing import List

from docarray import DocList
from langchain.docstore.document import Document

from aidial_rag.content_stream import SupportsWriteStr
from aidial_rag.document_record import Chunk
from aidial_rag.utils import format_size, get_bytes_length


def print_stats_numbers(
    file: SupportsWriteStr, chunks_num: int, total_text_size: int
):
    file.write(f"Number of chunks: {chunks_num}\n")
    file.write(f"Total text size: {format_size(total_text_size)}\n")


def print_chunks_stats(file: SupportsWriteStr, chunks: DocList[Chunk]):
    total_text_size = sum(get_bytes_length(chunk.text) for chunk in chunks)
    print_stats_numbers(file, len(chunks), total_text_size)


def print_documents_stats(file: SupportsWriteStr, documents: List[Document]):
    total_text_size = sum(
        get_bytes_length(document.page_content) for document in documents
    )
    print_stats_numbers(file, len(documents), total_text_size)
