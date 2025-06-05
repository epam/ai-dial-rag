import logging
from io import BytesIO
from typing import Annotated, List

import aiohttp

# Onnx 1.16.2 and 1.17.0 have DLL initialization issues for Windows
#  https://github.com/onnx/onnx/issues/6267
# The onnx module should be imported before any unstructured_inference imports to avoid the issue
import onnx  # noqa: F401
from aidial_sdk import HTTPException
from langchain.schema import Document
from langchain_unstructured import UnstructuredLoader
from pdf2image.exceptions import PDFInfoNotInstalledError
from pydantic import ByteSize, Field
from unstructured.file_utils.model import FileType
from unstructured_pytesseract.pytesseract import TesseractNotFoundError

from aidial_rag.attachment_link import AttachmentLink
from aidial_rag.base_config import BaseConfig, IndexRebuildTrigger
from aidial_rag.content_stream import SupportsWriteStr
from aidial_rag.errors import InvalidDocumentError
from aidial_rag.image_processor.extract_pages import (
    are_image_pages_supported,
    extract_number_of_pages,
)
from aidial_rag.print_stats import print_documents_stats
from aidial_rag.request_context import RequestContext
from aidial_rag.resources.cpu_pools import run_in_indexing_cpu_pool
from aidial_rag.utils import format_size, get_bytes_length, timed_block


class HttpClientConfig(BaseConfig):
    timeout_seconds: int = Field(
        default=30,
        description="Timeout for the whole request. Includes connection establishment, sending the request, and receiving the response.",
    )
    connect_timeout_seconds: int = Field(
        default=30,
        description="Timeout for establishing a connection to the server.",
    )

    def get_client_timeout(self) -> aiohttp.ClientTimeout:
        return aiohttp.ClientTimeout(
            total=self.timeout_seconds,
            connect=self.connect_timeout_seconds,
            sock_connect=self.connect_timeout_seconds,
        )


class ParserConfig(BaseConfig):
    max_document_text_size: Annotated[
        ByteSize,
        Field(
            default="5MiB",
            validate_default=True,
            description=(
                "Limits the size of the document the RAG will accept for processing. "
                "This limit is applied to the size of the text extracted from the document, "
                "not the size of the attached document itself. "
                "Could be integer for bytes, or a pydantic.ByteSize compatible string."
            ),
        ),
    ]
    unstructured_chunk_size: Annotated[
        int,
        IndexRebuildTrigger(),
        Field(
            default=1000,
            description="Sets the chunk size for unstructured document loader.",
        ),
    ]


async def download_attachment(
    url, headers, download_config: HttpClientConfig
) -> tuple[str, bytes]:
    async with aiohttp.ClientSession() as session:
        async with session.get(
            url, headers=headers, timeout=download_config.get_client_timeout()
        ) as response:
            response.raise_for_status()
            content_type = response.headers.get("Content-Type", "")

            content = await response.read()  # Await the coroutine
            logging.debug(f"Downloaded {url}: {len(content)} bytes")
            return content_type, content


def add_source_metadata(
    pages: List[Document], attachment_link: AttachmentLink
) -> List[Document]:
    for page in pages:
        page.metadata["source"] = attachment_link.dial_link
        page.metadata["source_display_name"] = attachment_link.display_name
    return pages


def add_pdf_source_metadata(
    pages: List[Document], attachment_link: AttachmentLink
) -> List[Document]:
    assert len(pages)
    pages = add_source_metadata(pages, attachment_link)

    for page in pages:
        if "page_number" in page.metadata:
            page.metadata["source"] += f"#page={page.metadata['page_number']}"
    return pages


async def load_dial_document_metadata(
    request_context: RequestContext,
    attachment_link: AttachmentLink,
    config: HttpClientConfig,
) -> dict:
    if not attachment_link.is_dial_document:
        raise ValueError("Not a Dial document")

    metadata_url = attachment_link.dial_metadata_url
    assert metadata_url is not None

    headers = request_context.get_file_access_headers(metadata_url)
    async with aiohttp.ClientSession(
        timeout=config.get_client_timeout()
    ) as session:
        async with session.get(metadata_url, headers=headers) as response:
            if not response.ok:
                error_message = f"{response.status} {response.reason}"
                raise InvalidDocumentError(error_message)
            return await response.json()


async def load_attachment(
    attachment_link: AttachmentLink,
    headers: dict,
    download_config: HttpClientConfig | None = None,
) -> tuple[str, str, bytes]:
    if download_config is None:
        download_config = HttpClientConfig()
    absolute_url = attachment_link.absolute_url
    file_name = attachment_link.display_name
    content_type, attachment_bytes = await download_attachment(
        absolute_url, headers, download_config
    )
    if attachment_bytes:
        return file_name, content_type, attachment_bytes
    raise InvalidDocumentError(
        f"Attachment {file_name}, can't be read properly"
    )


def add_image_only_chunks(
    document_bytes: bytes,
    mime_type: str,
    existing_chunks: List[Document],
) -> List[Document]:
    assert all(
        existing_chunks[i].metadata["page_number"]
        <= existing_chunks[i + 1].metadata["page_number"]
        for i in range(len(existing_chunks) - 1)
    )

    number_of_pages = extract_number_of_pages(mime_type, document_bytes)
    assert all(
        1 <= existing_chunk.metadata["page_number"] <= number_of_pages
        for existing_chunk in existing_chunks
    )

    result_chunks = []
    cur_existing_chunk_idx = 0
    for i in range(1, number_of_pages + 1):
        while (
            cur_existing_chunk_idx < len(existing_chunks)
            and existing_chunks[cur_existing_chunk_idx].metadata["page_number"]
            == i
        ):
            result_chunks.append(existing_chunks[cur_existing_chunk_idx])
            cur_existing_chunk_idx += 1

        if not result_chunks or result_chunks[-1].metadata["page_number"] != i:
            result_chunks.append(
                Document(
                    page_content="",
                    metadata={
                        "filetype": mime_type,
                        "page_number": i,
                    },
                )
            )

    assert cur_existing_chunk_idx == len(existing_chunks)
    return result_chunks


def get_document_chunks(
    document_bytes: bytes,
    mime_type: str,
    attachment_link: AttachmentLink,
    attachment_mime_type: str,
    parser_config: ParserConfig,
) -> List[Document]:
    try:
        chunks = UnstructuredLoader(
            file=BytesIO(document_bytes),
            metadata_filename=attachment_link.display_name,
            # Current version of unstructured library expect mime type instead of the full content type with encoding, etc.
            content_type=mime_type,
            mode="elements",
            strategy="fast",
            chunking_strategy="by_title",
            multipage_sections=False,
            # Disable combining text chunks, because it does not respect multipage_sections=False
            # TODO: Update unstructured library to the version with chunking/title.py refactoring
            combine_text_under_n_chars=0,
            new_after_n_chars=parser_config.unstructured_chunk_size,
            max_characters=parser_config.unstructured_chunk_size,
        ).load()
    except ValueError as e:
        raise HTTPException(
            "Unable to load document content. Try another document format.",
        ) from e
    except (PDFInfoNotInstalledError, TesseractNotFoundError):
        # TODO: Update unstructured library to avoid attempts to use ocr
        logging.warning("PDF file without text. Trying to extract images.")
        chunks = None

    if chunks is None:
        chunks = []

    if are_image_pages_supported(mime_type):
        # We will not have chunks from unstructured for the pages which does not contain text
        # So we need to add them manually
        chunks = add_image_only_chunks(document_bytes, mime_type, chunks)

    if not chunks:
        raise InvalidDocumentError("The document is empty")

    attachment_filetype = FileType.from_mime_type(attachment_mime_type)
    if attachment_filetype == FileType.PDF:
        chunks = add_pdf_source_metadata(chunks, attachment_link)
    else:
        chunks = add_source_metadata(chunks, attachment_link)
    return chunks


async def parse_document(
    stageio: SupportsWriteStr,
    document_bytes: bytes,
    mime_type: str,
    attachment_link: AttachmentLink,
    attachment_mime_type: str,
    parser_config: ParserConfig | None = None,
) -> List[Document]:
    if parser_config is None:
        # pyright does not understand default values in Annotated
        parser_config = ParserConfig()  # type: ignore
    async with timed_block("Parsing document", stageio):
        stageio.write("Loader: Unstructured\n")
        chunks = await run_in_indexing_cpu_pool(
            get_document_chunks,
            document_bytes,
            mime_type,
            attachment_link,
            attachment_mime_type,
            parser_config,
        )

        stageio.write(f"File type: {chunks[0].metadata['filetype']}\n")
        print_documents_stats(stageio, chunks)

        total_text_size = sum(
            get_bytes_length(chunk.page_content) for chunk in chunks
        )
        if total_text_size > parser_config.max_document_text_size:
            raise InvalidDocumentError(
                f"Document text is too large: {format_size(total_text_size)} > "
                f"{format_size(parser_config.max_document_text_size)}"
            )

        return chunks
