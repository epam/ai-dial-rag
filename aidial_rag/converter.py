import asyncio
import tempfile

from unstructured.partition.common.common import convert_office_doc

from aidial_rag.content_stream import SupportsWriteStr
from aidial_rag.resources.cpu_pools import run_in_indexing_cpu_pool
from aidial_rag.utils import format_size, timed_block

# Only one LibreOffice instance can run at a time
soffice_semaphore = asyncio.Semaphore(1)


PDF_MIME_TYPE = "application/pdf"

CONVERT_TO_PDF_MIME_TYPES = [
    # Docs
    "application/msword",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.oasis.opendocument.text",
    # Presentations
    "application/vnd.ms-powerpoint",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "application/vnd.oasis.opendocument.presentation",
]


def _convert_office_to_pdf(doc_bytes: bytes):
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file = f"{temp_dir}/doc_file"
        with open(temp_file, "wb") as f:
            f.write(doc_bytes)

        convert_office_doc(
            input_filename=temp_file,
            output_directory=temp_dir,
            target_format="pdf",
        )
        with open(f"{temp_dir}/doc_file.pdf", "rb") as f:
            pdf_bytes = f.read()

    return pdf_bytes


async def convert_office_to_pdf(
    doc_bytes: bytes, io_stream: SupportsWriteStr
) -> bytes:
    async with timed_block("Converting document to pdf", io_stream):
        async with soffice_semaphore:
            pdf_bytes = await run_in_indexing_cpu_pool(
                _convert_office_to_pdf, doc_bytes
            )
            io_stream.write(f"New size: {format_size(len(pdf_bytes))}\n")
            return pdf_bytes


async def convert_document_if_needed(
    mime_type: str, doc_bytes: bytes, io_stream: SupportsWriteStr
) -> tuple[str, bytes]:
    if mime_type in CONVERT_TO_PDF_MIME_TYPES:
        doc_bytes = await convert_office_to_pdf(doc_bytes, io_stream)
        return PDF_MIME_TYPE, doc_bytes

    return mime_type, doc_bytes
