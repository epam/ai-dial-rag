import asyncio
import logging
from contextlib import contextmanager
from email.policy import EmailPolicy
from typing import Iterable, List

from docarray import DocList

from aidial_rag.attachment_link import AttachmentLink
from aidial_rag.configuration_endpoint import RequestConfig
from aidial_rag.content_stream import (
    LoggerStream,
    MarkdownStream,
    MultiStream,
    StreamWithPrefix,
    SupportsWriteStr,
)
from aidial_rag.converter import convert_document_if_needed
from aidial_rag.dial_config import DialConfig
from aidial_rag.document_loaders import (
    load_attachment,
    load_dial_document_metadata,
    parse_document,
)
from aidial_rag.document_record import (
    FORMAT_VERSION,
    Chunk,
    DocumentRecord,
    IndexSettings,
    build_chunks_list,
)
from aidial_rag.errors import (
    DocumentProcessingError,
    InvalidDocumentError,
    convert_and_log_exceptions,
)
from aidial_rag.image_processor.extract_pages import is_image
from aidial_rag.index_storage import IndexStorage
from aidial_rag.indexing_results import (
    DocumentIndexingFailure,
    DocumentIndexingResult,
    DocumentIndexingSuccess,
)
from aidial_rag.indexing_task import IndexingTask
from aidial_rag.print_stats import print_chunks_stats
from aidial_rag.request_context import RequestContext
from aidial_rag.resources.dial_limited_resources import DialLimitedResources
from aidial_rag.retrievers.bm25_retriever import BM25Retriever
from aidial_rag.retrievers.description_retriever.description_retriever import (
    DescriptionRetriever,
)
from aidial_rag.retrievers.multimodal_retriever import (
    MultimodalRetriever,
)
from aidial_rag.retrievers.semantic_retriever import SemanticRetriever
from aidial_rag.utils import format_size, timed_stage

logger = logging.getLogger(__name__)


class FailStageException(Exception):
    pass


async def check_document_access(
    request_context: RequestContext,
    attachment_link: AttachmentLink,
    config: RequestConfig,
):
    # Try to load document metadata to check the access to the document for the documents in the Dial filesystem.
    if not attachment_link.is_dial_document:
        return

    with timed_stage(
        request_context.choice,
        f"Access document '{attachment_link.display_name}'",
    ) as access_stage:
        try:
            await load_dial_document_metadata(
                request_context, attachment_link, config.check_access
            )
        except InvalidDocumentError as e:
            access_stage.append_content(e.message)
            raise


def parse_content_type(content_type):
    header = EmailPolicy.header_factory("content-type", content_type)
    return header.content_type, dict(header.params)


def get_default_image_chunk(attachment_link: AttachmentLink):
    return Chunk(
        text="",
        metadata={
            "page_number": 1,
            "source_display_name": attachment_link.display_name,
            "source": attachment_link.dial_link,
        },
    )


async def load_document_impl(
    dial_config: DialConfig,
    dial_limited_resources: DialLimitedResources,
    attachment_link: AttachmentLink,
    stage_stream: SupportsWriteStr,
    index_settings: IndexSettings,
    config: RequestConfig,
) -> DocumentRecord:
    logger_stream = LoggerStream()
    if config.log_document_links:
        logger_stream = StreamWithPrefix(
            logger_stream, f"<{attachment_link.dial_link}>: "
        )
    io_stream = MultiStream(MarkdownStream(stage_stream), logger_stream)

    absolute_url = attachment_link.absolute_url
    headers = (
        {"api-key": dial_config.api_key.get_secret_value()}
        if absolute_url.startswith(dial_config.dial_url)
        else {}
    )

    file_name, content_type, original_doc_bytes = await load_attachment(
        attachment_link,
        headers,
        download_config=config.download,
    )
    logger.debug(f"Successfully loaded document {file_name} of {content_type}")
    attachment_mime_type, _ = parse_content_type(content_type)

    print(f"File type: {attachment_mime_type}\n", file=io_stream)
    print(
        f"Document size: {format_size(len(original_doc_bytes))}\n",
        file=io_stream,
    )

    mime_type, doc_bytes = await convert_document_if_needed(
        attachment_mime_type,
        original_doc_bytes,
        StreamWithPrefix(io_stream, "Converter: "),
    )

    index_config = config.indexing
    async with asyncio.TaskGroup() as tg:
        multimodal_index_task = None
        if index_config.multimodal_index is not None:
            multimodal_index_task = tg.create_task(
                MultimodalRetriever.build_index(
                    dial_config,
                    dial_limited_resources,
                    index_config.multimodal_index,
                    mime_type,
                    doc_bytes,
                    StreamWithPrefix(io_stream, "MultimodalRetriever: "),
                )
            )

        description_index_task = None
        if index_config.description_index is not None:
            description_index_task = tg.create_task(
                DescriptionRetriever.build_index(
                    dial_config,
                    dial_limited_resources,
                    index_config.description_index,
                    doc_bytes,
                    mime_type,
                    StreamWithPrefix(io_stream, "DescriptionRetriever: "),
                )
            )

        # TODO: try to move is_image check to the parse_document since another loader is not exposed here from the document_loaders.py
        if is_image(content_type):
            chunks_list = [get_default_image_chunk(attachment_link)]
        else:
            chunks = await parse_document(
                StreamWithPrefix(io_stream, "Parser: "),
                doc_bytes,
                mime_type,
                attachment_link,
                attachment_mime_type,
                index_config.parser,
            )
            chunks_list = await build_chunks_list(chunks)

        text_index_task = tg.create_task(
            BM25Retriever.build_index(
                chunks_list, StreamWithPrefix(io_stream, "BM25Retriever: ")
            )
        )

        embeddings_index_task = tg.create_task(
            SemanticRetriever.build_index(
                chunks_list, StreamWithPrefix(io_stream, "SemanticRetriever: ")
            )
        )

    multimodal_index = (
        multimodal_index_task.result() if multimodal_index_task else None
    )
    description_indexes = (
        description_index_task.result() if description_index_task else None
    )

    return DocumentRecord(
        format_version=FORMAT_VERSION,
        index_settings=index_settings,
        chunks=DocList(chunks_list),
        text_index=text_index_task.result(),
        embeddings_index=embeddings_index_task.result(),
        multimodal_embeddings_index=multimodal_index,
        description_embeddings_index=description_indexes,
        document_bytes=doc_bytes,
        mime_type=mime_type,
    )


@contextmanager
def handle_document_processing_error(
    attachment_link: AttachmentLink,
    log_document_links: bool = False,
):
    with convert_and_log_exceptions(logger):
        try:
            yield
        except Exception as e:
            raise DocumentProcessingError(
                attachment_link.dial_link, e, log_document_links
            ) from e


async def load_document(
    request_context: RequestContext,
    task: IndexingTask,
    index_storage: IndexStorage,
    config: RequestConfig,
) -> DocumentRecord:
    attachment_link = task.attachment_link
    with handle_document_processing_error(
        attachment_link, config.log_document_links
    ):
        index_settings = config.indexing.collect_fields_that_rebuild_index()

        choice = request_context.choice

        await check_document_access(request_context, attachment_link, config)

        doc_record = None
        # aidial-sdk does not allow to do stage.close(Status.FAILED) inside with-statement
        try:
            with timed_stage(
                choice, f"Load indexes for '{attachment_link.display_name}'"
            ) as load_stage:
                doc_record = await index_storage.load(task, index_settings)
                if doc_record is None:
                    raise FailStageException()
                print_chunks_stats(load_stage.content_stream, doc_record.chunks)
        except FailStageException:
            pass

        if doc_record is None:
            with timed_stage(
                choice, f"Processing document '{attachment_link.display_name}'"
            ) as doc_stage:
                io_stream = doc_stage.content_stream
                try:
                    doc_record = await load_document_impl(
                        request_context.dial_config,
                        request_context.dial_limited_resources,
                        attachment_link,
                        io_stream,
                        index_settings,
                        config,
                    )
                except InvalidDocumentError as e:
                    doc_stage.append_content(e.message)
                    raise

                print_chunks_stats(io_stream, doc_record.chunks)

            with timed_stage(
                choice, f"Store indexes for '{attachment_link.display_name}'"
            ):
                await index_storage.store(task, doc_record)

        return doc_record


async def load_document_task(
    request_context: RequestContext,
    task: IndexingTask,
    index_storage: IndexStorage,
    config: RequestConfig,
) -> DocumentIndexingResult:
    try:
        doc_record = await load_document(
            request_context, task, index_storage, config
        )
        return DocumentIndexingSuccess(
            task=task,
            doc_record=doc_record,
        )
    except DocumentProcessingError as e:
        return DocumentIndexingFailure(
            task=task,
            exception=e,
        )


async def load_documents(
    request_context: RequestContext,
    tasks: Iterable[IndexingTask],
    index_storage: IndexStorage,
    config: RequestConfig,
) -> List[DocumentIndexingResult]:
    # TODO: Rewrite this function using TaskGroup to cancel all tasks if one of them fails
    # if ignore_document_loading_errors is not set in the config
    return await asyncio.gather(
        *[
            load_document_task(request_context, task, index_storage, config)
            for task in tasks
        ],
    )
