import asyncio
import logging
from contextlib import contextmanager
from typing import Iterable, List

from aidial_sdk.exceptions import InvalidRequestError
from docarray import DocList

from aidial_rag.attachment_link import AttachmentLink
from aidial_rag.configuration_endpoint import Configuration, RequestConfig
from aidial_rag.content_stream import (
    LoggerStream,
    MarkdownStream,
    MultiStream,
    StreamWithPrefix,
    SupportsWriteStr,
)
from aidial_rag.converter import convert_document_if_needed
from aidial_rag.dial_api_client import DialApiClient
from aidial_rag.dial_config import DialConfig
from aidial_rag.document_loaders import (
    load_attachment,
    parse_document,
)
from aidial_rag.document_metadata import (
    FileMetadata,
    load_document_metadata,
)
from aidial_rag.document_record import (
    FORMAT_VERSION,
    Chunk,
    DocumentRecord,
    IndexSettings,
    ModificationMetadata,
    build_chunks_list,
    serialize_document_record,
)
from aidial_rag.document_record_migration import (
    MIN_FORMAT_VERSION,
    deserialize_and_migrate_document_record,
)
from aidial_rag.errors import (
    DocumentProcessingError,
    IndexBaseError,
    IndexIncompatibleError,
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
from aidial_rag.indexing_task import IndexingTask, validate_indexing_task
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


async def load_document_metadata_stage(
    request_context: RequestContext,
    attachment_link: AttachmentLink,
    config: RequestConfig,
):
    # Load document metadata also used to check access to the document
    # If the document is not accessible, an error will be raised
    with timed_stage(
        request_context.choice,
        f"Access document '{attachment_link.display_name}'",
    ) as access_stage:
        try:
            metadata = await load_document_metadata(
                request_context, attachment_link, config.check_access
            )
            logger.info(f"Document metadata loaded successfully: {metadata}")
            return metadata
        except InvalidDocumentError as e:
            access_stage.append_content(e.message)
            raise


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

    file_name = attachment_link.display_name
    file_metadata, original_doc_bytes = await load_attachment(
        attachment_link,
        headers,
        download_config=config.download,
    )
    attachment_mime_type = file_metadata.mime_type
    logger.debug(
        f"Successfully loaded document {file_name} of {attachment_mime_type}"
    )

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
        if is_image(mime_type):
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
        modification_metadata=ModificationMetadata(
            etag=file_metadata.etag,
            last_modified=file_metadata.last_modified,
        ),
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


async def _load_index_stage(
    request_context: RequestContext,
    index_storage: IndexStorage,
    task: IndexingTask,
    index_settings: IndexSettings,
) -> DocumentRecord:
    with timed_stage(
        request_context.choice,
        f"Load indexes for '{task.attachment_link.display_name}'",
    ) as load_stage:
        doc_record_bytes = await index_storage.load(task)
        doc_record = deserialize_and_migrate_document_record(doc_record_bytes)

        if doc_record.index_settings != index_settings:
            raise IndexIncompatibleError(
                f"Index settings mismatch: {doc_record.index_settings}"
            )

        print_chunks_stats(load_stage.content_stream, doc_record.chunks)
        return doc_record


async def _process_document_stage(
    request_context: RequestContext,
    config: Configuration,
    attachment_link: AttachmentLink,
    index_settings: IndexSettings,
):
    with timed_stage(
        request_context.choice,
        f"Processing document '{attachment_link.display_name}'",
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

    return doc_record


async def _store_index_stage(
    request_context: RequestContext,
    index_storage: IndexStorage,
    task: IndexingTask,
    doc_record: DocumentRecord,
) -> None:
    with timed_stage(
        request_context.choice,
        f"Store indexes for '{task.attachment_link.display_name}'",
    ):
        doc_record_bytes = serialize_document_record(doc_record)
        await index_storage.store(task, doc_record_bytes)


def _check_if_document_was_modified(
    config: Configuration,
    modification_metadata: ModificationMetadata,
    metadata: FileMetadata,
):
    if config.request.ignore_file_modification:
        return

    if (
        modification_metadata.etag != metadata.etag
        or modification_metadata.last_modified != metadata.last_modified
    ):
        raise IndexIncompatibleError("Document was modified")


async def load_document(
    request_context: RequestContext,
    task: IndexingTask,
    index_storage: IndexStorage,
    dial_api_client: DialApiClient,
    config: Configuration,
) -> DocumentRecord:
    if config.request.force_indexing and not config.request.allow_indexing:
        raise InvalidRequestError(
            "Cannot force indexing when indexing is disabled."
        )

    attachment_link = task.attachment_link
    with handle_document_processing_error(
        attachment_link, config.log_document_links
    ):
        validate_indexing_task(task, dial_api_client)
        index_settings = config.indexing.collect_fields_that_rebuild_index()

        metadata = await load_document_metadata_stage(
            request_context,
            attachment_link,
            config,
        )

        doc_record = None
        if not config.request.force_indexing:
            try:
                doc_record = await _load_index_stage(
                    request_context, index_storage, task, index_settings
                )
                if (
                    doc_record.format_version is None
                    or doc_record.format_version < MIN_FORMAT_VERSION
                ):
                    raise IndexIncompatibleError(
                        f"Index format version is not supported: {doc_record.format_version}"
                    )

                _check_if_document_was_modified(
                    config, doc_record.modification_metadata, metadata
                )

                if (
                    doc_record.format_version != FORMAT_VERSION
                    and config.request.save_index_on_migration
                ):
                    doc_record.format_version = FORMAT_VERSION
                    await _store_index_stage(
                        request_context, index_storage, task, doc_record
                    )
            except IndexBaseError as e:
                doc_record = None
                if not config.request.allow_indexing:
                    raise e
                if config.log_document_links:
                    logger.warning(
                        f"Failed to load index for {attachment_link}: {e}"
                    )
                else:
                    logger.warning(f"Failed to load index: {e}")

        if doc_record is None:
            doc_record = await _process_document_stage(
                request_context,
                config,
                attachment_link,
                index_settings,
            )
            await _store_index_stage(
                request_context, index_storage, task, doc_record
            )

        return doc_record


async def load_document_task(
    request_context: RequestContext,
    task: IndexingTask,
    index_storage: IndexStorage,
    dial_api_client: DialApiClient,
    config: Configuration,
) -> DocumentIndexingResult:
    try:
        doc_record = await load_document(
            request_context,
            task,
            index_storage,
            dial_api_client,
            config,
        )
        return DocumentIndexingSuccess(
            task=task,
            doc_record=doc_record,
        )
    except DocumentProcessingError as e:
        assert isinstance(e.__cause__, Exception)
        return DocumentIndexingFailure(
            task=task,
            exception=e.__cause__,
        )


async def load_documents(
    request_context: RequestContext,
    tasks: Iterable[IndexingTask],
    index_storage: IndexStorage,
    dial_api_client: DialApiClient,
    config: Configuration,
) -> List[DocumentIndexingResult]:
    # TODO: Rewrite this function using TaskGroup to cancel all tasks if one of them fails
    # if ignore_document_loading_errors is not set in the config
    return await asyncio.gather(
        *[
            load_document_task(
                request_context, task, index_storage, dial_api_client, config
            )
            for task in tasks
        ],
    )
