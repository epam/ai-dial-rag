import asyncio
import logging
from email.policy import EmailPolicy
from typing import Iterable, List

from docarray import DocList

from aidial_rag.app_config import RequestConfig
from aidial_rag.attachment_link import AttachmentLink
from aidial_rag.content_stream import StreamWithPrefix, SupportsWriteStr
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
from aidial_rag.errors import InvalidDocumentError, convert_and_log_exceptions
from aidial_rag.image_processor.extract_pages import is_image
from aidial_rag.index_storage import IndexStorage
from aidial_rag.print_stats import print_chunks_stats
from aidial_rag.request_context import RequestContext
from aidial_rag.resources.dial_limited_resources import DialLimitedResources
from aidial_rag.retrievers.bm25_retriever import BM25Retriever
from aidial_rag.retrievers.colpali_retriever.colpali_model_resource import (
    ColpaliModelResource,
)
from aidial_rag.retrievers.colpali_retriever.colpali_retriever import (
    ColpaliRetriever,
)
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
    io_stream: SupportsWriteStr,
    index_settings: IndexSettings,
    colpali_model_resource: ColpaliModelResource,
    config: RequestConfig,
) -> DocumentRecord:
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

        colpali_index_task = None
        if index_config.colpali_index is not None:
            colpali_index_task = tg.create_task(
                ColpaliRetriever.build_index(
                    model_resource=colpali_model_resource,
                    colpali_index_config=index_config.colpali_index,
                    stageio=StreamWithPrefix(io_stream, "ColpaliRetriever: "),
                    mime_type=mime_type,
                    original_document=doc_bytes,
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
    colpali_indexes = (
        colpali_index_task.result() if colpali_index_task else None
    )

    return DocumentRecord(
        format_version=FORMAT_VERSION,
        index_settings=index_settings,
        chunks=DocList(chunks_list),
        text_index=text_index_task.result(),
        embeddings_index=embeddings_index_task.result(),
        multimodal_embeddings_index=multimodal_index,
        description_embeddings_index=description_indexes,
        colpali_embeddings_index=colpali_indexes,
        document_bytes=doc_bytes,
        mime_type=mime_type,
    )


async def load_document(
    request_context: RequestContext,
    attachment_link: AttachmentLink,
    index_storage: IndexStorage,
    colpali_model_resource: ColpaliModelResource,
    config: RequestConfig,
) -> DocumentRecord:
    with convert_and_log_exceptions(logger):
        index_settings = config.indexing.collect_fields_that_rebuild_index()

        choice = request_context.choice

        await check_document_access(request_context, attachment_link, config)

        doc_record = None
        # aidial-sdk does not allow to do stage.close(Status.FAILED) inside with-statement
        try:
            with timed_stage(
                choice, f"Load indexes for '{attachment_link.display_name}'"
            ) as load_stage:
                doc_record = await index_storage.load(
                    attachment_link, index_settings, request_context
                )
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
                        colpali_model_resource,
                        config,
                    )
                except InvalidDocumentError as e:
                    doc_stage.append_content(e.message)
                    raise

                print_chunks_stats(io_stream, doc_record.chunks)

            with timed_stage(
                choice, f"Store indexes for '{attachment_link.display_name}'"
            ):
                await index_storage.store(
                    attachment_link, doc_record, request_context
                )

        return doc_record


async def load_documents(
    request_context: RequestContext,
    attachment_links: Iterable[AttachmentLink],
    index_storage: IndexStorage,
    colpali_model_resource: ColpaliModelResource,
    config: RequestConfig,
) -> List[DocumentRecord | BaseException]:
    return await asyncio.gather(
        *[
            load_document(
                request_context,
                attachment_link,
                index_storage,
                colpali_model_resource,
                config,
            )
            for attachment_link in attachment_links
        ],
        return_exceptions=True,
    )
