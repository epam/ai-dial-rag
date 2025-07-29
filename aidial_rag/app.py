import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Tuple, assert_never

from aidial_sdk import DIALApp, HTTPException
from aidial_sdk.chat_completion import (
    ChatCompletion,
    Choice,
    Message,
    Request,
    Response,
)
from langchain.schema import BaseRetriever, Document
from langchain_core.runnables import Runnable

from aidial_rag.app_config import AppConfig
from aidial_rag.attachment_link import (
    AttachmentLink,
    create_document_loading_exception,
    format_document_loading_errors,
    get_attachment_links,
)
from aidial_rag.base_config import merge_config
from aidial_rag.commands import (
    Commands,
    commands_to_config_dict,
    process_commands,
)
from aidial_rag.config_digest import ConfigDigest
from aidial_rag.configuration_endpoint import (
    Configuration,
    RequestConfig,
    RequestType,
    get_configuration,
)
from aidial_rag.dial_api_client import DialApiClient, create_dial_api_client
from aidial_rag.document_record import Chunk, DocumentRecord
from aidial_rag.documents import load_documents
from aidial_rag.index_record import ChunkMetadata, RetrievalType
from aidial_rag.index_storage import IndexStorage, link_to_index_url
from aidial_rag.indexing_task import IndexingTask
from aidial_rag.qa_chain import generate_answer
from aidial_rag.query_chain import create_get_query_chain
from aidial_rag.repository_digest import (
    RepositoryDigest,
    read_repository_digest,
)
from aidial_rag.request_context import RequestContext, create_request_context
from aidial_rag.resources.cpu_pools import init_cpu_pools
from aidial_rag.retrieval_chain import create_retrieval_chain
from aidial_rag.stages import RetrieverStage
from aidial_rag.transform_history import transform_history
from aidial_rag.utils import profiler_if_enabled, timed_stage

APP_NAME = "dial-rag"

REPOSITORY_DIGEST_PATH = "/opt/repository-digest.json"

logger = logging.getLogger(__name__)


def doc_to_attach(
    metadata_doc: Document, document_records: List[DocumentRecord], index=None
) -> dict | None:
    metadata = ChunkMetadata(**metadata_doc.metadata)

    doc_record: DocumentRecord = document_records[metadata["doc_id"]]
    chunk: Chunk = doc_record.chunks[metadata["chunk_id"]]
    if index is None:
        index = f"{metadata['doc_id']}.{metadata['chunk_id']}"

    if metadata["retrieval_type"] == RetrievalType.TEXT:
        type = "text/markdown"
        data = f"{chunk.text}"
    elif metadata["retrieval_type"] == RetrievalType.IMAGE:
        data = (
            f"[Image of the page {chunk.metadata['page_number']}]"
            if "page_number" in chunk.metadata
            else "[Image]"
        )
        type = "text/markdown"

    # aidial_sdk has a bug with empty string as an attachment data
    # https://github.com/epam/ai-dial-sdk/issues/167
    data = data or " "

    return {
        "type": type,
        "data": data,
        "title": "[{index}] {source_display_name}".format(
            **chunk.metadata, index=index
        ),
        "reference_url": chunk.metadata["source"],
    }


def process_load_errors(
    docs_and_errors: List[DocumentRecord | BaseException],
    attachment_links: List[AttachmentLink],
) -> Tuple[
    List[DocumentRecord],
    List[AttachmentLink],
    List[Tuple[BaseException, AttachmentLink]],
]:
    document_records: List[DocumentRecord] = []
    document_records_links: List[AttachmentLink] = []
    loading_errors: List[Tuple[BaseException, AttachmentLink]] = []

    for doc_or_error, link in zip(
        docs_and_errors, attachment_links, strict=True
    ):
        if isinstance(doc_or_error, DocumentRecord):
            document_records.append(doc_or_error)
            document_records_links.append(link)

        elif isinstance(doc_or_error, Exception):
            loading_errors.append((doc_or_error, link))
        else:
            # If the error is BaseException, but not Exception:
            # GeneratorExit, KeyboardInterrupt, SystemExit etc.
            raise HTTPException(
                message=f"Internal error during document loading: {str(doc_or_error)}",
                status_code=500,
            ) from doc_or_error

    return document_records, document_records_links, loading_errors


def create_indexing_tasks(
    attachment_links: List[AttachmentLink],
    dial_api_client: DialApiClient,
) -> List[IndexingTask]:
    return [
        IndexingTask(
            attachment_link=link,
            index_url=link_to_index_url(link, dial_api_client.bucket_id),
        )
        for link in attachment_links
    ]


async def _run_retrieval(
    choice: Choice,
    request_config: RequestConfig,
    retrieval_chain: Runnable[Dict[str, Any], Dict[str, Any]],
    messages: List[Message],
    document_records: List[DocumentRecord],
    document_records_links: List[AttachmentLink],
):
    chain_input = {
        "chat_history": transform_history(messages),
        "chat_chain_config": request_config.qa_chain.chat_chain,
        "doc_records": document_records,
        "doc_records_links": document_records_links,
    }

    retrieval_results = await retrieval_chain.pick("retrieval_results").ainvoke(
        chain_input
    )

    choice.add_attachment(
        title="Retrieval results",
        type=retrieval_results.CONTENT_TYPE,
        data=retrieval_results.model_dump_json(indent=2),
    )


async def _run_rag(
    request_context: RequestContext,
    request_config: RequestConfig,
    retrieval_chain: Runnable[Dict[str, Any], Dict[str, Any]],
    messages: List[Message],
    document_records: List[DocumentRecord],
    document_records_links: List[AttachmentLink],
):
    choice = request_context.choice
    chain_input = {
        "chat_history": transform_history(messages),
        "chat_chain_config": request_config.qa_chain.chat_chain,
        "doc_records": document_records,
        "doc_records_links": document_records_links,
    }

    reference_items = await generate_answer(
        request_context=request_context,
        retrieval_chain=retrieval_chain,
        chain_input=chain_input,
        content_callback=choice.append_content,
    )

    document_records = chain_input["doc_records"]
    # Answer has already been streamed to the user, so we don't need to do anything here.
    for i, reference_item in enumerate(reference_items):
        if attachment := doc_to_attach(
            reference_item, document_records, index=(i + 1)
        ):
            choice.add_attachment(**attachment)


class DialRAGApplication(ChatCompletion):
    app_config: AppConfig
    enable_debug_commands: bool
    repository_digest: RepositoryDigest

    def __init__(self, app_config: AppConfig):
        self.app_config = app_config
        self.enable_debug_commands = app_config.enable_debug_commands
        self.repository_digest = read_repository_digest(REPOSITORY_DIGEST_PATH)
        logger.info(
            f"Repository digest version: {self.repository_digest.version}"
        )
        logger.info(
            f"Repository digest status: {self.repository_digest.status}"
        )
        logger.info(f"App config: {self.app_config.model_dump_json(indent=2)}")
        if self.app_config.request.qa_chain.chat_chain.system_prompt_template_override:
            logger.warning(
                f"The system prompt is set to a custom value: "
                f"{self.app_config.request.qa_chain.chat_chain.system_prompt_template_override}."
            )
        super().__init__()

    def _merge_config_sources(
        self, request: Request, commands: Commands
    ) -> ConfigDigest:
        configuration = merge_config(
            Configuration(),
            self.app_config.request.model_dump(exclude_none=True),
        )

        custom_configuration_dict = get_configuration(request)
        if custom_configuration_dict:
            logger.info(
                f"Request config from configuration: {custom_configuration_dict}"
            )
            configuration = merge_config(
                configuration, custom_configuration_dict
            )

        commands_config_dict = commands_to_config_dict(commands)
        if commands_config_dict:
            logger.info(f"Commands config: {commands_config_dict}")
            configuration = merge_config(configuration, commands_config_dict)

        return ConfigDigest(
            app_config_path=str(self.app_config.config_path),
            configuration=configuration,
            from_custom_configuration=custom_configuration_dict,
            from_commands=commands_config_dict,
        )

    async def chat_completion(
        self, request: Request, response: Response
    ) -> None:
        loop = asyncio.get_running_loop()
        with create_request_context(
            self.app_config.dial_url, request, response
        ) as request_context:
            choice = request_context.choice
            assert choice is not None

            messages, commands = await loop.run_in_executor(
                None,
                process_commands,
                request.messages,
                self.enable_debug_commands,
            )
            config_digest = self._merge_config_sources(request, commands)
            request_config = config_digest.configuration

            choice.set_state(
                {
                    "repository_digest": self.repository_digest.model_dump(),
                    "config_digest": config_digest.model_dump(),
                }
            )

            attachment_links = list(
                get_attachment_links(request_context, messages)
            )

            dial_api_client = await create_dial_api_client(request_context)
            index_storage = IndexStorage(
                dial_api_client,
                self.app_config.index_storage,
            )

            # TODO: Allow to specify desired index URLs in the request
            indexing_tasks = create_indexing_tasks(
                attachment_links, dial_api_client
            )

            docs_and_errors = await load_documents(
                request_context,
                indexing_tasks,
                index_storage,
                config=request_config,
            )
            document_records, document_records_links, loading_errors = (
                process_load_errors(docs_and_errors, attachment_links)
            )

            if (
                len(loading_errors) > 0
                and not request_config.ignore_document_loading_errors
            ):
                if request_config.request.type != RequestType.RAG:
                    raise create_document_loading_exception(loading_errors)

                choice.append_content(
                    format_document_loading_errors(loading_errors)
                )
                return

            last_message_content = messages[-1].content
            if last_message_content is None:
                return
            if not isinstance(last_message_content, str):
                raise HTTPException(
                    message="Message content is not a string",
                    status_code=400,
                )
            if not last_message_content.strip():
                return

            with timed_stage(choice, "Prepare indexes for search"):

                def _make_retrieval_stage(retriever: BaseRetriever, stage_name):
                    return RetrieverStage(
                        choice=choice,
                        stage_name=stage_name,
                        document_records=document_records,
                        retriever=retriever,
                        doc_to_attach=doc_to_attach,
                    )

                query_chain = create_get_query_chain(
                    request_context, request_config.qa_chain.query_chain
                )

                retrieval_chain = await create_retrieval_chain(
                    dial_config=request_context.dial_config,
                    indexing_config=request_config.indexing,
                    document_records=document_records,
                    query_chain=query_chain,
                    make_retrieval_stage=_make_retrieval_stage,
                )

            with profiler_if_enabled(choice, request_config.use_profiler):
                request_type = request_config.request.type
                if request_type == RequestType.RETRIEVAL:
                    return await _run_retrieval(
                        choice,
                        request_config,
                        retrieval_chain,
                        messages,
                        document_records,
                        document_records_links,
                    )
                elif request_type == RequestType.RAG:
                    return await _run_rag(
                        request_context,
                        request_config,
                        retrieval_chain,
                        messages,
                        document_records,
                        document_records_links,
                    )
                else:
                    assert_never(request_type)

    async def configuration(self, request):
        return Configuration.model_json_schema()


def lifespan(app_config: AppConfig):
    @asynccontextmanager
    async def _lifespan(app):
        await init_cpu_pools(app_config.cpu_pools)
        yield

    return _lifespan


def create_app(app_config: AppConfig) -> DIALApp:
    logger.debug("Creating app")
    app = DIALApp(
        app_config.dial_url,
        propagate_auth_headers=False,
        telemetry_config=None,  # Telemetry is initialized in log_config
        lifespan=lifespan(app_config),
        add_healthcheck=True,
    )

    logger.debug("App created successfully")
    app.add_chat_completion(APP_NAME, DialRAGApplication(app_config))
    return app
