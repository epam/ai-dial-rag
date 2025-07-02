import asyncio
import logging
from contextlib import asynccontextmanager
from typing import List, Tuple

from aidial_sdk import DIALApp, HTTPException
from aidial_sdk.chat_completion import (
    ChatCompletion,
    Message,
    Request,
    Response,
)
from langchain.schema import Document

from aidial_rag.app_config import AppConfig
from aidial_rag.attachment_link import (
    AttachmentLink,
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
    RequestType,
    get_configuration,
)
from aidial_rag.document_loading_error import (
    create_document_loading_exception,
    format_document_loading_errors,
)
from aidial_rag.document_record import Chunk, DocumentRecord
from aidial_rag.documents import (
    DocumentIndexingResult,
    has_document_loading_errors,
    load_documents,
)
from aidial_rag.index_record import ChunkMetadata, RetrievalType
from aidial_rag.index_storage import IndexStorage
from aidial_rag.qa_chain import generate_answer
from aidial_rag.query_chain import create_get_query_chain
from aidial_rag.repository_digest import (
    RepositoryDigest,
    read_repository_digest,
)
from aidial_rag.request_context import RequestContext, create_request_context
from aidial_rag.resources.cpu_pools import init_cpu_pools
from aidial_rag.retireval_chain import create_retrieval_chain
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


def _get_last_message(messages: List[Message]) -> str:
    last_message_content = messages[-1].content
    if last_message_content is None:
        return ""
    if not isinstance(last_message_content, str):
        raise HTTPException(
            message="Message content is not a string",
            status_code=400,
        )
    return last_message_content


def _collect_document_records(
    document_indexing_results: List[DocumentIndexingResult],
) -> List[DocumentRecord]:
    return [
        result.doc_record
        for result in document_indexing_results
        if result.doc_record is not None
    ]


async def _run_retrieval(choice, retrieval_chain, chain_input):
    retrieval_chain.pick("retrieval_results")
    chain_results = await retrieval_chain.ainvoke(chain_input)
    retrieval_results = chain_results["retrieval_results"]
    choice.add_attachment(
        title="Retrieval results",
        type=retrieval_results.CONTENT_TYPE,
        data=retrieval_results.model_dump_json(indent=2),
    )


async def _run_rag(
    request_context, choice, request_config, retrieval_chain, chain_input
):
    reference_items = await generate_answer(
        request_context=request_context,
        request_config=request_config,
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
    index_storage: IndexStorage
    enable_debug_commands: bool
    repository_digest: RepositoryDigest

    def __init__(self, app_config: AppConfig):
        self.app_config = app_config
        self.index_storage = IndexStorage(
            self.app_config.dial_url, self.app_config.index_storage
        )
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
        request_config = Configuration()
        request_config = merge_config(
            request_config,
            self.app_config.request.model_dump(exclude_none=True),
        )

        custom_configuration_dict = get_configuration(request)
        if custom_configuration_dict:
            logger.info(
                f"Request config from configuration: {custom_configuration_dict}"
            )
            request_config = merge_config(
                request_config, custom_configuration_dict
            )

        commands_config_dict = commands_to_config_dict(commands)
        if commands_config_dict:
            logger.info(f"Commands config: {commands_config_dict}")
            request_config = merge_config(request_config, commands_config_dict)

        return ConfigDigest(
            app_config_path=str(self.app_config.config_path),
            request_config=request_config,
            from_custom_configuration=custom_configuration_dict,
            from_commands=commands_config_dict,
        )

    async def _load_documents(
        self,
        request_context,
        attachment_links: List[AttachmentLink],
        config: Configuration,
    ):
        try:
            return await load_documents(
                request_context,
                attachment_links,
                self.index_storage,
                config=config,
            )
        except BaseException as e:
            # If the error is BaseException, but not Exception:
            # GeneratorExit, KeyboardInterrupt, SystemExit etc.
            raise HTTPException(
                message=f"Internal error during document loading: {str(e)}",
                status_code=500,
            ) from e

    async def _process_request_input(
        self,
        request: Request,
        request_context: RequestContext,
    ) -> Tuple[List[Message], Configuration]:
        loop = asyncio.get_running_loop()
        choice = request_context.choice
        # Process request parameters here
        messages, commands = await loop.run_in_executor(
            None,
            process_commands,
            request.messages,
            self.enable_debug_commands,
        )
        config_digest = self._merge_config_sources(request, commands)
        request_config = config_digest.request_config

        choice.set_state(
            {
                "repository_digest": self.repository_digest.model_dump(),
                "config_digest": config_digest.model_dump(),
            }
        )

        return messages, request_config

    async def chat_completion(
        self, request: Request, response: Response
    ) -> None:
        with create_request_context(
            self.app_config.dial_url, request, response
        ) as request_context:
            choice = request_context.choice
            assert choice is not None

            messages, request_config = await self._process_request_input(
                request, request_context
            )
            attachment_links = list(
                get_attachment_links(request_context, messages)
            )

            document_indexing_results = await self._load_documents(
                request_context,
                attachment_links,
                config=request_config,
            )

            if (
                has_document_loading_errors(document_indexing_results)
                and not request_config.ignore_document_loading_errors
            ):
                if request_config.request.type != RequestType.RAG:
                    raise create_document_loading_exception(
                        document_indexing_results
                    )
                choice.append_content(
                    format_document_loading_errors(document_indexing_results)
                )
                return

            if request_config.request.type == RequestType.INDEXING:
                # TODO: Return indexing results as attachments
                return

            last_message_content = _get_last_message(messages)
            if not last_message_content.strip():
                return

            document_records = _collect_document_records(
                document_indexing_results
            )

            query_chain = create_get_query_chain(
                request_context, request_config.qa_chain.query_chain
            )

            with timed_stage(choice, "Prepare indexes for search"):

                def _make_stage(retriever, name):
                    return RetrieverStage(
                        choice=request_context.choice,
                        stage_name=name,
                        document_records=document_records,
                        retriever=retriever,
                        doc_to_attach=doc_to_attach,
                    )

                retrieval_chain = await create_retrieval_chain(
                    dial_config=request_context.dial_config,
                    request_config=request_config,
                    document_records=document_records,
                    query_chain=query_chain,
                    make_stage=_make_stage,
                )

            with profiler_if_enabled(choice, request_config.use_profiler):
                chain_input = {
                    "chat_history": transform_history(messages),
                    "chat_chain_config": request_config.qa_chain.chat_chain,
                    "doc_records": document_records,
                }
                if request_config.request.type == RequestType.RETRIEVAL:
                    await _run_retrieval(choice, retrieval_chain, chain_input)
                elif request_config.request.type == RequestType.RAG:
                    await _run_rag(
                        request_context,
                        choice,
                        request_config,
                        retrieval_chain,
                        chain_input,
                    )

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
