from enum import StrEnum
from typing import Any, Dict

from aidial_sdk import HTTPException
from aidial_sdk.chat_completion import (
    Request,
)
from pydantic import BaseModel, Field, ValidationError

from aidial_rag.base_config import BaseConfig
from aidial_rag.document_loaders import HttpClientConfig
from aidial_rag.indexing_config import IndexingConfig
from aidial_rag.llm import LlmConfig
from aidial_rag.qa_chain_config import ChatChainConfig, QAChainConfig
from aidial_rag.query_chain import QueryChainConfig


class RequestType(StrEnum):
    """Enumeration of request types for Dial RAG."""

    RAG = "rag"
    RETRIEVAL = "retrieval"
    # TODO: Add INDEXING request type


class ApiRequest(BaseModel):
    """Configuration for the Dial RAG API."""

    type: RequestType = Field(
        default=RequestType.RAG,
        description="Type of the request for the Dial RAG service.",
    )
    # TODO: Add request parameters here


class RequestConfig(BaseConfig):
    """Configuration for the request process.
    These config fields could be changed for particular request.
    Changes in the indexing fields will require index rebuilding.
    """

    ignore_document_loading_errors: bool = Field(
        default=False,
        description="Ignore errors during document loading. "
        "Used for Web RAG for the request with multiple documents.",
    )

    use_profiler: bool = Field(
        default=False,
        description="Use profiler to collect performance metrics for the request.",
    )

    log_document_links: bool = Field(
        default=False,
        description="Allows writing the links of the attached documents to the logs "
        "with log levels higher than DEBUG.\n\n"
        "If enabled, Dial RAG will log the links to the documents for log messages "
        "with levels from INFO to CRITICAL where relevant. For example, an ERROR log "
        "message with an exception during document processing will contain the link "
        "to the document.\n\n"
        "If disabled, only log messages with DEBUG level may contain the links to "
        "the documents, to avoid logging sensitive information. For example, the links "
        "to the documents will not be logged for the ERROR log messages with an exception "
        "during document processing.",
    )

    download: HttpClientConfig = Field(
        default=HttpClientConfig(),
        description="Configuration for downloading the attached documents.",
    )

    check_access: HttpClientConfig = Field(
        default=HttpClientConfig(),
        description="Configuration for checking access to the documents in the Dial.",
    )

    indexing: IndexingConfig = Field(default=IndexingConfig())

    qa_chain: QAChainConfig = Field(
        default=QAChainConfig(
            chat_chain=ChatChainConfig(
                llm=LlmConfig(
                    deployment_name="gpt-4o-2024-05-13",
                    max_prompt_tokens=16000,
                ),
            ),
            query_chain=QueryChainConfig(
                llm=LlmConfig(
                    deployment_name="gpt-4o-2024-05-13",
                    max_prompt_tokens=8000,
                ),
            ),
        ),
    )


class Configuration(RequestConfig):
    """Configuration for the Dial RAG service.
    This schema will be provided by the /configuration endpoint and the object
    is accepted as a custom_fields.configuration in the chat completion request.
    Based on the app_config.RequestConfig - request-related part of the application
    configuration, but also includes the fields available in the Dial RAG API only.
    """

    request: ApiRequest = Field(
        default=ApiRequest(),
        description="Configuration for the Dial RAG API request.",
    )


def get_configuration(request: Request) -> Dict[str, Any]:
    if (
        request.custom_fields is None
        or request.custom_fields.configuration is None
    ):
        return {}

    custom_configuration_dict = request.custom_fields.configuration

    # We want to validate the schema, but return the original dict to know which fields are not set
    try:
        Configuration.model_validate(custom_configuration_dict)  # type: ignore
    except ValidationError as e:
        raise HTTPException(
            message=f"Invalid configuration: {e.errors()}",
            status_code=400,
        ) from e

    return custom_configuration_dict
