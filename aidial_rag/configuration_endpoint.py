from enum import StrEnum
from typing import Dict, Optional

from aidial_sdk import HTTPException
from aidial_sdk.chat_completion import Request
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
    INDEXING = "indexing"


class RetrievalParameters(BaseConfig):
    """Parameters for the retrieval request."""

    allow_reindexing: bool = Field(
        default=True,
        description="Allow reindexing of the documents if they are not indexed yet or if the index format mismatches.",
    )


class IndexingParameters(BaseConfig):
    """Parameters for the indexing request."""

    index_mapping: Dict[str, str] = Field(
        default={},
        description="The mapping of the document URLs to the index URLs. The index URLs are the URLs where the index files will be stored in the Dial storage. If not specified, the Dial RAG will use the current algorithm to calculate the index file path in the Dial RAG bucket based on the document URLs.",
    )


class ApiRequest(BaseModel):
    """Configuration for the Dial RAG API."""

    type: RequestType = Field(
        default=RequestType.RAG,
        description="Type of the request for the Dial RAG service.",
    )
    parameters: Optional[RetrievalParameters | IndexingParameters] = Field(
        default=None,
        description="Parameters for the request, depending on the request type.",
    )


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


def get_configuration(request: Request) -> dict:
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
