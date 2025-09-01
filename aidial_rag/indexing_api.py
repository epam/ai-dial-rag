from typing import ClassVar, Dict, List, TypeAlias

from aidial_sdk import HTTPException
from aidial_sdk.chat_completion.request import Attachment
from pydantic import BaseModel, Field

from aidial_rag.error_types import ErrorType
from aidial_rag.index_mime_type import INDEX_MIME_TYPE
from aidial_rag.indexing_results import (
    DocumentIndexingFailure,
    DocumentIndexingResult,
    DocumentIndexingSuccess,
    get_user_facing_error_message,
)


class Error(BaseModel):
    """Error that occurred during the indexing process."""

    message: str = Field(
        description="Error message describing the issue that occurred during the indexing process."
    )
    type: ErrorType | None = Field(
        default=None,
        description="Type of the error that occurred during the indexing process.",
    )


class DocumentIndexingResultResponse(BaseModel):
    """Result of the indexing process for a single document."""

    errors: List[Error] = Field(
        default_factory=list,
        description="List of errors that occurred during the indexing process, if any.",
    )


DocumentsIndexingResult: TypeAlias = Dict[str, DocumentIndexingResultResponse]


class IndexingResponse(BaseModel):
    CONTENT_TYPE: ClassVar[str] = (
        "application/x.aidial-rag.indexing-response+json"
    )

    indexing_result: DocumentsIndexingResult = Field(
        default_factory=dict,
        description="Dictionary mapping document URLs to their indexing results.",
    )


def create_index_attachment(
    indexing_result: DocumentIndexingSuccess,
) -> Attachment:
    """Creates an attachment for the indexing result."""
    return Attachment(
        type=INDEX_MIME_TYPE,
        url=indexing_result.task.index_url,
        reference_url=indexing_result.task.attachment_link.dial_link,
    )


def _get_error_type(exception: BaseException) -> ErrorType | None:
    """Extracts known error types from the exception."""
    if isinstance(exception, HTTPException):
        try:
            return ErrorType(exception.type)
        except ValueError:
            # Skipping the types that are not declared in ErrorType, because the client
            # will not be able to use them.
            # It could be the default "runtime_error", or some type from the model call.
            return None
    return None


def create_documents_indexing_result(
    indexing_results: List[DocumentIndexingResult],
) -> DocumentsIndexingResult:
    doc_indexing_results: DocumentsIndexingResult = {}
    for result in indexing_results:
        if isinstance(result, DocumentIndexingFailure):
            doc_indexing_results[result.task.attachment_link.dial_link] = (
                DocumentIndexingResultResponse(
                    errors=[
                        Error(
                            message=get_user_facing_error_message(exception),
                            type=_get_error_type(exception),
                        )
                        for exception in result.iter_leaf_exceptions()
                    ]
                )
            )
    return doc_indexing_results


def create_indexing_response(
    indexing_results: List[DocumentIndexingResult],
) -> IndexingResponse:
    return IndexingResponse(
        indexing_result=create_documents_indexing_result(indexing_results)
    )


def create_indexing_results_attachments(
    indexing_results: List[DocumentIndexingResult],
) -> List[Attachment]:
    index_attachments: List[Attachment] = []
    for result in indexing_results:
        if isinstance(result, DocumentIndexingSuccess):
            index_attachments.append(create_index_attachment(result))

    indexing_response = create_indexing_response(indexing_results)
    index_attachments.append(
        Attachment(
            title="Indexing results",
            type=indexing_response.CONTENT_TYPE,
            data=indexing_response.model_dump_json(exclude_none=True),
        )
    )

    return index_attachments
