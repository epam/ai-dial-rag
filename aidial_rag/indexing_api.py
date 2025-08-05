import re
from typing import ClassVar, Dict, List

from aidial_sdk.chat_completion.request import Attachment
from pydantic import BaseModel, Field

from aidial_rag.indexing_results import (
    DocumentIndexingFailure,
    DocumentIndexingResult,
    DocumentIndexingSuccess,
    get_user_facing_error_message,
)

INDEX_MIME_TYPE = "application/x.aidial-rag-index.v0"

INDEX_MIME_TYPES_REGEX = re.compile(r"^application/x\.aidial-rag-index\..*$")

assert INDEX_MIME_TYPES_REGEX.match(INDEX_MIME_TYPE), (
    f"Invalid INDEX_MIME_TYPE: {INDEX_MIME_TYPE}. "
    f"It should match the regex {INDEX_MIME_TYPES_REGEX.pattern}."
)


class Error(BaseModel):
    """Error that occurred during the indexing process."""

    message: str = Field(
        description="Error message describing the issue that occurred during the indexing process."
    )


class DocumentIndexingResultResponse(BaseModel):
    """Result of the indexing process for a single document."""

    errors: List[Error] = Field(
        default_factory=list,
        description="List of errors that occurred during the indexing process, if any.",
    )


class IndexingResponse(BaseModel):
    CONTENT_TYPE: ClassVar[str] = (
        "application/x.aidial-rag.indexing-response+json"
    )

    indexing_result: Dict[str, DocumentIndexingResultResponse] = Field(
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


def create_indexing_response(
    indexing_results: List[DocumentIndexingResult],
) -> IndexingResponse:
    doc_indexing_results: Dict[str, DocumentIndexingResultResponse] = {}
    for result in indexing_results:
        if isinstance(result, DocumentIndexingFailure):
            doc_indexing_results[result.task.attachment_link.dial_link] = (
                DocumentIndexingResultResponse(
                    errors=[
                        Error(message=get_user_facing_error_message(exception))
                        for exception in result.iter_leaf_exceptions()
                    ]
                )
            )

    return IndexingResponse(indexing_result=doc_indexing_results)


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
            data=indexing_response.model_dump_json(),
        )
    )

    return index_attachments
