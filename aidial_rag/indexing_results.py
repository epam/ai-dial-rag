from collections.abc import Generator
from typing import List, Self

from aidial_sdk import HTTPException
from pydantic import BaseModel, ConfigDict, model_validator
from requests.exceptions import Timeout

from aidial_rag.document_record import DocumentRecord
from aidial_rag.errors import DocumentProcessingError
from aidial_rag.indexing_task import IndexingTask


class DocumentIndexingResult(BaseModel):
    """Result of the document indexing task."""

    task: IndexingTask
    doc_record: DocumentRecord | None = None
    exception: Exception | None = None

    def iter_leaf_exceptions(self) -> Generator[BaseException, None, None]:
        """Iterate over leaf exceptions in the result."""
        if self.exception is None:
            return
        yield from _iter_leaf_exceptions(self.exception)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def check_doc_record_or_exception_is_not_none(self) -> Self:
        """Ensure that either doc_record or exception is not None."""
        if self.doc_record is None and self.exception is None:
            raise ValueError("Either doc_record or exception must be provided.")
        return self


def has_document_loading_errors(
    indexing_results: List[DocumentIndexingResult],
) -> bool:
    """Check if there are any errors in the document loading results."""
    return any(result.exception is not None for result in indexing_results)


def _iter_leaf_exceptions(
    exception: BaseException,
) -> Generator[BaseException, None, None]:
    if isinstance(exception, DocumentProcessingError) and exception.__cause__:
        # We want to show the original cause of the error to the User.
        yield from _iter_leaf_exceptions(exception.__cause__)
    elif isinstance(exception, BaseExceptionGroup):
        # We could have multiple errors in the group because of the concurrent processing.
        for inner_exception in exception.exceptions:
            yield from _iter_leaf_exceptions(inner_exception)
    else:
        yield exception


def get_status_code(
    leaf_exception: BaseException,
) -> int:
    if isinstance(leaf_exception, HTTPException):
        return leaf_exception.status_code
    else:
        return 500  # Internal Server Error


def get_user_facing_error_message(
    leaf_exception: BaseException,
) -> str:
    if isinstance(leaf_exception, HTTPException):
        return leaf_exception.message.replace("\n", " ")
    elif isinstance(leaf_exception, Timeout):
        return "Timed out during download"
    else:
        return "Internal error"


def format_document_loading_errors(
    indexing_results: List[DocumentIndexingResult],
) -> str:
    return "\n".join(
        [
            "I'm sorry, but I can't process the documents because of the following errors:\n",
            "|Document|Error|",
            "|---|---|",
            *(
                f"|{result.task.attachment_link.display_name}|{get_user_facing_error_message(exception)}|"
                for result in indexing_results
                for exception in result.iter_leaf_exceptions()
            ),
            "\nPlease try again with different documents.",
        ]
    )


def create_document_loading_exception(
    indexing_results: List[DocumentIndexingResult],
) -> HTTPException:
    # The min is used to make 4xx errors more important than 5xx errors,
    # because we want to prioritize errors that are caused by the User's input.
    status_code = min(
        get_status_code(exception)
        for result in indexing_results
        for exception in result.iter_leaf_exceptions()
    )

    error_message = format_document_loading_errors(indexing_results)
    return HTTPException(
        status_code=status_code,
        message=error_message,
        display_message=error_message,
    )
