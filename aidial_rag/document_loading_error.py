from collections.abc import Generator
from typing import List

from aidial_sdk import HTTPException
from requests.exceptions import Timeout

from aidial_rag.documents import DocumentIndexingResult


def _get_status_code(exc: BaseException) -> int:
    if isinstance(exc, HTTPException):
        return exc.status_code
    return 500


def _get_user_facing_error_message(
    exc: BaseException,
) -> Generator[str, None, None]:
    if isinstance(exc, HTTPException):
        yield exc.message.replace("\n", " ")
    elif isinstance(exc, Timeout):
        yield "Timed out during download"
    elif isinstance(exc, BaseExceptionGroup):
        # We could have multiple errors in the group because of the concurrent processing.
        for inner_exception in exc.exceptions:
            yield from _get_user_facing_error_message(inner_exception)
    else:
        yield "Internal error"


def format_document_loading_errors(
    document_indexing_results: List[DocumentIndexingResult],
) -> str:
    return "\n".join(
        [
            "I'm sorry, but I can't process the documents because of the following errors:\n",
            "|Document|Error|",
            "|---|---|",
            *(
                f"|{result.attachment_link.display_name}|{message}|"
                for result in document_indexing_results
                if result.exception is not None
                for message in _get_user_facing_error_message(result.exception)
            ),
            "\nPlease try again with different documents.",
        ]
    )


def create_document_loading_exception(
    document_indexing_results: List[DocumentIndexingResult],
) -> HTTPException:
    # The max is used to make 5xx errors more important than 4xx errors.
    status_code = 0
    for result in document_indexing_results:
        if result.exception is not None:
            status_code = max(status_code, _get_status_code(result.exception))
    assert status_code != 0, "No errors in document indexing results"

    error_message = format_document_loading_errors(document_indexing_results)
    return HTTPException(
        status_code=status_code,
        message=error_message,
        display_message=error_message,
    )
