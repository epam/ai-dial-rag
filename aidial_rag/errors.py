import logging
from contextlib import contextmanager

import openai
from aidial_sdk import HTTPException


class InvalidDocumentError(HTTPException):
    def __init__(self, message: str):
        super().__init__(message, status_code=400)


class InvalidAttachmentError(HTTPException):
    # Do not inherit from InvalidDocumentError because it is an error for the attachment in the message
    # but not for the document itself. Do not provide a content_message because it is not a user error.
    def __init__(self, message: str):
        super().__init__(message, status_code=400)


class AuthenticationError(HTTPException):
    def __init__(self, message="Authentication error."):
        # We cannot use 401 here because it MUST have WWW-Authenticate header but the current
        # implementation of aidial_sdk does not support headers in aidial_sdk.HTTPException
        # and drops any headers passed to the fastapi.HTTPException
        # even if the exception is thrown before the start of the streaming response.
        super().__init__(message, status_code=400)


class RateLimitError(HTTPException):
    def __init__(self, e: openai.RateLimitError | None):
        message = "Rate limit exceeded."
        if isinstance(e, openai.RateLimitError) and isinstance(e.body, dict):
            message = e.body.get("message", e.message)
        super().__init__(message, status_code=429, display_message=message)


class NotEnoughDailyTokensError(HTTPException):
    def __init__(
        self,
        model_name: str,
        expected: int,
        reserved: int,
        used: int,
        total: int,
    ):
        message = (
            f"Not enough tokens day token limit for the {model_name}."
            f" The expected number of tokens is {expected}, but only {total - used - reserved} tokens are available."
        )
        super().__init__(message, status_code=400)


class DocumentProcessingError(HTTPException):
    """Represents an error that occurred during the processing of the attached document."""

    def __init__(
        self,
        link: str,
        exception: Exception,
        allow_log_document_links: bool = False,
    ):
        if allow_log_document_links:
            message = f"Error on processing document {link}: {exception}"
        else:
            message = f"Error on processing document: {exception}"

        if isinstance(exception, HTTPException):
            super().__init__(message, exception.status_code)
        else:
            super().__init__(message)


@contextmanager
def log_exceptions(logger: logging.Logger | None = None):
    if logger is None:
        logger = logging.getLogger()
    try:
        yield
    except Exception as e:
        logger.exception(e)
        raise


def generate_leaf_exceptions(exc: BaseException, type):
    if isinstance(exc, BaseExceptionGroup):
        for e in exc.exceptions:
            yield from generate_leaf_exceptions(e, type)
    else:
        if isinstance(exc, type):
            yield exc


@contextmanager
def convert_and_log_exceptions(logger: logging.Logger | None = None):
    if logger is None:
        logger = logging.getLogger()
    with log_exceptions(logger):
        try:
            yield
        # In case of ExceptionGroup, known exceptions will be unpacked and re-raised in order of priority
        except* openai.RateLimitError as e:
            rate_limits = next(
                generate_leaf_exceptions(e, openai.RateLimitError)
            )
            assert isinstance(rate_limits, openai.RateLimitError)
            raise RateLimitError(rate_limits) from e
