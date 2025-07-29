from collections.abc import Generator
from pathlib import PurePosixPath
from typing import List, Tuple
from urllib.parse import unquote, urljoin, urlparse

from aidial_sdk import HTTPException
from aidial_sdk.chat_completion import Message, Role
from pydantic import BaseModel
from requests.exceptions import Timeout

from aidial_rag.errors import DocumentProcessingError, InvalidAttachmentError
from aidial_rag.request_context import RequestContext


def to_absolute_url(request_context: RequestContext, link: str) -> str:
    """
    Converts link to an absolute URL.
    The link could be an absolute URL or a relative URL for the Dial API base url.
    """
    return urljoin(request_context.dial_base_url, link, allow_fragments=True)


def to_relative_url(absolute_url: str, base_url: str) -> str:
    """
    Converts absolute_url to a relative URL with specified base_url.
    """
    parsed_base_url = urlparse(base_url)
    parsed_url = urlparse(absolute_url)
    assert parsed_url.scheme == parsed_base_url.scheme
    assert parsed_url.netloc == parsed_base_url.netloc
    assert parsed_url.path.startswith(parsed_base_url.path)

    relative_url = parsed_url._replace(
        scheme="",
        netloc="",
        path=str(
            PurePosixPath(parsed_url.path).relative_to(parsed_base_url.path)
        ),
    )
    return relative_url.geturl()


def to_dial_relative_url(
    request_context: RequestContext, absolute_url: str
) -> str:
    """
    Converts absolute_url to a relative URL for the Dial API base url.
    """
    # If the link is not a Dial API URL, then we cannot convert it to a relative URL.
    if not request_context.is_dial_url(absolute_url):
        return absolute_url

    return to_relative_url(absolute_url, request_context.dial_base_url)


def to_dial_metadata_url(
    request_context: RequestContext, absolute_url: str, link: str
) -> str | None:
    """
    Converts link to a metadata URL.
    """
    # If the link is not a Dial API URL, then we cannot request metadata for it.
    if not request_context.is_dial_url(absolute_url):
        return None

    return urljoin(
        request_context.dial_metadata_base_url, link, allow_fragments=True
    )


class AttachmentLink(BaseModel):
    """
    Link to the attached document in the Dial could be an absolute URL or a relative URL for the Dial file API.
    This class represents both of them.

    Fields:
    - dial_link: The original URL. Could be relative. Should be used to refer the attachment in Dial API or messages.
    - absolute_url: The absolute URL. Should be used to get the content of the attached document.
    - display_name: The name of the attached document to display to the User. Does not include the bucket name for the documents in the Dial filesystem.
    - dial_metadata_url: The URL to get the metadata for the attached document. Could be None if the document is not in the Dial filesystem.
    """

    dial_link: str
    absolute_url: str
    display_name: str
    dial_metadata_url: str | None = None

    def __str__(self) -> str:
        return self.dial_link

    @property
    def is_dial_document(self) -> bool:
        return self.dial_metadata_url is not None

    @staticmethod
    def _get_display_name(link: str) -> str:
        parsed_url = urlparse(link)
        if parsed_url.netloc:
            return link

        path = PurePosixPath(parsed_url.path)
        if path.is_absolute():
            raise InvalidAttachmentError(f"Dial link is not relative: {link}")

        if len(path.parents) < 3:
            raise InvalidAttachmentError(f"Missing bucket in Dial link: {link}")

        assert str(path.parents[-1]) == "."
        if str(path.parents[-2]) != "files":
            raise InvalidAttachmentError(
                f"Dial link is not a link to files: {link}"
            )

        bucket = path.parents[-3]
        relative_path = PurePosixPath(path.relative_to(bucket))
        display_path = PurePosixPath(
            *[unquote(part) for part in relative_path.parts]
        )
        return str(display_path)

    @classmethod
    def from_link(
        cls, request_context: RequestContext, link: str
    ) -> "AttachmentLink":
        absolute_url = to_absolute_url(request_context, link)
        if request_context.is_dial_url(absolute_url) and absolute_url == link:
            # If the link is already a Dial file API URL, then we need to restore the relative URL
            # to be able to access the metadata and get the display name.
            link = to_dial_relative_url(request_context, absolute_url)
        return cls(
            dial_link=link,
            absolute_url=absolute_url,
            display_name=cls._get_display_name(link),
            dial_metadata_url=to_dial_metadata_url(
                request_context, absolute_url, link
            ),
        )


def get_attachment_links(
    request_context: RequestContext, messages: List[Message]
) -> Generator[AttachmentLink, None, None]:
    for message in messages:
        if message.role != Role.USER:
            continue
        if not message.custom_content or not message.custom_content.attachments:
            continue

        for attachment in message.custom_content.attachments:
            # Attachment url field is a link which could be an absolute URL or relative URL for the Dial file API.
            assert attachment.url is not None
            link = AttachmentLink.from_link(request_context, attachment.url)
            yield link


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


def _get_user_facing_error_message(
    exception: BaseException,
) -> Generator[str, None, None]:
    for leaf_exception in _iter_leaf_exceptions(exception):
        if isinstance(leaf_exception, HTTPException):
            yield leaf_exception.message.replace("\n", " ")
        elif isinstance(leaf_exception, Timeout):
            yield "Timed out during download"
        else:
            yield "Internal error"


def format_document_loading_errors(
    errors: List[Tuple[BaseException, AttachmentLink]],
) -> str:
    return "\n".join(
        [
            "I'm sorry, but I can't process the documents because of the following errors:\n",
            "|Document|Error|",
            "|---|---|",
            *(
                f"|{link.display_name}|{message}|"
                for exc, link in errors
                for message in _get_user_facing_error_message(exc)
            ),
            "\nPlease try again with different documents.",
        ]
    )


def _get_status_codes(exception: BaseException) -> Generator[int, None, None]:
    for leaf_exception in _iter_leaf_exceptions(exception):
        if isinstance(leaf_exception, HTTPException):
            yield leaf_exception.status_code
        else:
            yield 500


def create_document_loading_exception(
    errors: List[Tuple[BaseException, AttachmentLink]],
) -> HTTPException:
    # The min is used to make 4xx errors more important than 5xx errors,
    # because we want to prioritize errors that are caused by the User's input.
    status_code = min(
        status_code for e, _ in errors for status_code in _get_status_codes(e)
    )

    error_message = format_document_loading_errors(errors)
    return HTTPException(
        status_code=status_code,
        message=error_message,
        display_message=error_message,
    )
