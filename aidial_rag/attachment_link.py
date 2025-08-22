from collections.abc import Generator
from pathlib import PurePosixPath
from typing import List
from urllib.parse import unquote, urljoin, urlparse

from aidial_sdk.chat_completion import Message, Role
from pydantic import BaseModel

from aidial_rag.errors import InvalidAttachmentError
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

    absolute_metadata_url = urljoin(
        request_context.dial_metadata_base_url, link, allow_fragments=True
    )
    return to_dial_relative_url(request_context, absolute_metadata_url)


class AttachmentLink(BaseModel):
    """
    Link to the attached document in the Dial could be an absolute URL or a relative URL for the Dial file API.
    This class represents both of them.

    Fields:
    - dial_link: The original URL. Could be relative. Should be used to refer the attachment in Dial API or messages.
    - absolute_url: The absolute URL. Should be used to get the content of the attached document.
    - display_name: The name of the attached document to display to the User. Does not include the bucket name for the documents in the Dial filesystem.
    - dial_metadata_url: The URL to get the metadata for the attached document. Could be None if the document is not in the Dial filesystem.
    - type: The MIME type of the attached document. Could be None if not specified.
    - reference_url: The URL to the original document, if it is different from the dial_link.
    """

    dial_link: str
    absolute_url: str
    display_name: str
    dial_metadata_url: str | None = None
    type: str | None = None
    reference_url: str | None = None

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
        cls,
        request_context: RequestContext,
        link: str,
        type: str | None = None,
        reference_url: str | None = None,
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
            type=type,
            reference_url=reference_url,
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
            link = AttachmentLink.from_link(
                request_context,
                attachment.url,
                attachment.type,
                attachment.reference_url,
            )
            yield link
