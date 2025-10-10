from email.policy import EmailPolicy
from email.utils import parsedate_to_datetime
from typing import Mapping

import aiohttp
from pydantic import BaseModel, Field

from aidial_rag.attachment_link import AttachmentLink
from aidial_rag.document_loaders_config import HttpClientConfig
from aidial_rag.errors import InvalidDocumentError
from aidial_rag.request_context import RequestContext


class FileMetadata(BaseModel):
    mime_type: str
    content_length: int | None = None
    etag: str | None = None
    last_modified: int | None = None


class DialFileMetadata(BaseModel):
    """Dial file metadata response: https://dialx.ai/dial_api#tag/Files/operation/getFileMetadata"""

    etag: str | None = None
    content_length: int | None = Field(alias="contentLength", default=None)
    content_type: str | None = Field(alias="contentType", default=None)


def get_mime_type(content_type: str | None) -> str:
    """Get mime type from content type string, e.g. 'text/html; charset=UTF-8' -> 'text/html'"""
    if content_type is None:
        return "application/octet-stream"

    header = EmailPolicy.header_factory("content-type", content_type)
    return header.content_type


def _convert_file_metadata(
    dial_file_metadata: DialFileMetadata,
) -> FileMetadata:
    return FileMetadata(
        content_length=dial_file_metadata.content_length,
        mime_type=get_mime_type(dial_file_metadata.content_type),
        etag=dial_file_metadata.etag,
        # Do not use updated_at as last_modified
        # We want to rely on etag for the Dial files
        last_modified=None,
    )


async def _load_dial_document_metadata(
    request_context: RequestContext,
    attachment_link: AttachmentLink,
    config: HttpClientConfig,
) -> FileMetadata:
    if not attachment_link.is_dial_document:
        raise ValueError("Not a Dial document")

    metadata_url = attachment_link.dial_metadata_url
    assert metadata_url is not None

    headers = request_context.get_file_access_headers(metadata_url)
    async with aiohttp.ClientSession(
        timeout=config.get_client_timeout()
    ) as session:
        async with session.get(metadata_url, headers=headers) as response:
            if not response.ok:
                error_message = f"{response.status} {response.reason}"
                raise InvalidDocumentError(error_message)
            response_json = await response.json()

            dial_file_metadata = DialFileMetadata.model_validate(response_json)
            return _convert_file_metadata(dial_file_metadata)


def _get_content_length(headers: Mapping[str, str]) -> int | None:
    content_length = headers.get("Content-Length")
    if content_length is None:
        return None
    return int(content_length)


def parse_last_modified(last_modified: str) -> int:
    return int(parsedate_to_datetime(last_modified).timestamp())


def _get_last_modified(headers: Mapping[str, str]) -> int | None:
    last_modified = headers.get("Last-Modified")
    if last_modified is None:
        return None
    return parse_last_modified(last_modified)


def create_file_metadata_from_headers(
    headers: Mapping[str, str],
) -> FileMetadata:
    return FileMetadata(
        content_length=_get_content_length(headers),
        mime_type=get_mime_type(headers.get("Content-Type")),
        etag=headers.get("ETag"),
        last_modified=_get_last_modified(headers),
    )


async def _load_external_document_metadata(
    attachment_link: AttachmentLink,
    config: HttpClientConfig,
) -> FileMetadata:
    if attachment_link.is_dial_document:
        raise ValueError("Not an external document")

    url = attachment_link.dial_link

    async with aiohttp.ClientSession(
        timeout=config.get_client_timeout()
    ) as session:
        async with session.head(url) as response:
            response.raise_for_status()
            return create_file_metadata_from_headers(response.headers)


async def load_document_metadata(
    request_context: RequestContext,
    attachment_link: AttachmentLink,
    config: HttpClientConfig,
) -> FileMetadata:
    if attachment_link.is_dial_document:
        return await _load_dial_document_metadata(
            request_context, attachment_link, config
        )
    else:
        return await _load_external_document_metadata(attachment_link, config)
