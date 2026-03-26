from unittest.mock import MagicMock

import pytest
from aidial_sdk.chat_completion import Choice
from aioresponses import aioresponses
from pydantic import SecretStr

from aidial_rag.attachment_link import AttachmentLink
from aidial_rag.dial_config import DialConfig
from aidial_rag.document_loaders_config import HttpClientConfig
from aidial_rag.document_metadata import get_mime_type, load_document_metadata
from aidial_rag.request_context import RequestContext
from aidial_rag.resources.dial_limited_resources import DialLimitedResources
from tests.utils.user_limits_mock import user_limits_mock


@pytest.fixture
def request_context():
    return RequestContext(
        dial_config=DialConfig(
            dial_url="http://core.dial",
            api_key=SecretStr(""),
        ),
        choice=Choice(queue=MagicMock(), choice_index=0),
        dial_limited_resources=DialLimitedResources(user_limits_mock()),
    )


@pytest.mark.asyncio
async def test_load_document_metadata_dial(request_context):
    attachment_link = AttachmentLink.from_link(
        request_context,
        "files/bucket/my%20folder/my%20file%20(1).pdf",
    )

    with aioresponses() as mock:
        mock.get(
            "http://core.dial/v1/metadata/files/bucket/my%20folder/my%20file%20(1).pdf",
            payload={
                "name": "my file (1).pdf",
                "author": "user123",
                "parentPath": "my folder",
                "bucket": "bucket",
                "url": "files/bucket/my%20folder/my%20file%20(1).pdf",
                "nodeType": "ITEM",
                "resourceType": "FILE",
                "createdAt": 1758648364,
                "updatedAt": 1758648364,
                "contentLength": 12345,
                "contentType": "application/pdf",
                "etag": '"abc123"',
            },
        )

        file_metadata = await load_document_metadata(
            request_context=request_context,
            attachment_link=attachment_link,
            config=HttpClientConfig(),
        )

    assert file_metadata is not None
    assert file_metadata.content_length == 12345
    assert file_metadata.mime_type == "application/pdf"


@pytest.mark.asyncio
async def test_load_document_metadata_external(request_context):
    attachment_link = AttachmentLink.from_link(
        request_context,
        "https://example.com/my%20folder/my%20file%20(1).pdf",
    )

    with aioresponses() as mock:
        mock.head(
            "https://example.com/my%20folder/my%20file%20(1).pdf",
            headers={
                "Content-Length": "54321",
                "Content-Type": "application/pdf",
                "ETag": '"def456"',
                "Last-Modified": "Wed, 21 Oct 2015 07:28:00 GMT",
            },
        )

        file_metadata = await load_document_metadata(
            request_context=request_context,
            attachment_link=attachment_link,
            config=HttpClientConfig(),
        )

    assert file_metadata is not None
    assert file_metadata.content_length == 54321
    assert file_metadata.mime_type == "application/pdf"
    assert file_metadata.etag == '"def456"'
    assert (
        file_metadata.last_modified == 1445412480
    )  # Unix timestamp for the given date


@pytest.mark.parametrize(
    "content_type,expected_mime_type",
    [
        (None, "application/octet-stream"),
        ("application/pdf", "application/pdf"),
        ("text/html; charset=UTF-8", "text/html"),
    ],
)
def test_get_mime_type(content_type, expected_mime_type):
    assert get_mime_type(content_type) == expected_mime_type
