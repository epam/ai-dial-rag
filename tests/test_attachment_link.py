from unittest.mock import MagicMock

import aiohttp
import pytest
from aidial_sdk.chat_completion import Choice
from pydantic import SecretStr

from aidial_rag.attachment_link import AttachmentLink
from aidial_rag.dial_api_client import DialApiClient
from aidial_rag.errors import InvalidAttachmentError
from aidial_rag.request_context import RequestContext
from aidial_rag.resources.dial_limited_resources import DialLimitedResources
from tests.utils.user_limits_mock import user_limits_mock


@pytest.fixture
async def request_context_coro():
    return RequestContext(
        dial_url="http://core.dial",
        api_key=SecretStr(""),
        choice=Choice(queue=MagicMock(), choice_index=0),
        dial_api_client=DialApiClient(
            client_session=aiohttp.ClientSession(), bucket_id=""
        ),
        dial_limited_resources=DialLimitedResources(user_limits_mock()),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "link, expected_absolute_url, expected_display_name",
    [
        (
            "http://example.com/absolute_link/file.txt",
            "http://example.com/absolute_link/file.txt",
            "http://example.com/absolute_link/file.txt",
        ),
        (
            "http://example.com/absolute%20link/file%231.txt",
            "http://example.com/absolute%20link/file%231.txt",
            "http://example.com/absolute%20link/file%231.txt",
        ),
        (
            "files/bucket/file.txt",
            "http://core.dial/v1/files/bucket/file.txt",
            "file.txt",
        ),
        (
            "files/bucket/folder%231/folder%232/file%233.txt",
            "http://core.dial/v1/files/bucket/folder%231/folder%232/file%233.txt",
            "folder#1/folder#2/file#3.txt",
        ),
        (
            "files/bucket/fo%24l%2B%3Dd%2Ce%23r%201/file.txt",
            "http://core.dial/v1/files/bucket/fo%24l%2B%3Dd%2Ce%23r%201/file.txt",
            "fo$l+=d,e#r 1/file.txt",
        ),
        (
            "files/bucket/my%20folder/my%20file%20(1).pdf",
            "http://core.dial/v1/files/bucket/my%20folder/my%20file%20(1).pdf",
            "my folder/my file (1).pdf",
        ),
        # Russian text
        (
            "files/bucket/%D0%BC%D0%BE%D1%8F%20%D0%BF%D0%B0%D0%BF%D0%BA%D0%B0/%D0%BC%D0%BE%D0%B9%20%D1%84%D0%B0%D0%B9%D0%BB%20%281%29.pdf",
            "http://core.dial/v1/files/bucket/%D0%BC%D0%BE%D1%8F%20%D0%BF%D0%B0%D0%BF%D0%BA%D0%B0/%D0%BC%D0%BE%D0%B9%20%D1%84%D0%B0%D0%B9%D0%BB%20%281%29.pdf",
            "моя папка/мой файл (1).pdf",
        ),
        # Turkish text
        (
            "files/bucket/klas%C3%B6r%C3%BCm/dosyam%20%281%29.pdf",
            "http://core.dial/v1/files/bucket/klas%C3%B6r%C3%BCm/dosyam%20%281%29.pdf",
            "klasörüm/dosyam (1).pdf",
        ),
        # Chinese text
        (
            "files/bucket/%E6%88%91%E7%9A%84%E6%96%87%E4%BB%B6%E5%A4%B9/%E6%88%91%E7%9A%84%E6%96%87%E4%BB%B6%20%281%29.pdf",
            "http://core.dial/v1/files/bucket/%E6%88%91%E7%9A%84%E6%96%87%E4%BB%B6%E5%A4%B9/%E6%88%91%E7%9A%84%E6%96%87%E4%BB%B6%20%281%29.pdf",
            "我的文件夹/我的文件 (1).pdf",
        ),
        # Japanese text
        (
            "files/bucket/%E7%A7%81%E3%81%AE%E3%83%95%E3%82%A9%E3%83%AB%E3%83%80/%E7%A7%81%E3%81%AE%E3%83%95%E3%82%A1%E3%82%A4%E3%83%AB%20%281%29.pdf",
            "http://core.dial/v1/files/bucket/%E7%A7%81%E3%81%AE%E3%83%95%E3%82%A9%E3%83%AB%E3%83%80/%E7%A7%81%E3%81%AE%E3%83%95%E3%82%A1%E3%82%A4%E3%83%AB%20%281%29.pdf",
            "私のフォルダ/私のファイル (1).pdf",
        ),
        # Korean text
        (
            "files/bucket/%EB%82%B4%20%ED%8F%B4%EB%8D%94/%EB%82%B4%20%ED%8C%8C%EC%9D%BC%281%29.pdf",
            "http://core.dial/v1/files/bucket/%EB%82%B4%20%ED%8F%B4%EB%8D%94/%EB%82%B4%20%ED%8C%8C%EC%9D%BC%281%29.pdf",
            "내 폴더/내 파일(1).pdf",
        ),
        # Swedish text with two ways to represent the same character in Unicode
        (
            "files/bucket/%C3%85str%C3%B6m/A%CC%8Astro%CC%88m.txt",
            "http://core.dial/v1/files/bucket/%C3%85str%C3%B6m/A%CC%8Astro%CC%88m.txt",
            "Åström/Åström.txt",
        ),
    ],
)
async def test_attachment_link_from_link(
    request_context_coro, link, expected_absolute_url, expected_display_name
):
    request_context = await request_context_coro
    attachment_link = AttachmentLink.from_link(request_context, link)
    assert attachment_link.dial_link == link
    assert attachment_link.absolute_url == expected_absolute_url
    assert attachment_link.display_name == expected_display_name


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "link",
    [
        "/bucket/file.txt",
        "file.txt",
    ],
)
async def test_attachment_link_errors(request_context_coro, link):
    request_context = await request_context_coro
    with pytest.raises(InvalidAttachmentError):
        AttachmentLink.from_link(request_context, link)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "link, expected_dial_link, expected_absolute_url, expected_metadata_url",
    [
        (
            "http://example.com/absolute_link/file.txt",
            "http://example.com/absolute_link/file.txt",
            "http://example.com/absolute_link/file.txt",
            None,
        ),
        (
            "files/bucket/file.txt",
            "files/bucket/file.txt",
            "http://core.dial/v1/files/bucket/file.txt",
            "metadata/files/bucket/file.txt",
        ),
        (
            "http://core.dial/v1/files/bucket/file.txt",
            "files/bucket/file.txt",
            "http://core.dial/v1/files/bucket/file.txt",
            "metadata/files/bucket/file.txt",
        ),
    ],
)
async def test_metadata_url(
    request_context_coro,
    link,
    expected_dial_link,
    expected_absolute_url,
    expected_metadata_url,
):
    request_context = await request_context_coro
    attachment_link = AttachmentLink.from_link(request_context, link)
    assert attachment_link.dial_link == expected_dial_link
    assert attachment_link.absolute_url == expected_absolute_url
    assert attachment_link.dial_metadata_url == expected_metadata_url
