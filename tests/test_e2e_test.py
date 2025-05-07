import functools

import aiohttp
import pytest
import requests

from tests.utils.e2e_decorator import e2e_test

EXPECTED_ATTACHMENTS = [
    {
        "type": "text/csv",
        "title": "test_file.csv",
        "url": "files/6iTkeGUs2CvUehhYLmMYXB/test_file.csv",
    }
]

FILE_URL = "http://localhost:8081/v1/files/6iTkeGUs2CvUehhYLmMYXB/test_file.csv"


async def check_async_request(expected_status_code=200):
    """Async request to the server"""
    async with aiohttp.ClientSession() as session:
        async with session.get(FILE_URL) as response:
            assert response.status == expected_status_code
            return await response.text()


def check_server_lifetime(test_func):
    @functools.wraps(test_func)
    async def wrapper(*args, **kwargs):
        # check that the server is not started yet
        with pytest.raises(aiohttp.ClientConnectorError):
            await check_async_request()

        result = await test_func(*args, **kwargs)

        # check that server is stopped after the test
        with pytest.raises(aiohttp.ClientConnectorError):
            await check_async_request()
        return result

    return wrapper


@pytest.mark.asyncio
@check_server_lifetime
@e2e_test(filenames=["test_file.csv"])
async def test_e2e_test_async_requests(attachments):
    """Simple test to check if the e2e_test decorator works correctly:
    server is started, responds the request and stops properly.
    """

    assert attachments == EXPECTED_ATTACHMENTS
    response_text = await check_async_request(expected_status_code=200)
    assert len(response_text) == 77


@pytest.mark.asyncio
@check_server_lifetime
@e2e_test(filenames=["test_file.csv"])
async def test_e2e_test_blocking_requests(attachments):
    assert attachments == EXPECTED_ATTACHMENTS

    result = requests.get(FILE_URL, timeout=5)
    assert result.status_code == 200
    assert len(result.text) == 77
