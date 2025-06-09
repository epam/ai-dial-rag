import pytest
from aidial_sdk import HTTPException

from aidial_rag.resources.cpu_pools import run_in_indexing_cpu_pool


class TestException(Exception):
    message: str

    def __init__(self, message):
        self.message = message


def function_with_picklable_exception():
    raise TestException("This exception is picklable")


def function_with_unpicklable_exception():
    raise TestException(message="This exception is not picklable")


def function_with_http_exception():
    raise HTTPException(message="This exception is picklable after fix")


@pytest.mark.asyncio
async def test_picklable_exception():
    with pytest.raises(TestException, match="This exception is picklable"):
        await run_in_indexing_cpu_pool(function_with_picklable_exception)


@pytest.mark.asyncio
async def test_unpicklable_exception():
    with pytest.raises(TestException) as exc_info:
        await run_in_indexing_cpu_pool(function_with_unpicklable_exception)
    assert exc_info.value.message == "This exception is not picklable"


@pytest.mark.asyncio
async def test_http_exception():
    with pytest.raises(
        HTTPException, match="This exception is picklable after fix"
    ):
        await run_in_indexing_cpu_pool(function_with_http_exception)
