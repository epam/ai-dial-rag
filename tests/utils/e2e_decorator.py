import asyncio
import logging
import mimetypes
import os
from pathlib import Path
from typing import Tuple

import aiohttp
import pytest
import uvicorn

from aidial_rag.utils import bool_env_var
from tests.utils.cache_middleware import (
    FAILURE_MESSAGE,
    WARNING_MESSAGE,
    CacheMiddlewareApp,
    CacheMiddlewareConfig,
)

logger = logging.getLogger(__name__)
logger.setLevel("DEBUG")


# TODO: Need to refactor e2e_test to make content type the same on all platforms
# e2e_test uses mimetypes.guess_type to determine the type of the file
# which would detect the csv file as application/vnd.ms-excel on Windows and text/csv on Linux
# which would lead to different parsing process in unstructured.
# Explicitly setting the content type to text/csv to make it consistent on all platforms
mimetypes.add_type("text/csv", ".csv")


async def start_server(
    refresh, module_name, test_name
) -> Tuple[asyncio.Future, uvicorn.Server, CacheMiddlewareApp]:
    logger.info("Starting server...")
    config_data = {
        "dial_core_host": os.getenv("DIAL_CORE_HOST", "localhost:8080"),
        "dial_core_api_key": os.getenv("DIAL_CORE_API_KEY", "dial_api_key"),
        "base_path": os.getenv("BASE_PATH", "tests/"),
        "test_name": test_name,
        "module_name": module_name,
        "refresh": refresh,
    }
    cache_middleware_app = CacheMiddlewareApp(
        app_config=CacheMiddlewareConfig(**config_data)
    )

    config = uvicorn.Config(
        cache_middleware_app, host="127.0.0.1", port=8081, log_level="debug"
    )
    server = uvicorn.Server(config)

    # Start the server in a separate task
    loop = asyncio.get_event_loop()
    server_future: asyncio.Future = loop.run_in_executor(None, server.run)

    return (
        server_future,
        server,
        cache_middleware_app,
    )  # Return the future and server for later use


async def wait_for_server_ready(
    port: int, retries=5, sleep=0.1, timeout=1
) -> None:
    url = f"http://localhost:{port}/health"
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=timeout)
    ) as session:
        for _ in range(retries):
            try:
                async with session.get(url) as response:
                    response.raise_for_status()  # Fail test if servers ready but not 200
                    return
            except aiohttp.ClientConnectorError:
                logger.warning(f"Server at {url} is not ready yet, retrying...")
                await asyncio.sleep(sleep)
    raise RuntimeError(f"Server at {url} is not ready after {retries} retries.")


async def stop_server(
    server: uvicorn.Server, server_future: asyncio.Future
) -> None:
    # Shutdown the server after tests are done
    server.should_exit = True  # Signal the server to shut down
    logger.info("Shutting down server...")
    # Wait for the server to finish shutting down
    # Timeout will fail the test instead of hanging indefinitely
    await asyncio.wait_for(server_future, timeout=30)
    logger.info("Server shut down successfully.")


def fail_on_warnings(recwarn, exception):
    specific_warnings = [
        w for w in recwarn.list if WARNING_MESSAGE in str(w.message)
    ]
    if specific_warnings and exception is not None:
        exception.add_note(FAILURE_MESSAGE)
        raise exception
    elif specific_warnings:
        pytest.fail(FAILURE_MESSAGE)
    elif exception is not None:
        raise exception


def e2e_test(filenames=(), refresh=None):
    """
    Decorator to start the whole RAG application, supports emulate calls to LLM using cached LLM responses
    :param filename: optional string of the file in ./tests/data/ in case if test require input attachment. In this case, the test would be called with AttachmentLink param.
    :param refresh: optional boolean flag to determine, if LLM cache should be refreshed.
    :return: test result
    """
    if refresh is None:
        refresh = bool_env_var("REFRESH", default=False)

    def decorator(orig_func):
        func = pytest.mark.filterwarnings(f"always:{WARNING_MESSAGE}")(
            orig_func
        )

        async def wrapper(request, recwarn, caplog, *args, **kwargs):
            if request is not None:
                os.environ["TEST_NAME"] = request.node.name
            attachments = [
                {
                    "type": mimetypes.guess_type(filename)[0],
                    "title": filename,
                    "url": f"files/6iTkeGUs2CvUehhYLmMYXB/{filename}",
                }
                for filename in filenames
            ]

            exception = None
            server_future, server, cache = await start_server(
                module_name=Path(request.node.parent.name).with_suffix("").name,
                test_name=request.node.name,
                refresh=refresh,
            )
            try:
                await wait_for_server_ready(port=server.config.port)
                logger.info("Server started successfully.")
                # If the function has caplog fixture as parameter, we pass it to the function
                # Pytest gets fixtures from the function signature, not from the function call
                # TODO: Rewrite e2e_test from decorator to context manager to avoid interference with the fixture system
                if "caplog" in func.__code__.co_varnames:
                    args = (caplog,) + args
                if not attachments:
                    result = await func(*args, **kwargs)
                else:
                    result = await func(attachments, *args, **kwargs)
                cache.cleanup()
                return result
            except AssertionError as e:
                exception = e
            finally:
                await stop_server(server, server_future)
                fail_on_warnings(recwarn, exception)

        return wrapper

    return decorator
