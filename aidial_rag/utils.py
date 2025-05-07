import asyncio
import os
from contextlib import asynccontextmanager, contextmanager
from pathlib import PurePosixPath
from time import perf_counter
from typing import Collection
from urllib.parse import urlparse

import humanfriendly
from aidial_sdk.chat_completion.choice import Choice

from aidial_rag.content_stream import SupportsWriteStr
from aidial_rag.errors import HTTPException


async def periodic_ping(file: SupportsWriteStr, interval: int = 15):
    while True:
        try:
            await asyncio.sleep(interval)
        except asyncio.CancelledError:
            break
        file.write("")


@asynccontextmanager
async def timed_block(name: str, file: SupportsWriteStr):
    file.write(f"{name} started\n")
    start = perf_counter()
    try:
        yield
    finally:
        end = perf_counter()
        file.write(f"{name} took {end - start:.2f}s\n")


@contextmanager
def timed_stage(choice: Choice, *args, **kwargs):
    with choice.create_stage(*args, **kwargs) as stage:
        stageio = stage.content_stream
        ping_task = asyncio.create_task(periodic_ping(stageio))
        start = perf_counter()
        try:
            yield stage
        finally:
            end = perf_counter()
            ping_task.cancel()
            stage.append_name(f" [{end - start:.2f}s]")


@contextmanager
def profiler_if_enabled(choice: Choice, enabled: bool):
    if enabled:
        from pyinstrument import Profiler

        with choice.create_stage("Profiler") as stage:
            with Profiler() as pr:
                yield

            # We put html in a content instead of attachment because the Chat UI
            # does not display html in attachments, but allows to download it from content
            stage.append_content(f"```\n{pr.output_html()}\n```")
            stage.add_attachment(
                type="text/plain",
                title="Profiler.txt",
                data=pr.output_text(),
            )
    else:
        yield


def bool_env_var(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).lower() == "true"


def format_size(size: int) -> str:
    return humanfriendly.format_size(size, binary=True)


def get_bytes_length(s: str) -> int:
    return len(s.encode("utf-8"))


def extract_filename_from_url(url):
    parsed_url = urlparse(url)
    path = parsed_url.path
    return PurePosixPath(path).name


def check_mime_type(mime_type: str, supported_list: Collection[str]) -> bool:
    """
    Check if a given MIME type is supported based on a list of supported MIME types.

    This function checks for exact matches and also supports wildcard matching.
    A MIME type of the form 'type/*' will match any MIME type that starts with 'type/'.

    Args:
        mime_type (str): The MIME type to check.
        supported_list (Collection[str]): A collection of supported MIME types, which may include wildcards.

    Returns:
        bool: True if the MIME type is supported, False otherwise.
    """
    mime_to_check = [mime_type, mime_type.split("/")[0] + "/*"]
    return any(mime in supported_list for mime in mime_to_check)


@asynccontextmanager
async def timeout(seconds: float, error_message=None):
    try:
        async with asyncio.timeout(seconds):
            yield
    except asyncio.TimeoutError as e:
        message = (
            error_message or f"Failed to process request in {seconds} seconds"
        )
        raise HTTPException(message) from e
