import logging
from typing import TYPE_CHECKING

_logger = logging.getLogger(__name__)


# _typeshed module is not available at runtime, so we need to use a string literal
# But we want to avoid `if TYPE_CHECKING` in every file that uses this type
if TYPE_CHECKING:
    from _typeshed import SupportsWrite

    SupportsWriteStr = SupportsWrite[str]
else:
    SupportsWriteStr = "SupportsWrite[str]"


class StreamWithPrefix:
    def __init__(self, stream: SupportsWriteStr, prefix: str):
        self.stream = stream
        self.prefix = prefix

    def write(self, content):
        if not content.strip(" \n"):
            # Avoid prefixing empty lines, like on tqdm.close() calls
            return
        self.stream.write(f"{self.prefix} {content}")


class MarkdownStream:
    def __init__(self, stream: SupportsWriteStr):
        self.stream = stream

    def write(self, content):
        # Use double new line for markdown formatting
        self.stream.write(f"{content}\n\n")


class LoggerStream:
    def __init__(
        self, logger: logging.Logger = _logger, log_level: int = logging.INFO
    ):
        self.logger = logger
        self.log_level = log_level

    def write(self, content):
        if message := content.strip(" \n"):
            self.logger.log(self.log_level, f"{message}")


class MultiStream:
    def __init__(self, *streams: SupportsWriteStr):
        self.streams = streams

    def write(self, content):
        for stream in self.streams:
            stream.write(content)
