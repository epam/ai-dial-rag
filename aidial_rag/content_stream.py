import logging
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)


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
        if content == "\n" or content == "":
            # avoid prefixing empty lines, like on tqdm.close() calls
            return
        message = f"{self.prefix} {content}"
        logger.info(f"{message}")
        self.stream.write(f"{message}\n\n")
