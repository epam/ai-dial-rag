import hashlib
import logging
from pathlib import Path

from tests.utils.cache_response import CacheResponse

logger = logging.getLogger(__name__)


def get_cache_key(input_str):
    """Generate a hash key for the input string."""
    return hashlib.md5(input_str.encode()).hexdigest()  # noqa: S324


class LlmCache:
    def __init__(self, cache_dir="cache", enable_cache=True):
        self.cache_dir = Path(cache_dir)
        self.enable_cache = enable_cache

        if self.enable_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_cache_file_path(self, prefix, key, file_type="response"):
        """Get the file path for the cached data."""
        filename = (
            f"{prefix}/{key}.{file_type}" if prefix else f"{key}.{file_type}"
        )
        return self.cache_dir / filename

    def get(self, input_str, prefix=""):
        """Retrieve cached data if available."""
        if not self.enable_cache:
            return None, None

        cache_key = get_cache_key(input_str)

        file_path = self.get_cache_file_path(prefix, cache_key)
        if file_path.exists():
            try:
                with file_path.open("r") as f:
                    return CacheResponse.deserialize(f.read()), file_path
            except Exception as e:
                logger.error(e)
                return None, None
        logger.info(
            f"The cache was not found for inputstr:{input_str[:1000]}, hash: {cache_key}"
        )
        return None, None

    def set(self, input_str, data, prefix=""):
        if not self.enable_cache:
            return

        cache_key = get_cache_key(input_str)
        file_path = self.get_cache_file_path(prefix, cache_key)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(data, CacheResponse):
            with file_path.open("w") as file:
                file.write(data.serialize())

        logger.info("Response cached in file " + str(file_path))

    def cleanup(self, file_list):
        for file_path in file_list:
            try:
                file_path.unlink()  # Delete the file
                logger.info(f"Deleted: {file_path}")
            except Exception as e:
                logger.error(f"Error deleting {file_path}: {e}")

    def get_by_prefix(self, test_folder):
        path = self.cache_dir / test_folder
        result = []

        for file_path in path.glob("*.response"):
            result.append(file_path)

        return result
