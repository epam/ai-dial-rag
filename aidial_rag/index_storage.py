import hashlib
import io
import logging
from abc import ABC, abstractmethod
from typing import cast

import aiohttp
from cachetools import LRUCache
from pydantic import ByteSize, Field

from aidial_rag.attachment_link import AttachmentLink
from aidial_rag.base_config import BaseConfig
from aidial_rag.dial_api_client import DialApiClient
from aidial_rag.document_record import (
    FORMAT_VERSION,
    DocumentRecord,
    IndexSettings,
)
from aidial_rag.indexing_task import IndexingTask

logger = logging.getLogger(__name__)


class IndexStorageConfig(BaseConfig):
    use_dial_file_storage: bool = Field(
        default=False,
        description="Set to `True` to store indexes in the Dial File Storage instead of in memory storage",
        validation_alias="use_dial_file_storage",
    )
    in_memory_cache_capacity: ByteSize = Field(
        default="128MiB",
        validate_default=True,  # validation will convert string to int, but typecheck does not understand it
        description=(
            "Used to cache the document indexes and avoid requesting Dial Core File API every time, "
            "if user makes several requests for the same document. Could be increased to reduce load "
            "on the Dial Core File API if we have a lot of concurrent users "
            "(requires corresponding increase of the pod memory). "
            "Could be integer for bytes, or a pydantic.ByteSize compatible string (e.g. 128MiB, 1GiB, 2.5GiB)."
        ),
    )  # type: ignore


DEFAULT_IN_MEMORY_CACHE_CAPACITY = IndexStorageConfig().in_memory_cache_capacity


SERIALIZATION_CONFIG = {"protocol": "pickle", "compress": "gzip"}

INDEX_MIME_TYPE = "application/x.aidial-rag-index.v0"


# Number of characters in each directory part for index file paths
# This is treated as a part of an algorithm, not a configuration parameter,
# because if changed, the old index files will not be found.
INDEX_PATH_PART_SIZE = 8


def link_to_index_url(attachment_link: AttachmentLink, bucket_id: str) -> str:
    key = hashlib.sha256(attachment_link.dial_link.encode()).hexdigest()

    # split the key into parts to avoid too many files in one directory
    dir_path = "/".join(
        key[i : i + INDEX_PATH_PART_SIZE]
        for i in range(0, len(key), INDEX_PATH_PART_SIZE)
    )

    return f"files/{bucket_id}/dial-rag-index/{dir_path}/index.bin"


class IndexStorageBackend(ABC):
    @abstractmethod
    async def load(self, url: str, *args, **kwargs) -> bytes | None:
        pass

    @abstractmethod
    async def store(self, url: str, data: bytes, *args, **kwargs) -> dict:
        pass


class LRUCacheStorage(IndexStorageBackend):
    def __init__(self, capacity: int = DEFAULT_IN_MEMORY_CACHE_CAPACITY):
        self._cache = LRUCache(maxsize=capacity, getsizeof=len)

    async def load(self, url: str, *args, **kwargs) -> bytes | None:
        return cast(bytes | None, self._cache.get(url))

    async def store(self, url, data: bytes, *args, **kwargs) -> dict:
        self._cache[url] = data
        return {}


class DialFileStorage(IndexStorageBackend):
    """Dial File API storage for index files
    Implements API from https://gitlab.deltixhub.com/Deltix/openai-apps/documentation/-/issues/12
    """

    def __init__(self, dial_url: str):
        self._dial_url = dial_url
        self._dial_base_url = f"{dial_url}/v1/"

    def to_form_data(self, key: str, data: bytes) -> aiohttp.FormData:
        form_data = aiohttp.FormData()
        form_data.add_field(
            "file", io.BytesIO(data), filename=key, content_type=INDEX_MIME_TYPE
        )
        return form_data

    async def load(
        self, url: str, dial_api_client: DialApiClient
    ) -> bytes | None:
        try:
            return await dial_api_client.get_file(url)
        except aiohttp.ClientError as e:
            logger.warning(f"Failed to load index from {url}: {e}")
            return None

    async def store(
        self, url, data: bytes, dial_api_client: DialApiClient
    ) -> dict:
        return await dial_api_client.put_file(url, data, INDEX_MIME_TYPE)


class CachedStorage(IndexStorageBackend):
    def __init__(
        self,
        storage: IndexStorageBackend,
        capacity: int = DEFAULT_IN_MEMORY_CACHE_CAPACITY,
    ):
        self._storage = storage
        self._cache = LRUCacheStorage(capacity)

    async def load(self, url: str, *args, **kwargs) -> bytes | None:
        data = await self._cache.load(url, *args, **kwargs)
        if data is not None:
            return data

        data = await self._storage.load(url, *args, **kwargs)
        if data is not None:
            await self._cache.store(url, data, *args, **kwargs)
        return data

    async def store(self, url, data: bytes, *args, **kwargs) -> dict:
        await self._cache.store(url, data, *args, **kwargs)
        return await self._storage.store(url, data, *args, **kwargs)


class IndexStorage:
    def __init__(
        self,
        dial_url,
        index_storage_config: IndexStorageConfig | None = None,
    ):
        if index_storage_config is None:
            index_storage_config = IndexStorageConfig()
        if index_storage_config.use_dial_file_storage:
            self._storage = CachedStorage(
                DialFileStorage(dial_url),
                index_storage_config.in_memory_cache_capacity,
            )
        else:
            self._storage = LRUCacheStorage(
                index_storage_config.in_memory_cache_capacity
            )

    async def load(
        self,
        task: IndexingTask,
        index_settings: IndexSettings,
        dial_api_client: DialApiClient,
    ) -> DocumentRecord | None:
        doc_record_bytes = await self._storage.load(
            task.index_url, dial_api_client=dial_api_client
        )
        if doc_record_bytes is None:
            return None
        try:
            doc_record = DocumentRecord.from_bytes(
                doc_record_bytes, **SERIALIZATION_CONFIG
            )
            if doc_record.format_version != FORMAT_VERSION:
                logger.warning(
                    f"Index format version mismatch for {task.attachment_link}: {doc_record.format_version}"
                )
                return None
            if doc_record.index_settings != index_settings:
                logger.warning(
                    f"Index settings mismatch for {task.attachment_link}: {doc_record.index_settings}"
                )
                return None
            return doc_record
        except Exception as e:
            logger.warning(
                f"Failed to deserialize index for {task.attachment_link}: {e}"
            )
            return None

    async def store(
        self,
        task: IndexingTask,
        doc_record: DocumentRecord,
        dial_api_client: DialApiClient,
    ) -> dict:
        doc_record_bytes = doc_record.to_bytes(**SERIALIZATION_CONFIG)
        logger.debug(
            f"Stored document {task.attachment_link} index with url: {task.index_url}"
        )
        return await self._storage.store(
            task.index_url, doc_record_bytes, dial_api_client=dial_api_client
        )
