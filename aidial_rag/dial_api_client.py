import io
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import aiohttp

from aidial_rag.dial_config import DialConfig


async def _get_bucket_id(session: aiohttp.ClientSession, headers: dict) -> str:
    relative_url = (
        "bucket"  # /v1/ is already included in the base url for the Dial API
    )
    async with session.get(relative_url, headers=headers) as response:
        response.raise_for_status()
        data = await response.json()
        return data["bucket"]


def _to_form_data(key: str, data: bytes, content_type: str) -> aiohttp.FormData:
    form_data = aiohttp.FormData()
    form_data.add_field(
        "file", io.BytesIO(data), filename=key, content_type=content_type
    )
    return form_data


class DialApiClient:
    def __init__(self, client_session: aiohttp.ClientSession, bucket_id: str):
        self._client_session = client_session
        self.bucket_id = bucket_id

    @property
    def session(self) -> aiohttp.ClientSession:
        return self._client_session

    async def get_file(self, relative_url: str) -> bytes | None:
        async with self.session.get(relative_url) as response:
            response.raise_for_status()
            return await response.read()

    async def put_file(
        self, relative_url: str, data: bytes, content_type: str
    ) -> dict:
        form_data = _to_form_data(relative_url, data, content_type)
        async with self.session.put(relative_url, data=form_data) as response:
            response.raise_for_status()
            return await response.json()


@asynccontextmanager
async def create_dial_api_client(
    config: DialConfig,
) -> AsyncGenerator[DialApiClient, None]:
    headers = {"api-key": config.api_key.get_secret_value()}
    async with aiohttp.ClientSession(
        base_url=config.dial_base_url, headers=headers
    ) as session:
        bucket_id = await _get_bucket_id(session, headers)
        yield DialApiClient(session, bucket_id)
