import io

import aiohttp

from aidial_rag.request_context import RequestContext


async def _get_bucket_id(dial_base_url, headers: dict) -> str:
    relative_url = (
        "bucket"  # /v1/ is already included in the base url for the Dial API
    )
    async with aiohttp.ClientSession(base_url=dial_base_url) as session:
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
    def __init__(self, dial_api_base_url: str, headers: dict, bucket_id: str):
        self.bucket_id = bucket_id

        self._dial_api_base_url = dial_api_base_url
        self._headers = headers

    async def get_file(self, relative_url: str) -> bytes | None:
        async with aiohttp.ClientSession(
            base_url=self._dial_api_base_url
        ) as session:
            async with session.get(
                relative_url, headers=self._headers
            ) as response:
                response.raise_for_status()
                return await response.read()

    async def put_file(
        self, relative_url: str, data: bytes, content_type: str
    ) -> dict:
        async with aiohttp.ClientSession(
            base_url=self._dial_api_base_url
        ) as session:
            form_data = _to_form_data(relative_url, data, content_type)
            async with session.put(
                relative_url, data=form_data, headers=self._headers
            ) as response:
                response.raise_for_status()
                return await response.json()


async def create_dial_api_client(
    request_context: RequestContext,
) -> DialApiClient:
    headers = request_context.get_api_key_headers()
    bucket_id = await _get_bucket_id(request_context.dial_base_url, headers)
    return DialApiClient(request_context.dial_base_url, headers, bucket_id)
