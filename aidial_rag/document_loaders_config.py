import aiohttp
from pydantic import Field

from aidial_rag.base_config import BaseConfig


class HttpClientConfig(BaseConfig):
    timeout_seconds: int = Field(
        default=30,
        description="Timeout for the whole request. Includes connection establishment, sending the request, and receiving the response.",
    )
    connect_timeout_seconds: int = Field(
        default=30,
        description="Timeout for establishing a connection to the server.",
    )

    def get_client_timeout(self) -> aiohttp.ClientTimeout:
        return aiohttp.ClientTimeout(
            total=self.timeout_seconds,
            connect=self.connect_timeout_seconds,
            sock_connect=self.connect_timeout_seconds,
        )
