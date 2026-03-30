from contextlib import contextmanager
from typing import List

from aidial_sdk.chat_completion import Choice, Request, Response
from pydantic import BaseModel, SecretStr

from aidial_rag.dial_config import DialConfig
from aidial_rag.dial_user_limits import get_user_limits_for_model
from aidial_rag.errors import convert_and_log_exceptions
from aidial_rag.resources.dial_limited_resources import DialLimitedResources


class RequestContext(BaseModel):
    dial_config: DialConfig
    choice: Choice
    dial_limited_resources: DialLimitedResources

    class Config:
        # aidial_sdk.chat_completion.Choice is not a pydantic model
        arbitrary_types_allowed = True

    def is_dial_url(self, url: str) -> bool:
        return url.startswith(self.dial_config.dial_url)

    @property
    def dial_base_url(self) -> str:
        return f"{self.dial_config.dial_url}/v1/"

    @property
    def dial_metadata_base_url(self) -> str:
        return f"{self.dial_base_url}/metadata/"

    def get_file_access_headers(self, url: str) -> dict:
        if not self.is_dial_url(url):
            return {}

        return self.get_api_key_headers()

    def get_api_key_headers(self) -> dict:
        return {"api-key": self.dial_config.api_key.get_secret_value()}


def collect_headers_to_proxy(
    request: Request, headers_to_proxy: list[str]
) -> dict:
    return {
        header: request.headers[header]
        for header in headers_to_proxy
        if header in request.headers
    }


@contextmanager
def create_request_context(
    dial_url: str,
    headers_to_proxy: List[str],
    request: Request,
    response: Response,
):
    extra_headers = collect_headers_to_proxy(request, headers_to_proxy)

    with convert_and_log_exceptions():
        with response.create_single_choice() as choice:
            dial_config = DialConfig(
                dial_url=dial_url,
                api_key=SecretStr(request.api_key),
                extra_headers=extra_headers,
            )

            request_context = RequestContext(
                dial_config=dial_config,
                choice=choice,
                dial_limited_resources=DialLimitedResources(
                    lambda model_name: get_user_limits_for_model(
                        dial_config, model_name
                    )
                ),
            )
            yield request_context
