from contextlib import contextmanager

from aidial_sdk.chat_completion import Choice, Request, Response
from pydantic import BaseModel, SecretStr

from aidial_rag.dial_config import DialConfig
from aidial_rag.dial_user_limits import get_user_limits_for_model
from aidial_rag.errors import convert_and_log_exceptions
from aidial_rag.resources.dial_limited_resources import DialLimitedResources


class RequestContext(BaseModel):
    dial_url: str
    api_key: SecretStr
    choice: Choice
    dial_limited_resources: DialLimitedResources

    class Config:
        # aidial_sdk.chat_completion.Choice is not a pydantic model
        arbitrary_types_allowed = True

    def is_dial_url(self, url: str) -> bool:
        return url.startswith(self.dial_url)

    @property
    def dial_base_url(self) -> str:
        return f"{self.dial_url}/v1/"

    @property
    def dial_metadata_base_url(self) -> str:
        return f"{self.dial_base_url}/metadata/"

    @property
    def dial_config(self) -> DialConfig:
        return DialConfig(dial_url=self.dial_url, api_key=self.api_key)

    def get_file_access_headers(self, url: str) -> dict:
        if not self.is_dial_url(url):
            return {}

        return self.get_api_key_headers()

    def get_api_key_headers(self) -> dict:
        return {"api-key": self.api_key.get_secret_value()}


@contextmanager
def create_request_context(dial_url: str, request: Request, response: Response):
    with convert_and_log_exceptions():
        with response.create_single_choice() as choice:
            dial_config = DialConfig(
                dial_url=dial_url, api_key=SecretStr(request.api_key)
            )

            request_context = RequestContext(
                dial_url=dial_url,
                api_key=dial_config.api_key,
                choice=choice,
                dial_limited_resources=DialLimitedResources(
                    lambda model_name: get_user_limits_for_model(
                        dial_config, model_name
                    )
                ),
            )
            yield request_context
