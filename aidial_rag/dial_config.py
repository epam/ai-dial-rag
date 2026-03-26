from pydantic import Field, SecretStr

from aidial_rag.base_config import BaseConfig


class DialConfig(BaseConfig):
    dial_url: str
    api_key: SecretStr

    extra_headers: dict = Field(
        default_factory=dict,
        description=(
            "Extra headers to include in the requests to the Dial Core API."
        ),
    )
