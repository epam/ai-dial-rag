from pydantic import SecretStr

from aidial_rag.base_config import BaseConfig


class DialConfig(BaseConfig):
    dial_url: str
    api_key: SecretStr

    @property
    def dial_base_url(self) -> str:
        return f"{self.dial_url}/v1/"
