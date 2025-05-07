from pydantic import SecretStr

from aidial_rag.base_config import BaseConfig


class DialConfig(BaseConfig):
    dial_url: str
    api_key: SecretStr
