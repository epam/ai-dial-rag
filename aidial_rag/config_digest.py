from typing import Any, Dict

from pydantic import BaseModel

from aidial_rag.app_config import RequestConfig


class ConfigDigest(BaseModel):
    # We do not want to expose the entire app_config
    app_config_path: str

    request_config: RequestConfig
    from_custom_configuration: Dict[str, Any] = {}
    from_commands: Dict[str, Any] = {}
