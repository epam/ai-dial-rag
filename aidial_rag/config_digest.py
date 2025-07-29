from typing import Any, Dict

from pydantic import BaseModel

from aidial_rag.configuration_endpoint import Configuration


class ConfigDigest(BaseModel):
    # We do not want to expose the entire app_config
    app_config_path: str

    configuration: Configuration
    from_custom_configuration: Dict[str, Any] = {}
    from_commands: Dict[str, Any] = {}
