from typing import Any, Dict, TypeVar

from deepmerge.merger import Merger
from pydantic import BaseModel, ConfigDict


class IndexRebuildTrigger:
    """Marker class for config fields that require index rebuilding if changed."""

    pass


def collect_fields_with_trigger(
    config: BaseModel, trigger_cls=IndexRebuildTrigger
) -> Dict[str, Any]:
    """Return the config fields that determine when the index needs to be rebuilt."""
    rebuild_trigger_fields = {}
    for name, field_info in config.__class__.model_fields.items():
        if any(isinstance(meta, trigger_cls) for meta in field_info.metadata):
            rebuild_trigger_fields[name] = getattr(config, name)
    return rebuild_trigger_fields


class BaseConfig(BaseModel):
    """Base configuration class for the application.
    This class is needed because BaseSettings will parse the config file as if it is a root config,
    not as a subconfig, and BaseModel does not forbid extra fields.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )


BaseConfigType = TypeVar("BaseConfigType", bound=BaseConfig)


def merge_config(
    config: BaseConfigType, new_fields: Dict[str, Any]
) -> BaseConfigType:
    """Merge new fields into the existing config."""

    merger = Merger(
        [(dict, "merge"), (list, "append")],
        ["override"],
        ["override"],
    )

    config_dict = config.model_dump()
    updated_config_dict = merger.merge(config_dict, new_fields)
    return config.model_validate(updated_config_dict)


def create_update_dict(field_path: str, new_value: Any) -> Dict[str, Any]:
    """Create a dictionary to update a specific field in the config."""
    field_parts = field_path.split(".")
    update_dict = {field_parts[-1]: new_value}
    for part in reversed(field_parts[:-1]):
        update_dict = {part: update_dict}
    return update_dict


def update_config_field(
    config: BaseConfigType, field_path: str, new_value: Any
) -> BaseConfigType:
    """Update a specific field in the config.
    The field_path should be a dot-separated string representing the path to the field.
    For example, "request.indexing.index_storage".
    """
    update_dict = create_update_dict(field_path, new_value)
    return merge_config(config, update_dict)
