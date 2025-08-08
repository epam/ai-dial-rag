import importlib
import json
import sys
from typing import Type

from pydantic import BaseModel


def load_pydantic_class(pydantic_class_path: str) -> Type[BaseModel]:
    """Load a Pydantic class from a string path."""
    module, class_name = pydantic_class_path.rsplit(".", maxsplit=1)
    module_obj = importlib.import_module(module)
    if not hasattr(module_obj, class_name):
        raise ValueError(f"{class_name} not found in module {module}")

    pydantic_class = getattr(module_obj, class_name)
    if not issubclass(pydantic_class, BaseModel):
        raise ValueError(f"{class_name} is not a subclass of BaseModel")

    return pydantic_class


def generate_json_schema(pydantic_class_path: str, output_file: str):
    pydantic_class = load_pydantic_class(pydantic_class_path)
    with open(output_file, "w") as f:
        json.dump(pydantic_class.model_json_schema(), f, indent=2)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(
            "Usage: python generate_json_schema.py <pydantic_class_path> <output_file>"
        )
        sys.exit(1)
    pydantic_class_path, output_file = sys.argv[1], sys.argv[2]
    generate_json_schema(pydantic_class_path, output_file)
