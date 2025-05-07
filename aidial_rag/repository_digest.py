from pathlib import Path

from pydantic import BaseModel, ValidationError


class RepositoryDigest(BaseModel):
    version: str = "unknown"
    status: str = "unknown"


def parse_repository_digest(json_str: str) -> RepositoryDigest:
    return RepositoryDigest.model_validate_json(json_str)


def read_repository_digest(file_path: str) -> RepositoryDigest:
    try:
        return parse_repository_digest(Path(file_path).read_text())
    except (FileNotFoundError, ValidationError):
        return RepositoryDigest()
