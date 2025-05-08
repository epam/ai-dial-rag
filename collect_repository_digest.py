import logging
import subprocess
import sys
from pathlib import Path

from aidial_rag.repository_digest import RepositoryDigest


def _run_command(command: list[str]):
    try:
        result = subprocess.run(command, capture_output=True, check=True)  # noqa: S603
        return result.stdout.decode("utf-8").strip()
    except subprocess.CalledProcessError as e:
        logging.error("Failed to run command %s: %s", command, e.stderr.decode("utf-8"))
        raise


def get_repository_digest() -> RepositoryDigest:
    # git status returns a list of local changed files in porcelain format
    status = _run_command(["git", "status", "--porcelain", "-uno"])

    # git status should be run before git describe, because git describe --dirty may use stale data

    # git describe returns a string like "<tag>-<commits_since_tag>-g<commit_hash>[-<dirty>]"
    version = _run_command(["git", "describe", "--tags", "--dirty", "--broken", "--long", "--always"])

    return RepositoryDigest(version=version, status=status)


def save_repository_digest(digest_path: str):
    repository_digest = get_repository_digest()
    repository_digest_json = repository_digest.model_dump_json(indent=4)
    Path(digest_path).write_text(repository_digest_json)


if __name__ == "__main__":
    save_repository_digest(*sys.argv[1:])
