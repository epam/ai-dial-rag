import hashlib
from typing import List

from pydantic import BaseModel, ConfigDict

from aidial_rag.attachment_link import AttachmentLink
from aidial_rag.dial_api_client import DialApiClient
from aidial_rag.errors import InvalidAttachmentError, InvalidDocumentError
from aidial_rag.index_mime_type import INDEX_MIME_TYPE, INDEX_MIME_TYPES_REGEX


class IndexingTask(BaseModel):
    """Task for loading a document and indexing it."""

    attachment_link: AttachmentLink
    index_url: str

    model_config = ConfigDict(arbitrary_types_allowed=True)


def _is_rag_index(attachment: AttachmentLink) -> bool:
    """Check if the attachment is a RAG index."""

    if attachment.type is None:
        return False
    if not INDEX_MIME_TYPES_REGEX.match(attachment.type):
        return False
    if attachment.type != INDEX_MIME_TYPE:
        raise InvalidDocumentError(f"Unknown index type: {attachment.type}")
    if not attachment.reference_url:
        raise InvalidDocumentError("Index attachment must have a reference URL")
    return True


def link_to_index_url(attachment_link: AttachmentLink, bucket_id: str) -> str:
    # Number of characters in each directory part for index file paths
    # This is treated as a part of an algorithm, not a configuration parameter,
    # because if changed, the old index files will not be found.
    INDEX_PATH_PART_SIZE = 8

    key = hashlib.sha256(attachment_link.dial_link.encode()).hexdigest()

    # split the key into parts to avoid too many files in one directory
    dir_path = "/".join(
        key[i : i + INDEX_PATH_PART_SIZE]
        for i in range(0, len(key), INDEX_PATH_PART_SIZE)
    )

    return f"files/{bucket_id}/dial-rag-index/{dir_path}/index.bin"


def is_in_dial_rag_bucket(url: str, bucket_id: str) -> bool:
    """Check if the URL is in the Dial RAG bucket."""
    return url.startswith(f"files/{bucket_id}")


def validate_indexing_task(
    task: IndexingTask,
    dial_api_client: DialApiClient,
) -> None:
    index_url = task.index_url
    if not is_in_dial_rag_bucket(index_url, dial_api_client.bucket_id):
        # If the index URL is not in the Dial RAG bucket,
        # the Dial Core will check if RAG has an access to it.
        return

    # If the User specified index URL points to the Dial RAG bucket,
    # we have to make sure it will match the expected index path,
    # otherwise, we may overwrite the index for another document.
    expected_index_url = link_to_index_url(
        task.attachment_link, dial_api_client.bucket_id
    )
    if index_url != expected_index_url:
        raise InvalidAttachmentError(
            f"Index URL {index_url} does not match the expected index path {expected_index_url}."
        )


def create_indexing_tasks(
    attachment_links: List[AttachmentLink],
    dial_api_client: DialApiClient,
) -> List[IndexingTask]:
    index_attachments = {
        str(attachment.reference_url): attachment.dial_link
        for attachment in attachment_links
        if _is_rag_index(attachment)
    }

    # index_url validation is called in the load_documents function,
    # to have per-document error handling
    return [
        IndexingTask(
            attachment_link=link,
            index_url=(
                index_attachments.get(link.dial_link)
                or link_to_index_url(link, dial_api_client.bucket_id)
            ),
        )
        for link in attachment_links
        if not _is_rag_index(link)
    ]
