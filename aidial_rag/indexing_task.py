from typing import List

from pydantic import BaseModel, ConfigDict

from aidial_rag.attachment_link import AttachmentLink
from aidial_rag.dial_api_client import DialApiClient
from aidial_rag.errors import InvalidDocumentError
from aidial_rag.index_storage import link_to_index_url
from aidial_rag.indexing_api import INDEX_MIME_TYPE, INDEX_MIME_TYPES_REGEX


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


def create_indexing_tasks(
    attachment_links: List[AttachmentLink],
    dial_api_client: DialApiClient,
) -> List[IndexingTask]:
    index_attachments = {
        str(attachment.reference_url): attachment.dial_link
        for attachment in attachment_links
        if _is_rag_index(attachment)
    }

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
