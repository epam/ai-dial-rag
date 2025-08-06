from pydantic import BaseModel, ConfigDict

from aidial_rag.attachment_link import AttachmentLink


class IndexingTask(BaseModel):
    """Task for loading a document and indexing it."""

    attachment_link: AttachmentLink
    index_url: str

    model_config = ConfigDict(arbitrary_types_allowed=True)
