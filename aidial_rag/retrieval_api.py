from typing import ClassVar, List

from pydantic import BaseModel, Field


class Source(BaseModel):
    """Source of the chunk, intended to show the user where the chunk comes from.
    The source could be a document or some part of the document, like a page or a section.
    """

    url: str = Field(
        description="URL for the source. The URL could have a fragment to indicate "
        "a specific part of the document, like a page number (for example, #page=3)."
    )
    display_name: str | None = Field(
        default=None,
        description="Human-readable name of the source (the same that Dial RAG displays in the stages).",
    )


class Page(BaseModel):
    """Page of the document."""

    number: int = Field(
        description="Page number in the document, 1-based (i.e. the same as in the page fragment of the URL)."
    )
    image_index: int | None = Field(
        default=None,
        description="Index of the image of the document page in the `images` list, 0-based.",
    )


class Image(BaseModel):
    """Image related to the retrieved chunk."""

    data: str = Field(
        description="Base64 encoded image data in image/png format."
    )
    mime_type: str = Field(
        default="image/png",
        description="MIME type of the image. Only image/png is supported for now.",
    )


class Chunk(BaseModel):
    """Chunk of the document retrieved by the retriever."""

    attachment_url: str = Field(
        description="URL of the attached document, the chunk belongs to. "
        "Exactly matches with the `attachment.url` field in the request.",
    )

    source: Source = Field(
        description="Source this chunk belongs to. The source could be a document "
        "or some part of the document, like a page or a section."
    )

    text: str | None = Field(
        default=None,
        description="Text of the chunk, may be empty, for example, for an image.",
    )

    page: Page | None = Field(
        default=None,
        description="Page of the document, the chunk belongs to, if applicable.",
    )


class RetrievalResponse(BaseModel):
    """Response with the results of the Dial RAG retrieval process."""

    CONTENT_TYPE: ClassVar[str] = (
        "application/x.aidial-rag.retrieval-response+json"
    )

    chunks: List[Chunk] = Field(
        default_factory=list,
        description="List of chunks found by retriever in the order of their relevance.",
    )

    images: List[Image] = Field(
        default_factory=list,
        description="List of images related to the chunks.",
    )
