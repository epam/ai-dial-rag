from typing import ClassVar, List

from pydantic import BaseModel, Field


class RetrievalResults(BaseModel):
    """Results of the Dial RAG retrieval process."""

    CONTENT_TYPE: ClassVar[str] = (
        "application/vnd.aidial.rag.retrieval-results+json"
    )

    class Chunk(BaseModel):
        """Chunk of the document retrieved by the retriever."""

        attachment_url: str = Field(
            description="URL of the attached document, the chunk belongs to. "
            "Exactly matches with the `attachment.url` field in the request.",
        )

        source: str = Field(
            description="URL to the source of the chunk. The source could be a document "
            "or some part of the document, like a page or a section. The URL could have "
            "a fragment to indicate a specific part of the document, like a page number "
            "(for example, #page=3).",
        )
        source_display_name: str | None = Field(
            default=None,
            description="Human-readable name of the source (the same that Dial RAG displays in the stages).",
        )
        text: str | None = Field(
            default=None,
            description="Text of the chunk, may be empty, for example, for an image.",
        )
        page_number: int | None = Field(
            default=None,
            description="The the page number in the document, the chunk belongs to, if applicable, "
            "1-based (i.e. the same as in the page fragment of the URL).",
        )
        page_image_index: int | None = Field(
            default=None,
            description="Index of the image of the document page, the chunk belongs to, in the `images` list, 0-based.",
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

    chunks: List[Chunk] = Field(
        default_factory=list,
        description="List of chunks found by retriever in the order of their relevance.",
    )

    images: List[Image] = Field(
        default_factory=list,
        description="List of images related to the chunks.",
    )
