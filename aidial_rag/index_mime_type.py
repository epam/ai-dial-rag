import re

INDEX_MIME_TYPE = "application/x.aidial-rag.index.v0"
INDEX_MIME_TYPES_REGEX = re.compile(r"^application/x\.aidial-rag\.index\..*$")


assert INDEX_MIME_TYPES_REGEX.match(INDEX_MIME_TYPE), (
    f"Invalid INDEX_MIME_TYPE: {INDEX_MIME_TYPE}. "
    f"It should match the regex {INDEX_MIME_TYPES_REGEX.pattern}."
)
