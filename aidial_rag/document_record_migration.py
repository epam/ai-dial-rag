from __future__ import annotations

import gzip
import io
import logging
import pickle
from typing import Any

from docarray import BaseDoc, DocList
from docarray.typing import ID

from aidial_rag.document_record import (
    Chunk,
    DocumentRecord,
    IndexSettings,
    MultiEmbeddings,
    deserialize_document_record,
)
from aidial_rag.errors import IndexIncompatibleError
from aidial_rag.index_record import TextIndexItem
from aidial_rag.indexing_config import IndexingConfig

MIN_FORMAT_VERSION = 11

LEGACY_SERIALIZATION_CONFIG = {"protocol": "pickle", "compress": "gzip"}

logger = logging.getLogger(__name__)


class RestrictedUnpickler(pickle.Unpickler):
    """A restricted unpickler that only allows certain classes to be unpickled
    to prevent arbitrary code execution."""

    _ALLOWED_CLASSES = {
        ("aidial_rag.document_record", "DocumentRecord"),
        ("aidial_rag.document_record", "Chunk"),
        ("aidial_rag.document_record", "IndexSettings"),
        ("aidial_rag.document_record", "ItemEmbeddings"),
        ("aidial_rag.document_record", "ModificationMetadata"),
        ("aidial_rag.index_record", "TextIndexItem"),
        ("docarray.array.any_array", "DocList[Chunk]"),
        ("docarray.array.any_array", "DocList[TextIndexItem]"),
        ("docarray.array.any_array", "DocList[ItemEmbeddings]"),
        ("docarray.typing.tensor.ndarray", "NdArray"),
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy", "dtype"),
    }

    def find_class(self, module: str, name: str) -> Any:
        if (module, name) in self._ALLOWED_CLASSES:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"Forbidden class: {module}.{name}")


class Format12:
    @staticmethod
    def deserialize(data: bytes) -> DocumentRecord:
        doc_record_bytes = gzip.decompress(data)
        doc_record_old = RestrictedUnpickler(
            io.BytesIO(doc_record_bytes)
        ).load()
        if doc_record_old.format_version != 12:
            raise ValueError(
                f"Expected format_version 12, got {doc_record_old.format_version}"
            )

        return DocumentRecord(
            format_version=doc_record_old.format_version,
            index_settings=doc_record_old.index_settings,
            chunks=doc_record_old.chunks,
            text_index=doc_record_old.text_index,
            embeddings_index=doc_record_old.embeddings_index,
            multimodal_embeddings_index=doc_record_old.multimodal_embeddings_index,
            description_embeddings_index=doc_record_old.description_embeddings_index,
            mime_type=doc_record_old.mime_type,
            document_bytes=doc_record_old.document_bytes,
        )


class Format11:
    class _MultimodalIndexSettings(BaseDoc):
        id: ID | None = (
            None  # Disable random ID generation for performance reasons
        )
        embeddings_model: str

    class _IndexSettings(BaseDoc):
        id: ID | None = (
            None  # Disable random ID generation for performance reasons
        )
        multimodal_index: Format11._MultimodalIndexSettings | None
        use_description_index: bool

    class _DocumentRecord(BaseDoc):
        format_version: int | None
        index_settings: Format11._IndexSettings
        id: ID | None = (
            None  # Disable random ID generation for performance reasons
        )
        chunks: DocList[Chunk]
        text_index: DocList[TextIndexItem] | None
        embeddings_index: MultiEmbeddings | None
        multimodal_embeddings_index: MultiEmbeddings | None = None
        description_embeddings_index: MultiEmbeddings | None = None
        mime_type: str
        original_file_name: str
        original_document: bytes

    class Unpickler(RestrictedUnpickler):
        _MODULE_MAPPING = {
            "dial_rag.document_record": "aidial_rag.document_record",
            "dial_rag.index_record": "aidial_rag.index_record",
        }

        def find_class(self, module: str, name: str) -> Any:
            if (
                module == "dial_rag.document_record"
                and name == "DocumentRecord"
            ):
                return Format11._DocumentRecord
            if module == "dial_rag.document_record" and name == "IndexSettings":
                return Format11._IndexSettings
            if (
                module == "dial_rag.document_record"
                and name == "MultimodalIndexSettings"
            ):
                return Format11._MultimodalIndexSettings
            if module in self._MODULE_MAPPING:
                module = self._MODULE_MAPPING[module]
            return super().find_class(module, name)

    @staticmethod
    def _migrate_index_settings(
        old_settings: Format11._IndexSettings,
    ) -> IndexSettings:
        # Get default value for all the fields that not supported in old format
        index_settings = IndexingConfig().collect_fields_that_rebuild_index()

        def _delete_key(d: dict, key: str):
            if key in d:
                del d[key]

        if old_settings.multimodal_index is not None:
            index_settings.indexes["multimodal_index"] = {
                "embeddings_model": old_settings.multimodal_index.embeddings_model
            }
        else:
            _delete_key(index_settings.indexes, "multimodal_index")

        if old_settings.use_description_index:
            index_settings.indexes["description_index"] = {}
        else:
            _delete_key(index_settings.indexes, "description_index")

        return IndexSettings(indexes=index_settings.indexes)

    @staticmethod
    def deserialize(data: bytes) -> DocumentRecord:
        doc_record_bytes = gzip.decompress(data)
        doc_record_old = Format11.Unpickler(io.BytesIO(doc_record_bytes)).load()

        if doc_record_old.format_version != 11:
            raise ValueError(
                f"Expected format_version 11, got {doc_record_old.format_version}"
            )

        # Migrate to latest format
        return DocumentRecord(
            format_version=doc_record_old.format_version,
            index_settings=Format11._migrate_index_settings(
                doc_record_old.index_settings
            ),
            chunks=doc_record_old.chunks,
            text_index=doc_record_old.text_index,
            embeddings_index=doc_record_old.embeddings_index,
            multimodal_embeddings_index=doc_record_old.multimodal_embeddings_index,
            description_embeddings_index=doc_record_old.description_embeddings_index,
            mime_type=doc_record_old.mime_type,
            document_bytes=doc_record_old.original_document,
        )


# Update forward references in pydantic models
Format11._DocumentRecord.model_rebuild()
Format11._IndexSettings.model_rebuild()


def deserialize_and_migrate_document_record(data: bytes) -> DocumentRecord:
    try:
        return deserialize_document_record(data)
    except IndexIncompatibleError as e:
        logger.warning(f"Failed to deserialize index: {e}")

    try:
        return Format12.deserialize(data)
    except Exception as e:
        logger.warning(f"Failed to deserialize format version 12: {e}")

    try:
        return Format11.deserialize(data)
    except Exception as e:
        logger.warning(f"Failed to deserialize format version 11: {e}")

    raise IndexIncompatibleError("Unsupported index format")
