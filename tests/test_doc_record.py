import importlib.metadata
from pathlib import Path

import pytest

from aidial_rag.document_record import DocumentRecord
from aidial_rag.documents import (
    deserialize_and_migrate_document_record,
)
from aidial_rag.errors import IndexIncompatibleError
from tests.utils.local_http_server import start_local_server

DATA_DIR = "tests/data"
PORT = 5007

INDEX_DIR = Path("tests/data/index")

AIDIAL_RAG_VERSION = importlib.metadata.version("aidial-rag")


@pytest.fixture
def local_server():
    with start_local_server(data_dir=DATA_DIR, port=PORT) as server:
        yield server


# To get an index file, make sure that the DIAL RAG is started with
# DIAL_RAG__INDEX_STORAGE__USE_DIAL_FILE_STORAGE=True
# and download the index file from the file storage.
@pytest.mark.parametrize(
    "index_file",
    [
        "doc_record_0.22.0.bin",  # format 11
        "doc_record_0.33.0.bin",  # format 12
        "doc_record_0.34.0rc0.bin",  # before modification_metadata
    ],
)
@pytest.mark.asyncio
async def test_load_old_indexes(index_file):
    with open(INDEX_DIR / index_file, "rb") as f:
        doc_record = deserialize_and_migrate_document_record(f.read())

    assert isinstance(doc_record, DocumentRecord)

    # Check that new fields have valid default values
    assert doc_record.modification_metadata is not None
    assert doc_record.modification_metadata.etag is None
    assert doc_record.modification_metadata.last_modified is None


def test_load_rce():
    with pytest.raises(IndexIncompatibleError):
        # Attempt to load index which tries to execute os.system('ls ~')
        # should be blocked by the whitelist in the Unpickler
        with open(INDEX_DIR / "doc_record_rce.bin", "rb") as f:
            deserialize_and_migrate_document_record(f.read())
